// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/flash_structs.h"
#include "gemma/kv_cache.h"
#include "gemma/query.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"  // kMaxQKVDim
#include "util/threading.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/flash_attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"
#include "ops/matmul-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

static constexpr float kNegInf = -std::numeric_limits<float>::max() / 64.0f;

// Updates q in place for RMSNorm and positional encoding.
void RMSNormAndPositionalEncoding(const size_t num_tokens, const QBatch& qbatch,
                                  MatPtrT<float>& q,
                                  const MatPtr& query_norm_scale,
                                  const size_t layer_idx,
                                  const AttentionActivationsPtrs& activations,
                                  ThreadingContext& ctx) {
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const float query_scale = activations.query_scale;
  const hwy::Divisor div_qbatch(qbatch.Size());
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    GCPP_ZONE(ctx, worker, Zones::kFlashAttentionRmsNormAndPositionalEncoding);
    size_t qi = div_qbatch.Remainder(task);
    size_t batch_idx = div_qbatch.Divide(task);
    for (size_t h = 0; h < layer_config.heads; ++h) {
      const size_t tq_idx = qbatch.Size() * batch_idx + qi;
      // Find the token position in the query and calculate
      // the range of cache positions to attend to.
      constexpr size_t offset = 0;  // placeholder, do not remove
      const size_t pos = qbatch.Pos(qi) + batch_idx + offset;
      float* HWY_RESTRICT q_row = q.Row(tq_idx) + h * layer_config.qkv_dim;
      // Apply rope and scaling to Q.
      if (query_norm_scale.HasPtr()) {
        CallUpcasted(&query_norm_scale, [&](const auto* weights_t) {
          RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, q_row,
                         layer_config.qkv_dim, ctx, worker);
        });
      }
      PositionalEncodingQK(q_row, layer_idx, activations, ctx, worker, pos,
                           query_scale);
    }
  };
  {
    // kHierarchical is not worth the extra sync overhead because the tasks are
    // very lightweight.
    ParallelFor(Parallelism::kFlat, num_tokens * qbatch.Size(), ctx,
                /*cluster_idx=*/0, Callers::kFlashRMSNormAndPositionalEncoding,
                func);
  }
}

// Zeroes out kVTileSize of the given vectors.
template <size_t kVTileSize, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void ZeroResults(DF df, VF& sum0, VF& HWY_MAYBE_UNUSED sum1,
                            VF& HWY_MAYBE_UNUSED sum2,
                            VF& HWY_MAYBE_UNUSED sum3,
                            VF& HWY_MAYBE_UNUSED sum4,
                            VF& HWY_MAYBE_UNUSED sum5,
                            VF& HWY_MAYBE_UNUSED sum6,
                            VF& HWY_MAYBE_UNUSED sum7) {
  sum0 = hn::Zero(df);
  if constexpr (kVTileSize >= 4) {
    sum1 = hn::Zero(df);
    sum2 = hn::Zero(df);
    sum3 = hn::Zero(df);
  }
  if constexpr (kVTileSize >= 8) {
    sum4 = hn::Zero(df);
    sum5 = hn::Zero(df);
    sum6 = hn::Zero(df);
    sum7 = hn::Zero(df);
  }
}

// Returns a tile of 1, 4 or 8 Q rows by 2NF K Q.K dot products, in float32.
// K is always pre-transposed to shape:
// [seq_len / 2kNF, layers * kv_heads * qkv_dim/2 * 2kNF * 2], where the /2, *2
// represents that pairs of qkv_dim elements are kept together to make best use
// of BF16 dot product instructions.
// Note that this version assumes that Q is float32, and not transposed, and
// HWY_NATIVE_DOT_BF16 is false.
template <size_t kVTileSize, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void QDotKTile148FloatNotNative(
    DF df, const float* HWY_RESTRICT q, const uint32_t* HWY_RESTRICT q_offsets,
    size_t half_cols, const MatPtrT<KV_t>& k, size_t pos, VF& sum00, VF& sum01,
    VF& HWY_MAYBE_UNUSED sum10, VF& HWY_MAYBE_UNUSED sum11,
    VF& HWY_MAYBE_UNUSED sum20, VF& HWY_MAYBE_UNUSED sum21,
    VF& HWY_MAYBE_UNUSED sum30, VF& HWY_MAYBE_UNUSED sum31,
    VF& HWY_MAYBE_UNUSED sum40, VF& HWY_MAYBE_UNUSED sum41,
    VF& HWY_MAYBE_UNUSED sum50, VF& HWY_MAYBE_UNUSED sum51,
    VF& HWY_MAYBE_UNUSED sum60, VF& HWY_MAYBE_UNUSED sum61,
    VF& HWY_MAYBE_UNUSED sum70, VF& HWY_MAYBE_UNUSED sum71) {
  ZeroResults<kVTileSize>(df, sum00, sum10, sum20, sum30, sum40, sum50, sum60,
                          sum70);
  ZeroResults<kVTileSize>(df, sum01, sum11, sum21, sum31, sum41, sum51, sum61,
                          sum71);
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const size_t kNF = hn::Lanes(df);
  const float* HWY_RESTRICT q_base[kVTileSize];
  for (size_t i = 0; i < kVTileSize; ++i) {
    q_base[i] = q + q_offsets[i];
  }
  const BF16* HWY_RESTRICT k_base = k.Row(pos / (2 * kNF));
  for (size_t i = 0; i < half_cols; ++i, k_base += kNF * 4) {
    // TODO(rays): Replace with decompress2.
    VBF k0_vec = hn::LoadU(dbf, k_base);
    VBF k1_vec = hn::LoadU(dbf, k_base + kNF * 2);
    VF k0_even = hn::PromoteEvenTo(df, k0_vec);
    VF k0_odd = hn::PromoteOddTo(df, k0_vec);
    VF k1_even = hn::PromoteEvenTo(df, k1_vec);
    VF k1_odd = hn::PromoteOddTo(df, k1_vec);
    VF q0_even = hn::Set(df, q_base[0][i * 2]);
    VF q0_odd = hn::Set(df, q_base[0][i * 2 + 1]);
    sum00 = hn::MulAdd(q0_even, k0_even, sum00);
    sum01 = hn::MulAdd(q0_even, k1_even, sum01);
    sum00 = hn::MulAdd(q0_odd, k0_odd, sum00);
    sum01 = hn::MulAdd(q0_odd, k1_odd, sum01);
    if constexpr (kVTileSize >= 4) {
      VF q1_even = hn::Set(df, q_base[1][i * 2]);
      VF q1_odd = hn::Set(df, q_base[1][i * 2 + 1]);
      sum10 = hn::MulAdd(q1_even, k0_even, sum10);
      sum11 = hn::MulAdd(q1_even, k1_even, sum11);
      sum10 = hn::MulAdd(q1_odd, k0_odd, sum10);
      sum11 = hn::MulAdd(q1_odd, k1_odd, sum11);
      VF q2_even = hn::Set(df, q_base[2][i * 2]);
      VF q2_odd = hn::Set(df, q_base[2][i * 2 + 1]);
      sum20 = hn::MulAdd(q2_even, k0_even, sum20);
      sum21 = hn::MulAdd(q2_even, k1_even, sum21);
      sum20 = hn::MulAdd(q2_odd, k0_odd, sum20);
      sum21 = hn::MulAdd(q2_odd, k1_odd, sum21);
      VF q3_even = hn::Set(df, q_base[3][i * 2]);
      VF q3_odd = hn::Set(df, q_base[3][i * 2 + 1]);
      sum30 = hn::MulAdd(q3_even, k0_even, sum30);
      sum31 = hn::MulAdd(q3_even, k1_even, sum31);
      sum30 = hn::MulAdd(q3_odd, k0_odd, sum30);
      sum31 = hn::MulAdd(q3_odd, k1_odd, sum31);
    }
    if constexpr (kVTileSize >= 8) {
      VF q4_even = hn::Set(df, q_base[4][i * 2]);
      VF q4_odd = hn::Set(df, q_base[4][i * 2 + 1]);
      sum40 = hn::MulAdd(q4_even, k0_even, sum40);
      sum41 = hn::MulAdd(q4_even, k1_even, sum41);
      sum40 = hn::MulAdd(q4_odd, k0_odd, sum40);
      sum41 = hn::MulAdd(q4_odd, k1_odd, sum41);
      VF q5_even = hn::Set(df, q_base[5][i * 2]);
      VF q5_odd = hn::Set(df, q_base[5][i * 2 + 1]);
      sum50 = hn::MulAdd(q5_even, k0_even, sum50);
      sum51 = hn::MulAdd(q5_even, k1_even, sum51);
      sum50 = hn::MulAdd(q5_odd, k0_odd, sum50);
      sum51 = hn::MulAdd(q5_odd, k1_odd, sum51);
      VF q6_even = hn::Set(df, q_base[6][i * 2]);
      VF q6_odd = hn::Set(df, q_base[6][i * 2 + 1]);
      sum60 = hn::MulAdd(q6_even, k0_even, sum60);
      sum61 = hn::MulAdd(q6_even, k1_even, sum61);
      sum60 = hn::MulAdd(q6_odd, k0_odd, sum60);
      sum61 = hn::MulAdd(q6_odd, k1_odd, sum61);
      VF q7_even = hn::Set(df, q_base[7][i * 2]);
      VF q7_odd = hn::Set(df, q_base[7][i * 2 + 1]);
      sum70 = hn::MulAdd(q7_even, k0_even, sum70);
      sum71 = hn::MulAdd(q7_even, k1_even, sum71);
      sum70 = hn::MulAdd(q7_odd, k0_odd, sum70);
      sum71 = hn::MulAdd(q7_odd, k1_odd, sum71);
    }
  }
}

// Loads an adjacent pair of floats, converts them to BF16, and broadcasts them
// across a vector of BF16 as alternating odd and even elements.
// hn::ReorderDemote2To(dbf, q_1_float, q_1_float); with q1_float containing
// alternating odd and even floats appears not to do this.
HWY_INLINE hn::Vec<hn::ScalableTag<BF16>> DemoteAndBroadcast2ToBF16(
    const float* HWY_RESTRICT base) {
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  VF v_even = hn::Set(df, base[0]);
  VF v_odd = hn::Set(df, base[1]);
  VF interleaved = hn::OddEven(v_odd, v_even);
  return hn::OrderedDemote2To(hn::ScalableTag<BF16>(), interleaved,
                              interleaved);
}

// Returns a tile of 1, 4 or 8 Q rows by 2NF K Q.K dot products, in float32.
// K is always pre-transposed to shape:
// [seq_len / 2kNF, layers * kv_heads * qkv_dim/2 * 2kNF * 2], where the /2, *2
// represents that pairs of qkv_dim elements are kept together to make best use
// of BF16 dot product instructions.
// Note that this version assumes that Q is float32, and not transposed, and
// HWY_NATIVE_DOT_BF16 is true.
template <size_t kVTileSize, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void QDotKTile148FloatNative(
    DF df, const float* HWY_RESTRICT q, const uint32_t* HWY_RESTRICT q_offsets,
    size_t half_cols, const MatPtrT<KV_t>& k, size_t pos, VF& sum00, VF& sum01,
    VF& HWY_MAYBE_UNUSED sum10, VF& HWY_MAYBE_UNUSED sum11,
    VF& HWY_MAYBE_UNUSED sum20, VF& HWY_MAYBE_UNUSED sum21,
    VF& HWY_MAYBE_UNUSED sum30, VF& HWY_MAYBE_UNUSED sum31,
    VF& HWY_MAYBE_UNUSED sum40, VF& HWY_MAYBE_UNUSED sum41,
    VF& HWY_MAYBE_UNUSED sum50, VF& HWY_MAYBE_UNUSED sum51,
    VF& HWY_MAYBE_UNUSED sum60, VF& HWY_MAYBE_UNUSED sum61,
    VF& HWY_MAYBE_UNUSED sum70, VF& HWY_MAYBE_UNUSED sum71) {
  ZeroResults<kVTileSize>(df, sum00, sum10, sum20, sum30, sum40, sum50, sum60,
                          sum70);
  ZeroResults<kVTileSize>(df, sum01, sum11, sum21, sum31, sum41, sum51, sum61,
                          sum71);
  VF unused = hn::Zero(df);
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const size_t kNF = hn::Lanes(df);
  const float* HWY_RESTRICT q_base[kVTileSize];
  for (size_t i = 0; i < kVTileSize; ++i) {
    q_base[i] = q + q_offsets[i];
  }
  const BF16* HWY_RESTRICT k_base = k.Row(pos / (2 * kNF));
  for (size_t i = 0; i < half_cols; ++i, k_base += kNF * 4) {
    VBF kvec0 = hn::LoadU(dbf, k_base);
    VBF kvec1 = hn::LoadU(dbf, k_base + kNF * 2);
    VBF q0_bf16 = DemoteAndBroadcast2ToBF16(q_base[0] + i * 2);
    sum00 = hn::ReorderWidenMulAccumulate(df, q0_bf16, kvec0, sum00, unused);
    sum01 = hn::ReorderWidenMulAccumulate(df, q0_bf16, kvec1, sum01, unused);
    if constexpr (kVTileSize >= 4) {
      VBF q1_bf16 = DemoteAndBroadcast2ToBF16(q_base[1] + i * 2);
      sum10 = hn::ReorderWidenMulAccumulate(df, q1_bf16, kvec0, sum10, unused);
      sum11 = hn::ReorderWidenMulAccumulate(df, q1_bf16, kvec1, sum11, unused);
      VBF q2_bf16 = DemoteAndBroadcast2ToBF16(q_base[2] + i * 2);
      sum20 = hn::ReorderWidenMulAccumulate(df, q2_bf16, kvec0, sum20, unused);
      sum21 = hn::ReorderWidenMulAccumulate(df, q2_bf16, kvec1, sum21, unused);
      VBF q3_bf16 = DemoteAndBroadcast2ToBF16(q_base[3] + i * 2);
      sum30 = hn::ReorderWidenMulAccumulate(df, q3_bf16, kvec0, sum30, unused);
      sum31 = hn::ReorderWidenMulAccumulate(df, q3_bf16, kvec1, sum31, unused);
    }
    if constexpr (kVTileSize >= 8) {
      VBF q4_bf16 = DemoteAndBroadcast2ToBF16(q_base[4] + i * 2);
      sum40 = hn::ReorderWidenMulAccumulate(df, q4_bf16, kvec0, sum40, unused);
      sum41 = hn::ReorderWidenMulAccumulate(df, q4_bf16, kvec1, sum41, unused);
      VBF q5_bf16 = DemoteAndBroadcast2ToBF16(q_base[5] + i * 2);
      sum50 = hn::ReorderWidenMulAccumulate(df, q5_bf16, kvec0, sum50, unused);
      sum51 = hn::ReorderWidenMulAccumulate(df, q5_bf16, kvec1, sum51, unused);
      VBF q6_bf16 = DemoteAndBroadcast2ToBF16(q_base[6] + i * 2);
      sum60 = hn::ReorderWidenMulAccumulate(df, q6_bf16, kvec0, sum60, unused);
      sum61 = hn::ReorderWidenMulAccumulate(df, q6_bf16, kvec1, sum61, unused);
      VBF q7_bf16 = DemoteAndBroadcast2ToBF16(q_base[7] + i * 2);
      sum70 = hn::ReorderWidenMulAccumulate(df, q7_bf16, kvec0, sum70, unused);
      sum71 = hn::ReorderWidenMulAccumulate(df, q7_bf16, kvec1, sum71, unused);
    }
  }
}

// Returns a tile of 1, 4 or 8 Q rows by 2NF K Q.K dot products, in float32.
// K is always pre-transposed to shape:
// [seq_len / 2kNF, layers * kv_heads * qkv_dim/2 * 2kNF * 2], where the /2, *2
// represents that pairs of qkv_dim elements are kept together to make best use
// of BF16 dot product instructions.
// Note that this is optimized for the case where q and k are bf16, but there is
// no native_bf16 instruction.
template <size_t kVTileSize, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void QDotKTile148BF16NotNative(
    DF df, const BF16* HWY_RESTRICT q, const uint32_t* HWY_RESTRICT q_offsets,
    size_t half_cols, const MatPtrT<KV_t>& k, size_t pos, VF& sum00, VF& sum01,
    VF& HWY_MAYBE_UNUSED sum10, VF& HWY_MAYBE_UNUSED sum11,
    VF& HWY_MAYBE_UNUSED sum20, VF& HWY_MAYBE_UNUSED sum21,
    VF& HWY_MAYBE_UNUSED sum30, VF& HWY_MAYBE_UNUSED sum31,
    VF& HWY_MAYBE_UNUSED sum40, VF& HWY_MAYBE_UNUSED sum41,
    VF& HWY_MAYBE_UNUSED sum50, VF& HWY_MAYBE_UNUSED sum51,
    VF& HWY_MAYBE_UNUSED sum60, VF& HWY_MAYBE_UNUSED sum61,
    VF& HWY_MAYBE_UNUSED sum70, VF& HWY_MAYBE_UNUSED sum71) {
  ZeroResults<kVTileSize>(df, sum00, sum10, sum20, sum30, sum40, sum50, sum60,
                          sum70);
  ZeroResults<kVTileSize>(df, sum01, sum11, sum21, sum31, sum41, sum51, sum61,
                          sum71);
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const size_t kNF = hn::Lanes(df);
  const float* HWY_RESTRICT q_base[kVTileSize];
  for (size_t i = 0; i < kVTileSize; ++i) {
    q_base[i] = reinterpret_cast<const float*>(q + q_offsets[i]);
  }
  const BF16* HWY_RESTRICT k_base = k.Row(pos / (2 * kNF));
  for (size_t i = 0; i < half_cols; ++i, k_base += kNF * 4) {
    VBF kvec0 = hn::LoadU(dbf, k_base);
    VBF kvec1 = hn::LoadU(dbf, k_base + kNF * 2);
    VBF q0 = hn::BitCast(dbf, hn::Set(df, q_base[0][i]));
    VF k0_even = hn::PromoteEvenTo(df, kvec0);
    VF k0_odd = hn::PromoteOddTo(df, kvec0);
    VF k1_even = hn::PromoteEvenTo(df, kvec1);
    VF k1_odd = hn::PromoteOddTo(df, kvec1);
    VF q0_even = hn::PromoteEvenTo(df, q0);
    sum00 = hn::MulAdd(q0_even, k0_even, sum00);
    sum01 = hn::MulAdd(q0_even, k1_even, sum01);
    VF q0_odd = hn::PromoteOddTo(df, q0);
    sum00 = hn::MulAdd(q0_odd, k0_odd, sum00);
    sum01 = hn::MulAdd(q0_odd, k1_odd, sum01);
    if constexpr (kVTileSize >= 4) {
      VBF q1 = hn::BitCast(dbf, hn::Set(df, q_base[1][i]));
      VF q1_even = hn::PromoteEvenTo(df, q1);
      sum10 = hn::MulAdd(q1_even, k0_even, sum10);
      sum11 = hn::MulAdd(q1_even, k1_even, sum11);
      VF q1_odd = hn::PromoteOddTo(df, q1);
      sum10 = hn::MulAdd(q1_odd, k0_odd, sum10);
      sum11 = hn::MulAdd(q1_odd, k1_odd, sum11);
      VBF q2 = hn::BitCast(dbf, hn::Set(df, q_base[2][i]));
      VF q2_even = hn::PromoteEvenTo(df, q2);
      sum20 = hn::MulAdd(q2_even, k0_even, sum20);
      sum21 = hn::MulAdd(q2_even, k1_even, sum21);
      VF q2_odd = hn::PromoteOddTo(df, q2);
      sum20 = hn::MulAdd(q2_odd, k0_odd, sum20);
      sum21 = hn::MulAdd(q2_odd, k1_odd, sum21);
      VBF q3 = hn::BitCast(dbf, hn::Set(df, q_base[3][i]));
      VF q3_even = hn::PromoteEvenTo(df, q3);
      sum30 = hn::MulAdd(q3_even, k0_even, sum30);
      sum31 = hn::MulAdd(q3_even, k1_even, sum31);
      VF q3_odd = hn::PromoteOddTo(df, q3);
      sum30 = hn::MulAdd(q3_odd, k0_odd, sum30);
      sum31 = hn::MulAdd(q3_odd, k1_odd, sum31);
    }
    if constexpr (kVTileSize >= 8) {
      VBF q4 = hn::BitCast(dbf, hn::Set(df, q_base[4][i]));
      VF q4_even = hn::PromoteEvenTo(df, q4);
      sum40 = hn::MulAdd(q4_even, k0_even, sum40);
      sum41 = hn::MulAdd(q4_even, k1_even, sum41);
      VF q4_odd = hn::PromoteOddTo(df, q4);
      sum40 = hn::MulAdd(q4_odd, k0_odd, sum40);
      sum41 = hn::MulAdd(q4_odd, k1_odd, sum41);
      VBF q5 = hn::BitCast(dbf, hn::Set(df, q_base[5][i]));
      VF q5_even = hn::PromoteEvenTo(df, q5);
      sum50 = hn::MulAdd(q5_even, k0_even, sum50);
      sum51 = hn::MulAdd(q5_even, k1_even, sum51);
      VF q5_odd = hn::PromoteOddTo(df, q5);
      sum50 = hn::MulAdd(q5_odd, k0_odd, sum50);
      sum51 = hn::MulAdd(q5_odd, k1_odd, sum51);
      VBF q6 = hn::BitCast(dbf, hn::Set(df, q_base[6][i]));
      VF q6_even = hn::PromoteEvenTo(df, q6);
      sum60 = hn::MulAdd(q6_even, k0_even, sum60);
      sum61 = hn::MulAdd(q6_even, k1_even, sum61);
      VF q6_odd = hn::PromoteOddTo(df, q6);
      sum60 = hn::MulAdd(q6_odd, k0_odd, sum60);
      sum61 = hn::MulAdd(q6_odd, k1_odd, sum61);
      VBF q7 = hn::BitCast(dbf, hn::Set(df, q_base[7][i]));
      VF q7_even = hn::PromoteEvenTo(df, q7);
      sum70 = hn::MulAdd(q7_even, k0_even, sum70);
      sum71 = hn::MulAdd(q7_even, k1_even, sum71);
      VF q7_odd = hn::PromoteOddTo(df, q7);
      sum70 = hn::MulAdd(q7_odd, k0_odd, sum70);
      sum71 = hn::MulAdd(q7_odd, k1_odd, sum71);
    }
  }
}

// Returns a tile of 1, 4 or 8 Q rows by 2NF K Q.K dot products, in float32.
// K is always pre-transposed to shape:
// [seq_len / 2kNF, layers * kv_heads * qkv_dim/2 * 2kNF * 2], where the /2, *2
// represents that pairs of qkv_dim elements are kept together to make best use
// of BF16 dot product instructions.
// Note that this is optimized for the case where q and k are bf16, and there is
// a native_bf16 instruction.
template <size_t kVTileSize, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void QDotKTile148BF16Native(
    DF df, const BF16* HWY_RESTRICT q, const uint32_t* HWY_RESTRICT q_offsets,
    size_t half_cols, const MatPtrT<KV_t>& k, size_t pos, VF& sum00, VF& sum01,
    VF& HWY_MAYBE_UNUSED sum10, VF& HWY_MAYBE_UNUSED sum11,
    VF& HWY_MAYBE_UNUSED sum20, VF& HWY_MAYBE_UNUSED sum21,
    VF& HWY_MAYBE_UNUSED sum30, VF& HWY_MAYBE_UNUSED sum31,
    VF& HWY_MAYBE_UNUSED sum40, VF& HWY_MAYBE_UNUSED sum41,
    VF& HWY_MAYBE_UNUSED sum50, VF& HWY_MAYBE_UNUSED sum51,
    VF& HWY_MAYBE_UNUSED sum60, VF& HWY_MAYBE_UNUSED sum61,
    VF& HWY_MAYBE_UNUSED sum70, VF& HWY_MAYBE_UNUSED sum71) {
  ZeroResults<kVTileSize>(df, sum00, sum10, sum20, sum30, sum40, sum50, sum60,
                          sum70);
  ZeroResults<kVTileSize>(df, sum01, sum11, sum21, sum31, sum41, sum51, sum61,
                          sum71);
  VF unused_sum1 = hn::Zero(df);
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const size_t kNF = hn::Lanes(df);
  const float* HWY_RESTRICT q_base[kVTileSize];
  for (size_t i = 0; i < kVTileSize; ++i) {
    q_base[i] = reinterpret_cast<const float*>(q + q_offsets[i]);
  }
  const BF16* HWY_RESTRICT k_base = k.Row(pos / (2 * kNF));
  for (size_t i = 0; i < half_cols; ++i, k_base += kNF * 4) {
    VBF k0_vec = hn::LoadU(dbf, k_base);
    VBF k1_vec = hn::LoadU(dbf, k_base + kNF * 2);
    VBF q0 = hn::BitCast(dbf, hn::Set(df, q_base[0][i]));
    sum00 = hn::ReorderWidenMulAccumulate(df, q0, k0_vec, sum00, unused_sum1);
    sum01 = hn::ReorderWidenMulAccumulate(df, q0, k1_vec, sum01, unused_sum1);
    if constexpr (kVTileSize >= 4) {
      VBF q1 = hn::BitCast(dbf, hn::Set(df, q_base[1][i]));
      sum10 = hn::ReorderWidenMulAccumulate(df, q1, k0_vec, sum10, unused_sum1);
      sum11 = hn::ReorderWidenMulAccumulate(df, q1, k1_vec, sum11, unused_sum1);
      VBF q2 = hn::BitCast(dbf, hn::Set(df, q_base[2][i]));
      sum20 = hn::ReorderWidenMulAccumulate(df, q2, k0_vec, sum20, unused_sum1);
      sum21 = hn::ReorderWidenMulAccumulate(df, q2, k1_vec, sum21, unused_sum1);
      VBF q3 = hn::BitCast(dbf, hn::Set(df, q_base[3][i]));
      sum30 = hn::ReorderWidenMulAccumulate(df, q3, k0_vec, sum30, unused_sum1);
      sum31 = hn::ReorderWidenMulAccumulate(df, q3, k1_vec, sum31, unused_sum1);
    }
    if constexpr (kVTileSize >= 8) {
      VBF q4 = hn::BitCast(dbf, hn::Set(df, q_base[4][i]));
      sum40 = hn::ReorderWidenMulAccumulate(df, q4, k0_vec, sum40, unused_sum1);
      sum41 = hn::ReorderWidenMulAccumulate(df, q4, k1_vec, sum41, unused_sum1);
      VBF q5 = hn::BitCast(dbf, hn::Set(df, q_base[5][i]));
      sum50 = hn::ReorderWidenMulAccumulate(df, q5, k0_vec, sum50, unused_sum1);
      sum51 = hn::ReorderWidenMulAccumulate(df, q5, k1_vec, sum51, unused_sum1);
      VBF q6 = hn::BitCast(dbf, hn::Set(df, q_base[6][i]));
      sum60 = hn::ReorderWidenMulAccumulate(df, q6, k0_vec, sum60, unused_sum1);
      sum61 = hn::ReorderWidenMulAccumulate(df, q6, k1_vec, sum61, unused_sum1);
      VBF q7 = hn::BitCast(dbf, hn::Set(df, q_base[7][i]));
      sum70 = hn::ReorderWidenMulAccumulate(df, q7, k0_vec, sum70, unused_sum1);
      sum71 = hn::ReorderWidenMulAccumulate(df, q7, k1_vec, sum71, unused_sum1);
    }
  }
}

// Handles NF v rows of flash attention for NF q.k dot products from one q row.
// Automatically handles masking for causal attention and different start_pos
// and last_pos values.
template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE float SingleFlashAttentionRowVector(DF df, size_t start_pos,
                                               size_t pos, size_t last_pos,
                                               VF& x, float& old_max,
                                               float& old_d) {
  if (pos < start_pos) {
    size_t mask_size = start_pos - pos;
    const VF neg_inf = hn::Neg(hn::Inf(df));
    x = hn::IfThenElse(hn::FirstN(df, mask_size), neg_inf, x);
  }
  if (pos + hn::Lanes(df) > last_pos) {
    size_t mask_size = pos <= last_pos ? last_pos + 1 - pos : 0;
    const VF neg_inf = hn::Neg(hn::Inf(df));
    x = hn::IfThenElse(hn::FirstN(df, mask_size), x, neg_inf);
  }
  float m = hn::ReduceMax(df, x);
  m = std::max(m, old_max);
  x = hn::Exp(df, hn::Sub(x, hn::Set(df, m)));
  float scale = old_d * std::exp(old_max - m);
  old_d = hn::ReduceSum(df, x) + scale;
  old_max = m;
  if (old_d > 0.0f) {
    const float one_over_d = 1.0f / old_d;
    scale *= one_over_d;
    x = hn::Mul(x, hn::Set(df, one_over_d));
  } else {
    scale = 0.0f;
    x = hn::Zero(df);
  }
  return scale;
}

// Handles 2NF v rows of flash attention for 2NF q.k dot products from 1 q row.
// Automatically handles masking for causal attention and different start_pos
// and last_pos values.
template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE float DoubleFlashAttentionRowVector(DF df, size_t start_pos,
                                               size_t pos, size_t last_pos,
                                               VF& x0, VF& x1, float& old_max,
                                               float& old_d) {
  const size_t kNF = hn::Lanes(df);
  const VF neg_inf = hn::Neg(hn::Inf(df));
  if (pos < start_pos) {
    if (pos + kNF <= start_pos) {
      x0 = neg_inf;
      size_t mask_size = start_pos - (pos + kNF);
      x1 = hn::IfThenElse(hn::FirstN(df, mask_size), neg_inf, x1);
    } else {
      size_t mask_size = start_pos - pos;
      x0 = hn::IfThenElse(hn::FirstN(df, mask_size), neg_inf, x0);
    }
  }
  if (pos + 2 * kNF > last_pos) {
    if (pos + kNF > last_pos) {
      size_t mask_size = pos <= last_pos ? last_pos + 1 - pos : 0;
      x0 = hn::IfThenElse(hn::FirstN(df, mask_size), x0, neg_inf);
      x1 = neg_inf;
    } else {
      size_t mask_size = last_pos + 1 - (pos + kNF);
      x1 = hn::IfThenElse(hn::FirstN(df, mask_size), x1, neg_inf);
    }
  }
  VF x_max = hn::Max(x0, x1);
  float m = hn::ReduceMax(df, x_max);
  m = std::max(m, old_max);
  VF m_vec = hn::Set(df, m);
  x0 = hn::Exp(df, hn::Sub(x0, m_vec));
  x1 = hn::Exp(df, hn::Sub(x1, m_vec));
  float scale = old_d * std::exp(old_max - m);
  VF x_sum = hn::Add(x0, x1);
  old_d = hn::ReduceSum(df, x_sum) + scale;
  old_max = m;
  if (old_d > 0.0f) {
    const float one_over_d = 1.0f / old_d;
    scale *= one_over_d;
    VF one_over_d_vec = hn::Set(df, one_over_d);
    x0 = hn::Mul(x0, one_over_d_vec);
    x1 = hn::Mul(x1, one_over_d_vec);
  } else {
    scale = 0.0f;
    x0 = hn::Zero(df);
    x1 = hn::Zero(df);
  }
  return scale;
}

// Reduces each of x and stores in following lanes of max (tested with float32)
template <class DF, typename T = hn::TFromD<DF>,
          class DF4 = hn::CappedTag<T, 4>, class VF4 = hn::Vec<DF4>,
          class VF = hn::Vec<DF>, typename F>
static HWY_INLINE VF4 Reduce4(DF df, VF x_0, VF x_1, VF x_2, VF x_3,
                              F reducer) {
  const DF4 df4;
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  HWY_LANES_CONSTEXPR size_t kLanes = hn::Lanes(df);
  HWY_ALIGN T x_transposed[4 * kMaxLanes];
  hn::StoreInterleaved4(x_0, x_1, x_2, x_3, df, x_transposed);
  VF x01 =
      reducer(hn::Load(df, x_transposed), hn::Load(df, x_transposed + kLanes));
  VF x23 = reducer(hn::Load(df, x_transposed + 2 * kLanes),
                   hn::Load(df, x_transposed + 3 * kLanes));
  VF x0123 = reducer(x01, x23);
  hn::Store(x0123, df, x_transposed);

  VF4 result = hn::Load(df4, x_transposed);
  for (int i = 1; i < kLanes / 4; ++i) {
    result = reducer(result, hn::Load(df4, x_transposed + i * 4));
  }
  return result;
}

// Handles Up to 4 Q rows by NF*2 timesteps of flash attention.
template <int kNumQueries, class DF, class VF = hn::Vec<DF>>
static HWY_INLINE void FlashAttentionTileStepAndApplySoftCap(
    DF df, float att_cap, float one_over_att_cap, VF& x_0_p0, VF& x_0_p1,
    VF& x_1_p0, VF& x_1_p1, VF& x_2_p0, VF& x_2_p1, VF& x_3_p0, VF& x_3_p1,
    float* HWY_RESTRICT old_max, float* HWY_RESTRICT old_d,
    float* HWY_RESTRICT scales) {
  using DF4 = hn::CappedTag<float, 4>;
  const DF4 df4;
  using VF4 = hn::Vec<DF4>;
  static_assert(kNumQueries >= 1 && kNumQueries <= 4);
  VF4 new_max = hn::Set(df4, kNegInf);
  VF max_0, max_1, max_2, max_3 = hn::Zero(df);
  max_0 = hn::Max(x_0_p0, x_0_p1);
  if constexpr (kNumQueries >= 2) {
    max_1 = hn::Max(x_1_p0, x_1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    max_2 = hn::Max(x_2_p0, x_2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    max_3 = hn::Max(x_3_p0, x_3_p1);
  }
  if constexpr (kNumQueries == 1) {
    new_max = hn::InsertLane(new_max, 0, hn::ReduceMax(df, max_0));
  } else {
    new_max = Reduce4(df, max_0, max_1, max_2, max_3,
                      [](auto a, auto b) HWY_ATTR { return hn::Max(a, b); });
  }
  if (att_cap > 0.0f) {
    VF4 cap = hn::Set(df4, att_cap);
    VF4 one_over_cap = hn::Set(df4, one_over_att_cap);
    new_max = hn::Mul(cap, hn::Tanh(df4, hn::Mul(new_max, one_over_cap)));
  }
  VF4 old_max_vf = hn::Set(df4, kNegInf);
  old_max_vf = hn::LoadU(df4, old_max);
  new_max = hn::Max(new_max, old_max_vf);
  auto changed_max = hn::Gt(new_max, hn::Set(df4, kNegInf));
  // TODO figure out what was wrong with broadcasts and change to that.
  hn::StoreU(new_max, df4, old_max);
  if constexpr (kNumQueries >= 1) {
    const VF new_max_0 = hn::Set(df, old_max[0]);
    x_0_p0 = hn::Exp(df, hn::Sub(x_0_p0, new_max_0));
    x_0_p1 = hn::Exp(df, hn::Sub(x_0_p1, new_max_0));
  }
  if constexpr (kNumQueries >= 2) {
    const VF new_max_0 = hn::Set(df, old_max[1]);
    x_1_p0 = hn::Exp(df, hn::Sub(x_1_p0, new_max_0));
    x_1_p1 = hn::Exp(df, hn::Sub(x_1_p1, new_max_0));
  }
  if constexpr (kNumQueries >= 3) {
    const VF new_max_0 = hn::Set(df, old_max[2]);
    x_2_p0 = hn::Exp(df, hn::Sub(x_2_p0, new_max_0));
    x_2_p1 = hn::Exp(df, hn::Sub(x_2_p1, new_max_0));
  }
  if constexpr (kNumQueries >= 4) {
    const VF new_max_0 = hn::Set(df, old_max[3]);
    x_3_p0 = hn::Exp(df, hn::Sub(x_3_p0, new_max_0));
    x_3_p1 = hn::Exp(df, hn::Sub(x_3_p1, new_max_0));
  }
  VF4 old_d_vf = hn::Set(df4, 0.0f);
  old_d_vf = hn::LoadU(df4, old_d);
  VF4 scale = hn::Mul(old_d_vf, hn::Exp(df4, hn::Sub(old_max_vf, new_max)));

  VF4 x_sum = hn::Zero(df4);
  if constexpr (kNumQueries == 1) {
    x_sum = hn::Set(df4, hn::ReduceSum(df, x_0_p0) + hn::ReduceSum(df, x_0_p1));
  } else {
    VF x_0_sum = hn::Add(x_0_p0, x_0_p1);
    VF x_1_sum = hn::Add(x_1_p0, x_1_p1);
    VF x_2_sum = hn::Add(x_2_p0, x_2_p1);
    VF x_3_sum = hn::Add(x_3_p0, x_3_p1);
    x_sum = Reduce4(df, x_0_sum, x_1_sum, x_2_sum, x_3_sum,
                    [](auto a, auto b) HWY_ATTR { return hn::Add(a, b); });
  }
  old_d_vf = hn::Add(scale, x_sum);
  auto non_zero_mask = hn::Gt(old_d_vf, hn::Set(df4, 0.0f));
  const VF zero = hn::Zero(df);
  const VF4 zero4 = hn::Zero(df4);
  const VF4 one_over_d =
      hn::MaskedDivOr(zero4, non_zero_mask, hn::Set(df4, 1.0f), old_d_vf);
  HWY_ALIGN float tmp_one_over_d[4];
  hn::Store(one_over_d, df4, tmp_one_over_d);
  hn::BlendedStore(old_d_vf, changed_max, df4, old_d);
  scale = hn::Mul(scale, one_over_d);
  hn::BlendedStore(scale, changed_max, df4, scales);
  if (hn::ExtractLane(old_d_vf, 0) > 0.0f && scales[0] != 1.0f) {
    const VF one_over_d_0 = hn::Set(df, tmp_one_over_d[0]);
    x_0_p0 = hn::Mul(x_0_p0, one_over_d_0);
    x_0_p1 = hn::Mul(x_0_p1, one_over_d_0);
  } else {
    x_0_p0 = zero;
    x_0_p1 = zero;
  }
  if constexpr (kNumQueries >= 2) {
    if (hn::ExtractLane(old_d_vf, 1) > 0.0f && scales[1] != 1.0f) {
      const VF one_over_d_1 = hn::Set(df, tmp_one_over_d[1]);
      x_1_p0 = hn::Mul(x_1_p0, one_over_d_1);
      x_1_p1 = hn::Mul(x_1_p1, one_over_d_1);
    } else {
      x_1_p0 = zero;
      x_1_p1 = zero;
    }
  }
  if constexpr (kNumQueries >= 3) {
    if (hn::ExtractLane(old_d_vf, 2) > 0.0f && scales[2] != 1.0f) {
      const VF one_over_d_2 = hn::Set(df, tmp_one_over_d[2]);
      x_2_p0 = hn::Mul(x_2_p0, one_over_d_2);
      x_2_p1 = hn::Mul(x_2_p1, one_over_d_2);
    } else {
      x_2_p0 = zero;
      x_2_p1 = zero;
    }
  }
  if constexpr (kNumQueries >= 4) {
    if (hn::ExtractLane(old_d_vf, 3) > 0.0f && scales[3] != 1.0f) {
      const VF one_over_d_3 = hn::Set(df, tmp_one_over_d[3]);
      x_3_p0 = hn::Mul(x_3_p0, one_over_d_3);
      x_3_p1 = hn::Mul(x_3_p1, one_over_d_3);
    } else {
      x_3_p0 = zero;
      x_3_p1 = zero;
    }
  }
}

// Implements flash attention for a strip of tiles of size 1, 4 or 8 query
// vectors by 2NF positions in K.
// It iterates through tiles in K from `params.min_start_pos / 2NF * 2NF` up to
// `params.max_last_pos` (rounded up to the nearest multiple of 2NF).
// Masking allows each row within a tile to have a different start and end
// position.
//
// @param params FlashAttentionParams containing the extent of the strip and
//   size of the tiles.
// @param q The query matrix [batch_size * q_heads, qkv_dim] in BF16 format.
// @param k Key matrix from KV cache. K is always pre-transposed to shape:
//   [seq_len / 2kNF, layers * kv_heads * qkv_dim/2 * 2kNF * 2],
//   where the /2, *2 represents that pairs of qkv_dim elements are kept
//   together to make best use of BF16 dot product instructions.
// @param v Value matrix [seq_len, qkv_dim] from KV cache.
// @param layer_idx The index of the current transformer layer.
// @param activations Attention configurations and buffers.
// @param att_out Output buffer for attention results.
// @param ctx Threading context.
// @param worker Worker thread index.
template <size_t kVTileSize, typename QType, typename KVType>
Tile4FlashState TileFlashAttention148(
    const FlashAttentionParams& params, const MatPtrT<QType>& q,
    const MatPtrT<KVType>& k, const MatPtrT<KVType>& v, const size_t layer_idx,
    const AttentionActivationsPtrs& activations, MatPtrT<float>& att_out,
    size_t qkv_dim, ThreadingContext& ctx, const size_t worker,
    AttentionImpl attention_impl) {
  constexpr Zones kZone =
      kVTileSize == 8
          ? Zones::kFlashAttentionTileFlashAttention8
          : (kVTileSize == 4 ? Zones::kFlashAttentionTileFlashAttention4
                             : Zones::kFlashAttentionTileFlashAttention1);
  GCPP_ZONE(ctx, worker, kZone);
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  float att_cap = activations.config.att_cap;
  float one_over_cap = att_cap > 0.0f ? 1.0f / att_cap : 0.0f;
  const size_t kHTileSize = 2 * hn::Lanes(df);
  float scales[kVTileSize];
  for (size_t i = 0; i < kVTileSize; ++i) {
    hwy::ZeroBytes(att_out.Row(0) + params.out_offsets[i],
                   qkv_dim * sizeof(att_out.Row(0)[0]));
  }
  Tile4FlashState state;
  size_t position = params.min_start_pos / kHTileSize * kHTileSize;
  while (position <= params.max_last_pos) {
    // Each pair of vectors covers 2NF positions in K, with up to 8 pairs of
    // vectors covering 1, 4 or 8 queries.
    VF x00, x01;
    VF HWY_MAYBE_UNUSED x10, x11;
    VF HWY_MAYBE_UNUSED x20, x21;
    VF HWY_MAYBE_UNUSED x30, x31;
    VF HWY_MAYBE_UNUSED x40, x41;
    VF HWY_MAYBE_UNUSED x50, x51;
    VF HWY_MAYBE_UNUSED x60, x61;
    VF HWY_MAYBE_UNUSED x70, x71;
    constexpr size_t kMaxNF = hn::MaxLanes(df);
    size_t v_pos[2 * kMaxNF];
    for (size_t i = 0; i < kHTileSize; ++i) {
      v_pos[i] = activations.div_seq_len.Remainder(position + i);
    }
    if constexpr (IsF32<QType>()) {
      if constexpr (HWY_NATIVE_DOT_BF16) {
        QDotKTile148FloatNative<kVTileSize>(df, q.Row(0), params.out_offsets,
                                            qkv_dim / 2, k, position, x00, x01,
                                            x10, x11, x20, x21, x30, x31, x40,
                                            x41, x50, x51, x60, x61, x70, x71);
      } else {
        QDotKTile148FloatNotNative<kVTileSize>(
            df, q.Row(0), params.out_offsets, qkv_dim / 2, k, position, x00,
            x01, x10, x11, x20, x21, x30, x31, x40, x41, x50, x51, x60, x61,
            x70, x71);
      }
    } else {
      if constexpr (HWY_NATIVE_DOT_BF16) {
        QDotKTile148BF16Native<kVTileSize>(df, q.Row(0), params.q_offsets,
                                           qkv_dim / 2, k, position, x00, x01,
                                           x10, x11, x20, x21, x30, x31, x40,
                                           x41, x50, x51, x60, x61, x70, x71);
      } else {
        QDotKTile148BF16NotNative<kVTileSize>(
            df, q.Row(0), params.q_offsets, qkv_dim / 2, k, position, x00, x01,
            x10, x11, x20, x21, x30, x31, x40, x41, x50, x51, x60, x61, x70,
            x71);
      }
    }
    if (att_cap > 0.0f) {
      // Compute tanh(x / cap) * cap, being LogitsSoftCap on the tile.
      ApplySoftCap<kVTileSize>(df, att_cap, one_over_cap, x00, x10, x20, x30,
                               x40, x50, x60, x70);
      ApplySoftCap<kVTileSize>(df, att_cap, one_over_cap, x01, x11, x21, x31,
                               x41, x51, x61, x71);
    }
    scales[0] = DoubleFlashAttentionRowVector(
        df, params.start_pos[0], position, params.last_pos[0], x00, x01,
        state.row_states[0].max, state.row_states[0].d);
    if constexpr (kVTileSize >= 4) {
      scales[1] = DoubleFlashAttentionRowVector(
          df, params.start_pos[1], position, params.last_pos[1], x10, x11,
          state.row_states[1].max, state.row_states[1].d);
      scales[2] = DoubleFlashAttentionRowVector(
          df, params.start_pos[2], position, params.last_pos[2], x20, x21,
          state.row_states[2].max, state.row_states[2].d);
      scales[3] = DoubleFlashAttentionRowVector(
          df, params.start_pos[3], position, params.last_pos[3], x30, x31,
          state.row_states[3].max, state.row_states[3].d);
      MulByConstAndAddVT4Mem(df, scales, x00, x01, x10, x11, x20, x21, x30, x31,
                             v, v_pos, params.max_last_pos + 1 - position,
                             att_out.Row(0), params.out_offsets, qkv_dim);
    } else {
      MulByConstAndAddVT1Mem(df, scales, x00, x01, v, v_pos,
                             params.max_last_pos + 1 - position, att_out.Row(0),
                             params.out_offsets, qkv_dim);
    }
    if constexpr (kVTileSize >= 8) {
      scales[4] = DoubleFlashAttentionRowVector(
          df, params.start_pos[4], position, params.last_pos[4], x40, x41,
          state.row_states[4].max, state.row_states[4].d);
      scales[5] = DoubleFlashAttentionRowVector(
          df, params.start_pos[5], position, params.last_pos[5], x50, x51,
          state.row_states[5].max, state.row_states[5].d);
      scales[6] = DoubleFlashAttentionRowVector(
          df, params.start_pos[6], position, params.last_pos[6], x60, x61,
          state.row_states[6].max, state.row_states[6].d);
      scales[7] = DoubleFlashAttentionRowVector(
          df, params.start_pos[7], position, params.last_pos[7], x70, x71,
          state.row_states[7].max, state.row_states[7].d);
      MulByConstAndAddVT4Mem(df, scales + 4, x40, x41, x50, x51, x60, x61, x70,
                             x71, v, v_pos, params.max_last_pos + 1 - position,
                             att_out.Row(0), params.out_offsets + 4, qkv_dim);
    }
    position += kHTileSize;
  }
  return state;
}

// The vertical tile size is determined by the ability to use tiling and the
// target_parallelism. In practice the possible tile sizes in order of
// preference for efficiency are 8, 4, 1. The final tile size is chosen to be
// the largest possible that allows for target_parallelism parallel tasks.
size_t GetVTileSize(size_t kNF, size_t num_head_groups, size_t num_tokens,
                    size_t total_tasks, size_t target_parallelism) {
  const size_t kMaxEqualK = num_head_groups * num_tokens;
  if (total_tasks / k8xNFVTileSize >= target_parallelism &&
      kMaxEqualK >= k8xNFVTileSize && kNF >= k8xNFVTileSize) {
    return k8xNFVTileSize;
  }
  if (total_tasks / k4xNFVTileSize >= target_parallelism &&
      kMaxEqualK >= k4xNFVTileSize && kNF >= k4xNFVTileSize) {
    return k4xNFVTileSize;
  }
  return 1;
}

// Clears and fills the params vector with FlashAttentionParams for the given
// num_tokens, target_parallelism, and layer_idx. Computes tile sizes and
// offsets for each tile to achieve target_parallelism.
void ComputeFlashParams(size_t num_tokens, const size_t target_parallelism,
                        size_t layer_idx, AttentionActivationsPtrs& activations,
                        QBatch& qbatch, AttentionImpl attention_impl,
                        std::vector<FlashAttentionParams>& params) {
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const hwy::Divisor div_qbatch(qbatch.Size());
  const size_t qkv_dim = layer_config.qkv_dim;
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t kNF = hn::Lanes(df);

  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;
  const size_t cache_layer_size = layer_config.CacheLayerSize();
  const size_t token_batch = num_tokens * div_qbatch.GetDivisor();
  const size_t total_tasks = token_batch * layer_config.heads;
  size_t kVTileSize = GetVTileSize(kNF, kHeadGroups, num_tokens, total_tasks,
                                   target_parallelism);
  // All layers should have the same number of heads.
  HWY_DASSERT(activations.div_heads.GetDivisor() == layer_config.heads);
  // To maximize adjacent tasks with the same kv matrices, task index is encoded
  // thus: [qi][kv_head][batch_idx][head_group]. Note that the head index is
  // split into kv_head and head_group, since the head_group does not affect
  // the KV matrices, and kv_head does. batch_idx does not affect the KV
  // matrices, but does affect the last position in the sequence. qi affects
  // everything.
  params.clear();
  for (uint32_t qi = 0; qi < div_qbatch.GetDivisor(); ++qi) {
    for (uint32_t kv_head = 0; kv_head < layer_config.kv_heads; ++kv_head) {
      const size_t head_offset = kv_head * qkv_dim * 2;
      const uint32_t kv_offset = layer_idx * cache_layer_size + head_offset;
      params.push_back(FlashAttentionParams{
          .qi_index = qi,
          .kv_offset = kv_offset,
      });
      for (uint32_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
        const size_t pos = qbatch.Pos(qi) + batch_idx;
        const size_t start_pos = StartPos(pos, activations.config, layer_idx);
        size_t last = pos;
        const size_t prefix_end = qbatch.PrefixEnd(qi);
        if (prefix_end > 0 && prefix_end - 1 > last) {
          // last_pos is inclusive.
          last = prefix_end - 1;
        }
        for (size_t head_group = 0; head_group < kHeadGroups; ++head_group) {
          size_t tasks_remaining = kHeadGroups - head_group +
                                   kHeadGroups * (num_tokens - 1 - batch_idx);
          // We want to fill a tile of size kVTileSize or k4xNFVTileSize if
          // smaller, otherwise everything is singles to the next head group.
          size_t tasks_required = params.back().v_tile_size < k4xNFVTileSize
                                      ? k4xNFVTileSize
                                      : kVTileSize;
          if (params.back().v_tile_size + tasks_remaining < tasks_required ||
              params.back().v_tile_size == kVTileSize) {
            // We don't have enough tasks remaining to fill a tile, or the
            // current tile is full so start new tile.
            params.push_back(FlashAttentionParams{
                .qi_index = qi,
                .kv_offset = kv_offset,
            });
          }
          const size_t head = head_group + kHeadGroups * kv_head;
          const size_t tq_idx = div_qbatch.GetDivisor() * batch_idx + qi;
          auto& param = params.back();
          size_t offset = param.v_tile_size;
          param.q_offsets[offset] = activations.q_bf.Row(tq_idx) +
                                    head * qkv_dim - activations.q_bf.Row(0);
          param.out_offsets[offset] = activations.att_out.Row(tq_idx) +
                                      head * qkv_dim -
                                      activations.att_out.Row(0);
          param.tq_idx[offset] = tq_idx;
          param.start_pos[offset] = start_pos;
          param.min_start_pos = HWY_MIN(param.min_start_pos, start_pos);
          param.last_pos[offset] = last;
          param.max_last_pos = HWY_MAX(param.max_last_pos, last);
          ++param.v_tile_size;
        }
      }
    }
  }
}

// Returns the maximum number of tiles needed for any query in the batch.
size_t GetMaxTiles(const std::vector<FlashAttentionParams>& params,
                   const size_t kHTileSize) {
  size_t max_tiles = 0;
  for (const auto& param : params) {
    size_t start = param.min_start_pos / kHTileSize;
    size_t last = param.max_last_pos / kHTileSize;
    max_tiles = HWY_MAX(last + 1 - start, max_tiles);
  }
  return max_tiles;
}

// Splits params into smaller k-strips to allow for more parallelism.
// The strips are of size num_tiles_per_task * kHTileSize.
// split_params is cleared and filled with the split tasks.
void SplitTasksByKPos(std::vector<FlashAttentionParams>& params,
                      const size_t kHTileSize, const size_t num_tiles_per_task,
                      const size_t out_stride,
                      std::vector<FlashAttentionParams>& split_params) {
  split_params.clear();
  for (auto& param : params) {
    param.split_index = split_params.size();
    size_t start = param.min_start_pos / kHTileSize;
    size_t last = param.max_last_pos / kHTileSize;
    for (size_t tile_pos = start; tile_pos <= last;
         tile_pos += num_tiles_per_task) {
      auto& split_param = split_params.emplace_back(param);
      split_param.i_of_n = (tile_pos - start) / num_tiles_per_task;
      uint32_t tile_last = (tile_pos + num_tiles_per_task) * kHTileSize - 1;
      if (tile_last < param.max_last_pos) {
        split_param.max_last_pos = tile_last;
        for (auto& last_pos : split_param.last_pos) {
          last_pos = std::min(last_pos, tile_last);
        }
      }
      uint32_t tile_start = tile_pos * kHTileSize;
      if (tile_start > param.min_start_pos) {
        split_param.min_start_pos = tile_start;
        for (auto& start_pos : split_param.start_pos) {
          start_pos = std::max(start_pos, tile_start);
        }
      }
      if (split_param.i_of_n > 0) {
        for (size_t i = 0; i < split_param.v_tile_size; ++i) {
          split_param.tq_idx[i] =
              param.tq_idx[i] * AttentionActivations::kThreadReplicationFactor +
              split_param.i_of_n - 1;
          split_param.out_offsets[i] =
              param.out_offsets[i] +
              (split_param.tq_idx[i] - param.tq_idx[i]) * out_stride;
        }
      }
    }
  }
}

// Clears and fills activations.flash_params with FlashAttentionParams for the
// given num_tokens, target_parallelism, and layer_idx. Computes tile sizes and
// offsets for each tile to achieve target_parallelism.
// If the parallelism is insufficient for this processor type, and the sequence
// length is sufficient, the tiles are upgraded to k4xNFVTileSize and the tasks
// are split along the k positions to achieve the desired parallelism.
// If splitting was required, returns that factor by which the tiles were
// upgraded, k4xNFVTileSize, otherwise returns 0.
uint32_t ComputeAndSplitFlashParams(const size_t kNF, const size_t num_tokens,
                                    const size_t target_parallelism,
                                    size_t layer_idx,
                                    AttentionActivationsPtrs& activations,
                                    QBatch& qbatch, ThreadingContext& ctx,
                                    AttentionImpl attention_impl) {
  ComputeFlashParams(num_tokens, target_parallelism, layer_idx, activations,
                     qbatch, attention_impl, activations.flash_params);
  if (activations.flash_params.size() < ctx.pools.MaxWorkers()) {
    // Insufficient parallelism for this processor type. Try splitting along the
    // k positions.
    size_t max_tiles = GetMaxTiles(activations.flash_params, kNF);
    size_t desired_tiles_per_task = hwy::DivCeil(
        activations.flash_params.size() * max_tiles, ctx.pools.MaxWorkers());
    // The cost of combining split tasks is significant, so we want a minimum
    // number of tiles per task, and we want to use k4xNFVTileSize if possible.
    constexpr size_t kMinTilesPerTask = 4;
    if (desired_tiles_per_task >= k4xNFVTileSize * kMinTilesPerTask) {
      // We can afford to use k4xNFVTileSize vertically, so recompute params.
      ComputeFlashParams(num_tokens,
                         activations.flash_params.size() / k4xNFVTileSize,
                         layer_idx, activations, qbatch, attention_impl,
                         activations.flash_params);
      desired_tiles_per_task =
          hwy::DivCeil(desired_tiles_per_task, k4xNFVTileSize);
      SplitTasksByKPos(activations.flash_params, kNF, desired_tiles_per_task,
                       activations.att_out_reps.Stride(),
                       activations.split_flash_params);
      return k4xNFVTileSize;
    }
  }
  return 0;
}

// Combines results from split tasks, processing kNumNF * NF qkv values where
// kNumNF can be 1 4 or 16. This enables the intermediate results to be held in
// registers, which speeds up the combination step significantly.
template <size_t kNumNF>
void CombineSplitTasks1416(hwy::Span<const FlashAttentionParams> params,
                           size_t tile_pos, size_t qkv_offset,
                           AttentionActivationsPtrs& activations) {
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  const size_t kNF = hn::Lanes(df);
  float overall_m = params[0].end_state.row_states[tile_pos].max;
  float overall_d = params[0].end_state.row_states[tile_pos].d;
  float* HWY_RESTRICT att_out =
      activations.att_out.Row(0) + params[0].out_offsets[tile_pos] + qkv_offset;
  VF result_0 = hn::Load(df, att_out);
  VF result_1, result_2, result_3, result_4, result_5, result_6, result_7;
  VF result_8, result_9, result_10, result_11, result_12, result_13, result_14;
  VF result_15;
  if constexpr (kNumNF > 1) {
    result_1 = hn::Load(df, att_out + kNF);
    result_2 = hn::Load(df, att_out + 2 * kNF);
    result_3 = hn::Load(df, att_out + 3 * kNF);
  }
  if constexpr (kNumNF == 16) {
    result_4 = hn::Load(df, att_out + 4 * kNF);
    result_5 = hn::Load(df, att_out + 5 * kNF);
    result_6 = hn::Load(df, att_out + 6 * kNF);
    result_7 = hn::Load(df, att_out + 7 * kNF);
    result_8 = hn::Load(df, att_out + 8 * kNF);
    result_9 = hn::Load(df, att_out + 9 * kNF);
    result_10 = hn::Load(df, att_out + 10 * kNF);
    result_11 = hn::Load(df, att_out + 11 * kNF);
    result_12 = hn::Load(df, att_out + 12 * kNF);
    result_13 = hn::Load(df, att_out + 13 * kNF);
    result_14 = hn::Load(df, att_out + 14 * kNF);
    result_15 = hn::Load(df, att_out + 15 * kNF);
  }
  for (size_t i = 1; i < params.size() && params[i].i_of_n > 0; ++i) {
    float m = params[i].end_state.row_states[tile_pos].max;
    float d = params[i].end_state.row_states[tile_pos].d;
    float new_m = std::max(overall_m, m);
    // Scale factor for existing total given the change in max.
    float old_scale = overall_d * std::exp(overall_m - new_m);
    // Scale factor for new group to add.
    float new_scale = d * std::exp(m - new_m);
    float new_d = old_scale + new_scale;
    float one_over_d = 1.0f / new_d;
    old_scale *= one_over_d;
    new_scale *= one_over_d;
    overall_m = new_m;
    overall_d = new_d;
    float* HWY_RESTRICT att_in = activations.att_out_reps.Row(0) +
                                 params[i].out_offsets[tile_pos] + qkv_offset;
    VF old_scale_vec = hn::Set(df, old_scale);
    VF new_scale_vec = hn::Set(df, new_scale);
    result_0 = hn::Mul(result_0, old_scale_vec);
    result_0 = hn::MulAdd(hn::Load(df, att_in), new_scale_vec, result_0);
    if constexpr (kNumNF > 1) {
      result_1 = hn::Mul(result_1, old_scale_vec);
      result_2 = hn::Mul(result_2, old_scale_vec);
      result_3 = hn::Mul(result_3, old_scale_vec);
      result_1 =
          hn::MulAdd(hn::Load(df, att_in + kNF), new_scale_vec, result_1);
      result_2 =
          hn::MulAdd(hn::Load(df, att_in + 2 * kNF), new_scale_vec, result_2);
      result_3 =
          hn::MulAdd(hn::Load(df, att_in + 3 * kNF), new_scale_vec, result_3);
    }
    if constexpr (kNumNF == 16) {
      result_4 = hn::Mul(result_4, old_scale_vec);
      result_5 = hn::Mul(result_5, old_scale_vec);
      result_6 = hn::Mul(result_6, old_scale_vec);
      result_7 = hn::Mul(result_7, old_scale_vec);
      result_8 = hn::Mul(result_8, old_scale_vec);
      result_9 = hn::Mul(result_9, old_scale_vec);
      result_10 = hn::Mul(result_10, old_scale_vec);
      result_11 = hn::Mul(result_11, old_scale_vec);
      result_12 = hn::Mul(result_12, old_scale_vec);
      result_13 = hn::Mul(result_13, old_scale_vec);
      result_14 = hn::Mul(result_14, old_scale_vec);
      result_15 = hn::Mul(result_15, old_scale_vec);
      result_4 =
          hn::MulAdd(hn::Load(df, att_in + 4 * kNF), new_scale_vec, result_4);
      result_5 =
          hn::MulAdd(hn::Load(df, att_in + 5 * kNF), new_scale_vec, result_5);
      result_6 =
          hn::MulAdd(hn::Load(df, att_in + 6 * kNF), new_scale_vec, result_6);
      result_7 =
          hn::MulAdd(hn::Load(df, att_in + 7 * kNF), new_scale_vec, result_7);
      result_8 =
          hn::MulAdd(hn::Load(df, att_in + 8 * kNF), new_scale_vec, result_8);
      result_9 =
          hn::MulAdd(hn::Load(df, att_in + 9 * kNF), new_scale_vec, result_9);
      result_10 =
          hn::MulAdd(hn::Load(df, att_in + 10 * kNF), new_scale_vec, result_10);
      result_11 =
          hn::MulAdd(hn::Load(df, att_in + 11 * kNF), new_scale_vec, result_11);
      result_12 =
          hn::MulAdd(hn::Load(df, att_in + 12 * kNF), new_scale_vec, result_12);
      result_13 =
          hn::MulAdd(hn::Load(df, att_in + 13 * kNF), new_scale_vec, result_13);
      result_14 =
          hn::MulAdd(hn::Load(df, att_in + 14 * kNF), new_scale_vec, result_14);
      result_15 =
          hn::MulAdd(hn::Load(df, att_in + 15 * kNF), new_scale_vec, result_15);
    }
  }
  hn::Store(result_0, df, att_out);
  if constexpr (kNumNF > 1) {
    hn::Store(result_1, df, att_out + kNF);
    hn::Store(result_2, df, att_out + 2 * kNF);
    hn::Store(result_3, df, att_out + 3 * kNF);
  }
  if constexpr (kNumNF == 16) {
    hn::Store(result_4, df, att_out + 4 * kNF);
    hn::Store(result_5, df, att_out + 5 * kNF);
    hn::Store(result_6, df, att_out + 6 * kNF);
    hn::Store(result_7, df, att_out + 7 * kNF);
    hn::Store(result_8, df, att_out + 8 * kNF);
    hn::Store(result_9, df, att_out + 9 * kNF);
    hn::Store(result_10, df, att_out + 10 * kNF);
    hn::Store(result_11, df, att_out + 11 * kNF);
    hn::Store(result_12, df, att_out + 12 * kNF);
    hn::Store(result_13, df, att_out + 13 * kNF);
    hn::Store(result_14, df, att_out + 14 * kNF);
    hn::Store(result_15, df, att_out + 15 * kNF);
  }
}

// Recombines results from split tasks, activations.att_out_reps ->
// activations.att_out. Instead of repeatedly calling MultiplyByConstAndAdd,
// which reads/writes the sum each time, the result is kept entirely in
// registers, and the task is split into 16NF, 4NF, and NF chunks, so that there
// are enough registers to hold the intermediate results.
void CombineSplitTasks(size_t qkv_dim, uint32_t tile_factor,
                       AttentionActivationsPtrs& activations,
                       ThreadingContext& ctx) {
  GCPP_ZONE(ctx, 0, Zones::kFlashAttentionCombineSplit);
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t kNF = hn::Lanes(df);
  uint32_t num_16 = qkv_dim / (16 * kNF);
  uint32_t num_4 = (qkv_dim - kNF * 16 * num_16) / (4 * kNF);
  uint32_t num_1 = hwy::DivCeil(qkv_dim - kNF * (16 * num_16 + 4 * num_4), kNF);
  uint32_t tasks_per_qkv = num_16 + num_4 + num_1;
  ParallelFor(
      Parallelism::kFlat,
      activations.flash_params.size() * tasks_per_qkv * tile_factor, ctx,
      /*cluster_idx=*/0, Callers::kFlashAttention,
      [&](size_t p, size_t worker) {
        uint32_t tile = p / tasks_per_qkv;
        uint32_t p_idx =
            activations.flash_params[tile / tile_factor].split_index;
        const auto& param = activations.split_flash_params[p_idx];
        size_t remaining_params = activations.split_flash_params.size() - p_idx;
        tile %= tile_factor;
        if (tile >= param.v_tile_size) return;
        int32_t qkv_task = p % tasks_per_qkv;
        if (qkv_task < num_16) {
          uint32_t qkv_offset = qkv_task * 16 * kNF;
          CombineSplitTasks1416<16>(
              hwy::Span<const FlashAttentionParams>(&param, remaining_params),
              tile, qkv_offset, activations);
        } else if (qkv_task < num_16 + num_4) {
          uint32_t qkv_offset = (num_16 * 16 + (qkv_task - num_16) * 4) * kNF;
          CombineSplitTasks1416<4>(
              hwy::Span<const FlashAttentionParams>(&param, remaining_params),
              tile, qkv_offset, activations);
        } else {
          uint32_t qkv_offset =
              (num_16 * 16 + num_4 * 4 + (qkv_task - num_16 - num_4)) * kNF;
          CombineSplitTasks1416<1>(
              hwy::Span<const FlashAttentionParams>(&param, remaining_params),
              tile, qkv_offset, activations);
        }
      });
}

// The nominal aim of attention is to combine 3 inputs Q[L,D], K[L,D], V[L,D]
// into a single output O[L,D].
// Conventional attention first computes A[L,L] = Q . KT
// followed by A = softmax(A) (over invididual rows).
// Then A is multiplied by V to get O[L,D].
// For each row of O, this takes a read of one row of Q L times, all of K,
// 3 write/reads of one row of A, read all of V, and read/write the one row of O
// L times. Ignoring the computation for now, and focusing just on memory,
// the one row of O takes L(4D+3) reads and L(D+3) writes.
// For the whole of Q, this is L^2(4D+3) reads and L^2(D+3) writes.
//
// Flash attention fuses these operations together, and operates on tiles of
// n Q rows x NF K positions, accumulated in n registers, where n is in
// {1, 4, 8} and NF is the number of float lanes in a register, being 16 for
// AVX3. This reduces the number of reads of Q by NF and reads of K by n. The
// softmax is converted to streaming form using the algorithm from:
// https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf,
// which eliminates the need to store A to memory. The accumulated Q.KT result
// is passed via the streaming softmax directly to the A.V step.
// To make the dot product computation more efficient, Q, K, and V are stored
// as BF16 and K is transposed to shape:
//   [seq_len / NF, layers * kv_heads * qkv_dim/2 * NF * 2], where the /2, *2
//   represents that pairs of qkv_dim elements are kept together to make best
//   use of BF16 dot product instructions, where each pair of adjacent BF16
//   values from Q and K are mul-added into a single F32 result.
//
// A further complication is that real attention is not as simple as documented
// in the paper and above. There are multiple query heads, differing KV, and
// different sequence lengths, and the difference between prefill and decode,
// so a lot of the work in FlashAttention is making sure that a collection of q
// rows with the same KV and sequence length are grouped together so that the
// largest possible tiles can be used. This is dealt with by the
// ComputeAndSplitFlashParams() function.
void FlashAttention(const size_t num_tokens, const size_t target_parallelism,
                    const size_t layer_idx, const MatPtr& query_norm_scale,
                    AttentionActivationsPtrs& activations, QBatch& qbatch,
                    ThreadingContext& ctx, AttentionImpl attention_impl) {
  GCPP_ZONE(ctx, 0, Zones::kFlashAttentionInclusive);
  RMSNormAndPositionalEncoding(num_tokens, qbatch, activations.q,
                               query_norm_scale, layer_idx, activations, ctx);
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t seq_len =
      static_cast<size_t>(activations.div_seq_len.GetDivisor());

  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t kNF = hn::Lanes(df);
  // Compress q to q_bf.
  // TODO(rays): Move this into RMSNormAndPositionalEncoding().
  ParallelFor(
      Parallelism::kWithinCluster, activations.q.Rows(), ctx,
      /*cluster_idx=*/0, Callers::kFlashAttention,
      [&](size_t row, size_t worker) {
        CompressPerThread tls;
        const hn::ScalableTag<float> df;
        CompressTraits<BF16>::Compress(
            df, activations.q.Row(row), activations.q.Cols(), tls,
            MakeSpan(activations.q_bf.Row(row), activations.q_bf.Cols()), 0);
      });
  int tile_factor =
      ComputeAndSplitFlashParams(kNF, num_tokens, target_parallelism, layer_idx,
                                 activations, qbatch, ctx, attention_impl);
  auto& params = tile_factor >= 1 ? activations.split_flash_params
                                  : activations.flash_params;
  size_t num_tasks = params.size();

  // For each head/token/query, compute fused flash Q.K, softmax and weighted V.
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    GCPP_ZONE(ctx, worker, Zones::kFlashAttentionFlashAttention);
    auto& param = params[task];
    auto& kv_cache = qbatch.KV(param.qi_index).kv_cache;
    auto& kT_cache = qbatch.KV(param.qi_index).k_cache;
    MatPtrT<KV_t> kT("k_T_view", Extents2D(hwy::DivCeil(seq_len, 2 * kNF),
                                           qkv_dim * 2 * kNF));
    kT.SetPtr(kT_cache.Row(0) + param.kv_offset * kNF, kT_cache.Stride());
    MatPtrT<KV_t> v("v_view", Extents2D(seq_len, qkv_dim));
    v.SetPtr(kv_cache.Row(0) + param.kv_offset + qkv_dim, kv_cache.Stride());
    auto& vT_cache = qbatch.KV(param.qi_index).v_cache;
    MatPtrT<KV_t> vT("v_T_view", Extents2D(hwy::DivCeil(seq_len, 2 * kNF),
                                           qkv_dim * 2 * kNF));
    vT.SetPtr(vT_cache.Row(0) + param.kv_offset * kNF, vT_cache.Stride());
    MatPtrT<float>& att_out =
        param.i_of_n == 0 ? activations.att_out : activations.att_out_reps;
    if (param.v_tile_size == k8xNFVTileSize) {
      param.end_state = TileFlashAttention148<k8xNFVTileSize>(
          param, activations.q_bf, kT, vT, layer_idx, activations, att_out,
          qkv_dim, ctx, worker, attention_impl);
    } else if (param.v_tile_size == k4xNFVTileSize) {
      param.end_state = TileFlashAttention148<k4xNFVTileSize>(
          param, activations.q_bf, kT, vT, layer_idx, activations, att_out,
          qkv_dim, ctx, worker, attention_impl);
    } else {
      param.end_state = TileFlashAttention148<1>(
          param, activations.q_bf, kT, vT, layer_idx, activations, att_out,
          qkv_dim, ctx, worker, attention_impl);
    }
  };

  {
    PROFILER_ZONE("Gen.FlashAttention.ForkJoin");
    // Full parallelism is helpful, SmallParallelFor is insufficient.
    HierarchicalParallelFor(num_tasks, ctx, Callers::kFlashAttention, func);
  }
  if (tile_factor >= 1) {
    // Run the flash attention correction on the partial outputs.
    CombineSplitTasks(qkv_dim, tile_factor, activations, ctx);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
