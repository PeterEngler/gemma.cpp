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

#include "gemma/tensor_stats.h"

#if GCPP_TENSOR_STATS
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <atomic>
#include <cmath>
#include <memory>

#include "io/io.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/profiler.h"  // StringTable

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/tensor_stats.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/dot-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

float Correlation(const float* x, size_t num) {
  double sum = 0.0;
  for (size_t i = 0; i < num; ++i) {
    sum += x[i];
  }
  const double mean = sum / static_cast<double>(num);

  double numerator = 0.0;
  double sum_sq_current = 0.0;
  double sum_sq_next = 0.0;

  for (size_t i = 0; i < num - 1; ++i) {
    const double diff_current = static_cast<double>(x[i]) - mean;
    const double diff_next = static_cast<double>(x[i + 1]) - mean;

    numerator += diff_current * diff_next;
    sum_sq_current += diff_current * diff_current;
    sum_sq_next += diff_next * diff_next;
  }

  if (sum_sq_current == 0.0 || sum_sq_next == 0.0) return 0.0f;
  const double denominator = std::sqrt(sum_sq_current * sum_sq_next);
  const float corr = static_cast<float>(numerator / denominator);
  HWY_DASSERT(-1.0f <= corr && corr <= 1.0f);
  return corr;
}

// Only write tensor data the first time it is encountered per layer. This is
// a concurrent string+layer -> flag map which avoids std::mutex (incompatible
// with fibers). We use a string table to index into per-layer atomic flags.
static bool ShouldWrite(const char* name, size_t layer_idx) {
  constexpr size_t kMaxNames = 128;
  constexpr size_t kMaxLayers = 128;
  HWY_DASSERT(layer_idx < kMaxLayers);
  static hwy::StringTable<kMaxNames> s_table;
  const size_t name_idx = s_table.Add(name);
  static std::atomic_flag flags[kMaxNames * kMaxLayers] = {};
  return !flags[name_idx * kMaxLayers + layer_idx].test_and_set(
      std::memory_order_acq_rel);
}

std::unique_ptr<File> MaybeOpenFile(size_t layer_idx, const MatPtr& type_erased,
                                    const Path& tensor_output) {
  if (tensor_output.Empty()) return nullptr;
  if (!ShouldWrite(type_erased.Name(), layer_idx)) return nullptr;
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s_L%02zu_%zux%zu_%s.bin",
           tensor_output.path.c_str(), type_erased.Name(), layer_idx,
           type_erased.Rows(), type_erased.Cols(),
           TypeName(type_erased.GetType()));
  return OpenFileOrAbort(Path(path), "wb");
}

void MaybeWriteRow(const std::unique_ptr<File>& file, const MatPtr& type_erased,
                   size_t row_idx) {
  if (!file) return;
  const size_t bytes_per_row = type_erased.Cols() * type_erased.ElementBytes();
  file->Write(type_erased.RowBytes(row_idx), bytes_per_row,
              bytes_per_row * row_idx);
}

// First dispatch to the type, then parallel over rows, then vectorized
// decompress and Notify for each value.
void UpdateStatsT(TensorStats& stats, size_t layer_idx,
                  const MatPtr& type_erased, ThreadingContext& ctx, int flags,
                  size_t cluster_idx, Parallelism parallelism) {
  std::unique_ptr<File> file =
      MaybeOpenFile(layer_idx, type_erased, ctx.tensor_output);

  if ((flags & kTensorStatsIsWeight) && layer_idx != 0) {
    // Still compute stats, but remember not to print them.
    stats.Get(layer_idx, 0).DoNotPrint();
  }

  CallUpcasted(&type_erased, [&](const auto* mat) {
    const size_t cols = mat->Cols();

    ParallelFor(
        parallelism, mat->Rows(), ctx, cluster_idx, Callers::kTensorStats,
        [&](size_t row_idx, size_t global_idx) {
          GCPP_ZONE(ctx, global_idx, Zones::kGenStats);

          auto* HWY_RESTRICT row = mat->Row(row_idx);
          MaybeWriteRow(file, type_erased, row_idx);

          using Packed = hwy::RemoveCvRef<decltype(*row)>;
          PackedSpan<Packed> packed(const_cast<Packed*>(row), cols);

          TensorStatsAccumulator& my_stats = stats.Get(layer_idx, global_idx);
          my_stats.NotifyCond(ConditionNumber(row, cols));

          namespace hn = hwy::HWY_NAMESPACE;
          hn::ScalableTag<float> df;
          using VF = hn::Vec<decltype(df)>;
          HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
          HWY_ALIGN float buf[2 * hn::MaxLanes(df)];

          size_t packed_ofs = 0;
          if (cols >= 2 * NF) {
            for (; packed_ofs <= cols - 2 * NF; packed_ofs += 2 * NF) {
              VF v0, v1;
              Decompress2(df, packed, packed_ofs, v0, v1);
              hn::Store(v0, df, buf);
              hn::Store(v1, df, buf + NF);
              const VF min_mag = hn::Min(hn::Abs(v0), hn::Abs(v1));
              const VF max_mag = hn::Max(hn::Abs(v0), hn::Abs(v1));
              const float min = hn::ReduceMin(df, min_mag);
              if (min != 0.0f) {  // Avoid division by zero.
                my_stats.NotifyGroup(min, hn::ReduceMax(df, max_mag));
              }

              for (size_t i = 0; i < 2 * NF; ++i) {
                my_stats.Notify(buf[i], row_idx, packed_ofs + i);
              }
              my_stats.NotifyCorr(Correlation(buf, 2 * NF));
            }
          }

          // Zero to two vectors remaining.
          for (; packed_ofs < cols; packed_ofs += NF) {
            const size_t remaining = HWY_MIN(NF, cols - packed_ofs);
            DecompressAndZeroPad(df, packed, packed_ofs, buf, remaining);
            // Skip NotifyGroup for this partial group.
            for (size_t i = 0; i < remaining; ++i) {
              my_stats.Notify(buf[i], row_idx, packed_ofs + i);
            }
            my_stats.NotifyCorr(Correlation(buf, remaining));
          }
        });
  });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_EXPORT(UpdateStatsT);

// Must reside in .cc file so that we can #include compress-inl.h.
void TensorStats::Notify(size_t layer_idx, const MatPtr& type_erased,
                         ThreadingContext& ctx, int flags, size_t cluster_idx,
                         Parallelism parallelism) {
  // Ignore empty tensors.
  if (type_erased.GetType() == Type::kUnknown || type_erased.Cols() == 0) {
    return;
  }
  HWY_DYNAMIC_DISPATCH(UpdateStatsT)(*this, layer_idx, type_erased, ctx, flags,
                                     cluster_idx, parallelism);
}

}  // namespace gcpp
#endif  // HWY_ONCE

#endif  // GCPP_TENSOR_STATS
