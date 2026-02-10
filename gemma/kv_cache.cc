// Copyright 2024 Google LLC
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

#include "gemma/kv_cache.h"

#include <stddef.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "util/mat.h"  // ZeroInit
#include "hwy/base.h"    // HWY_MAX

namespace gcpp {

// Number of rows for KV cache. Note that both rows and cols are u32, and
// the total number of elements can exceed 2^32.
static size_t CappedSeqLen(const ModelConfig& config,
                           const InferenceArgs& inference_args) {
  if (inference_args.seq_len > config.max_seq_len) {
    HWY_WARN("Capping seq_len %zu to config.max_seq_len %u.",
             inference_args.seq_len, config.max_seq_len);
    return config.max_seq_len;
  }
  return inference_args.seq_len;
}

KVCache::KVCache(const Extents2D& kv_extents, const Allocator& allocator)
    : kv_cache("kv", kv_extents, allocator, MatPadding::kOdd),
      // WARNING: the rows and cols of k_cache and v_cache will be modified
      // before use!
      // The rows will be reduced by a factor of 2xkFloatsPerVector, and the
      // cols will be increased by 2xkFloatsPerVector on first use. This is to
      // avoid making KVCache another class that has to be duplicated for each
      // machine architecture, since kFloatsPerVector is architecture dependent.
      // The change is shape is safe only if the padding is kPacked.
      k_cache("k", Extents2D(kv_extents.rows, kv_extents.cols / 2), allocator,
              MatPadding::kPacked),
      v_cache("v", Extents2D(kv_extents.rows, kv_extents.cols / 2), allocator,
              MatPadding::kPacked),
      allocator_(allocator) {}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const Allocator& allocator)
    : KVCache(
          Extents2D(CappedSeqLen(config, inference_args), config.KVCacheCols()),
          allocator) {}

KVCache KVCache::Copy() {
  KVCache copy(kv_cache.Extents(), allocator_);

  CopyMat(kv_cache, copy.kv_cache);
  CopyMat(k_cache, copy.k_cache);
  CopyMat(v_cache, copy.v_cache);
  return copy;
}

std::vector<KVCachePtr> ToKVCachePtrs(const hwy::Span<KVCache>& kv_caches) {
  std::vector<KVCachePtr> ptrs;
  ptrs.reserve(kv_caches.size());
  for (size_t i = 0; i < kv_caches.size(); ++i) {
    ptrs.push_back(kv_caches[i].ToPtr());
  }
  return ptrs;
}

}  // namespace gcpp
