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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_

#include <stddef.h>

#include <optional>
#include <utility>
#include <vector>

#include "gemma/configs.h"     // ModelConfig
#include "gemma/gemma_args.h"  // InferenceArgs
#include "util/basics.h"       // BF16
#include "util/mat.h"
#include "hwy/base.h"

namespace gcpp {

using KV_t = BF16;

// A non-owning view of a KVCache.
struct KVCachePtr {
  bool IsEmpty() const { return kv_cache.Rows() == 0; }
  size_t SeqLen() const;

  MatPtrT<KV_t> kv_cache;
  MatPtrT<KV_t> k_cache;
  MatPtrT<KV_t> v_cache;
};

struct KVCache {
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const Allocator& allocator);
  // Returns a deep copy of the KVCache. Use explicit function instead of
  // copy ctor to make the cost explicit.
  KVCache Copy();

  size_t SeqLen() const {
    return kv_cache.Rows();
  }

  MatStorageT<KV_t> kv_cache;  // [seq_len, layers * kv_heads * qkv_dim * 2]
  // The format of k_cache indicates that there are pairs of values from
  // qkv_dim in groups of 2x kFloatsPerVector(=NF) elements from the sequence,
  // in groups of qkv_dim/2 elements in groups of kv_heads elements.
  // This enables sequential loading of the data when filling 2 vectors with
  // NF sequence elements of pairs of BF16 qkv values. The next vector then
  // continues reading the rest of qkv.
  // [seq_len / 2NF, layers * kv_heads * qkv_dim/2 * 2NF * 2]
  MatStorageT<KV_t> k_cache;
  // v_cache is formatted to allow sequential access to V during scaling and
  // update of att_out.
  // Originally [seq_len, layers * kv_heads * qkv_dim]
  // v_cache is transposed to:
  // [layers, kv_heads, seq_len, qkv_dim], reshaped to:
  // [layers, kv_heads, seq_len/(2NF), 2NF, qkv_dim/(2NF), 2NF]
  // then transposed to:
  // [seq_len/(2NF), layers, kv_heads, qkv_dim/(2NF), 2NF, 2NF]
  // and finally packed in a 2D MatStorageT as:
  // [seq_len/(2NF), layers * kv_heads * qkv_dim/(2NF) * 2NF * 2NF]
  // This allows sequential reads of 2NF registers each of 2NF BF16 values,
  // repeatedly until all of qkv_dim is read.
  MatStorageT<KV_t> v_cache;

  KVCachePtr ToPtr() {
    return KVCachePtr{
        .kv_cache = kv_cache,
        .k_cache = k_cache,
        .v_cache = v_cache,
    };
  }

 private:
  const Allocator& allocator_;

  // For use by other ctor and Copy()
  KVCache(const Extents2D& kv_extents, const Allocator& allocator);
};

inline size_t KVCachePtr::SeqLen() const {
  return kv_cache.Rows();
}

// Convenience function to create views into KVCaches.
std::vector<KVCachePtr> ToKVCachePtrs(const hwy::Span<KVCache>& kv_caches);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
