#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_

#include <stddef.h>

#include <limits>

namespace gcpp {

struct OnlineSoftmaxState {
  float max = -std::numeric_limits<float>::max() / 2.0f;
  float d = 0.0f;
};

static constexpr size_t kVTileSize4 = 4;

struct Tile4FlashState {
  OnlineSoftmaxState row_states[kVTileSize4];
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_
