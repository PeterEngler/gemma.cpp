#include <cstddef>
#include <cstring>  // strcmp
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "third_party/absl/types/span.h"
#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#ifndef HWY_DISABLED_TARGETS
// These tests aren't designed to suss out instruction set specific problems.
// Disable most targets to keep the tests fast and simple and not have to
// worry about tolerances on floating point results.
#define HWY_DISABLED_TARGETS                                                   \
  (GEMMA_DISABLED_TARGETS | HWY_AVX2 | HWY_AVX3 | HWY_AVX3_SPR | HWY_AVX3_DL | \
   HWY_AVX3_ZEN4)
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/attention_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"
#include "gemma/configs.h"
#include "util/test_util.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void FillRandom(MatPtrT<float>& mat, uint64_t seed) {
  hwy::RandomState rng(seed);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    float* row = mat.Row(r);
    for (size_t c = 0; c < mat.Cols(); ++c) {
      row[c] = static_cast<float>(RandomGaussian(rng));
    }
  }
}

void AllocateAndFillRandom(MatPtr& mat, const Allocator& allocator,
                           std::vector<MatOwner>& mat_owners, uint64_t seed) {
  if (mat.IsEmpty()) return;
  if (mat.GetType() == Type::kUnknown) {
    mat.SetType(Type::kF32);
  }
  mat_owners.emplace_back();
  mat_owners.back().AllocateFor(mat, allocator, MatPadding::kPacked);
  MatPtrT<float> mat_f32(mat);
  FillRandom(mat_f32, seed);
}

struct TestState {
  TestState() : ctx({}), env(ctx) {}
  ThreadingContext ctx;
  std::vector<MatOwner> mat_owners;
  MatMulEnv env;
};

struct TestModelState {
  TestModelState(TestState& state)
      : config(Model::GEMMA2_2B, Type::kF32, PromptWrapping::GEMMA_PT),
        tensor_info_registry(config),
        layer_config(config.layer_configs[0]),
        layers(0, layer_config, tensor_info_registry) {
    config.att_cap = 1024.0f;
    AllocateAndFillRandom(layers.qkv_einsum_w, state.ctx.allocator,
                          state.mat_owners, 42);
    AllocateAndFillRandom(layers.attn_vec_einsum_w, state.ctx.allocator,
                          state.mat_owners, 43);
    AllocateAndFillRandom(layers.gating_einsum_w, state.ctx.allocator,
                          state.mat_owners, 44);
    AllocateAndFillRandom(layers.linear_w, state.ctx.allocator,
                          state.mat_owners, 45);
    layers.Fixup(state.mat_owners, state.ctx);
  }

  ModelConfig config;
  TensorInfoRegistry tensor_info_registry;
  const LayerConfig& layer_config;
  LayerWeightsPtrs layers;
};

struct TestAttentionState {
  TestAttentionState(TestState& state, TestModelState& model_state,
                     size_t num_tokens, size_t qbatch_size,
                     AttentionImpl attention_impl)
      : num_tokens(num_tokens),
        qbatch_size(qbatch_size),
        batch_size(qbatch_size * num_tokens),
        tokens(num_tokens),
        attention_storage_(model_state.config, model_state.layer_config,
                           batch_size, num_tokens, attention_impl,
                           state.ctx.allocator, row_ptrs_),
        attention(model_state.config, num_tokens, attention_storage_) {
    for (size_t i = 0; i < qbatch_size; ++i) {
      kv_caches.emplace_back(model_state.config, inference_args,
                             state.ctx.allocator);
    }
    activations.emplace(
        runtime_config, model_state.config, runtime_config.prefill_tbatch_size,
        kv_caches[0].SeqLen(), state.env.ctx, state.env.row_ptrs);
    // Tokens don't matter, since we fill in pre_att_rms_out before calling
    // GemmaAttention.
    std::iota(tokens.begin(), tokens.end(), 1);
    for (size_t i = 0; i < qbatch_size; ++i) {
      prompts.emplace_back(tokens);
    }
    all_queries.emplace(prompts,
                        hwy::Span<KVCache>(kv_caches.data(), kv_caches.size()));
    qbatch.emplace(/*start=*/0, /*max_size=*/qbatch_size, *all_queries);
    FillRandom(attention.pre_att_rms_out, 46);
  }

  const size_t num_tokens;
  const size_t qbatch_size;
  const size_t batch_size;
  InferenceArgs inference_args;
  RuntimeConfig runtime_config;
  std::vector<KVCache> kv_caches;
  std::optional<Activations> activations;
  std::vector<int> tokens;
  std::vector<PromptTokens> prompts;
  std::optional<AllQueries> all_queries;
  std::optional<QBatch> qbatch;
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> row_ptrs_;
  AttentionActivations attention_storage_;
  AttentionActivationsPtrs attention;
};

auto GetFloatNearMatcher() {
  // TODO(mstoll): Consider removing this function and just disabling targets.
  if (strcmp(hwy::TargetName(HWY_TARGET), "EMU128") == 0) {
    // Do quasi exact match for EMU128 since we want to use it for golden value
    // update.
    return testing::FloatNear(1e-7f);
  }
  return testing::FloatEq();  // Note: this does a 4ULP fuzzy match
}

template <size_t kNumTokens, size_t kQBatchSize, size_t kDims>
void CompareAttSumsWithGolden(
    const AttentionActivationsPtrs& attention,
    const float (&golden)[kNumTokens][kQBatchSize][kDims]) {
  ASSERT_EQ(attention.att_sums.Rows(), kNumTokens * kQBatchSize);
  ASSERT_LE(kDims, attention.att_sums.Cols());

  auto float_near = GetFloatNearMatcher();
  hwy::AlignedFreeUniquePtr<float[]> actual_row =
      hwy::AllocateAligned<float>(kDims);
  for (size_t token_idx = 0; token_idx < kNumTokens; ++token_idx) {
    for (size_t qi = 0; qi < kQBatchSize; ++qi) {
      const size_t i = token_idx * kQBatchSize + qi;
      for (size_t j = 0; j < kDims; ++j) {
        actual_row[j] = hwy::F32FromBF16(attention.att_sums.Row(i)[j]);
      }
      EXPECT_THAT(
          absl::MakeSpan(actual_row.get(), kDims),
          testing::Pointwise(float_near,
                             absl::MakeSpan(golden[token_idx][qi], kDims)));
    }
  }
}

template <size_t kNumTokens, size_t kQBatchSize, size_t kDims>
void CompareKVCacheWithGolden(
    const ModelConfig& config, hwy::Span<KVCache> kv_caches, const size_t layer,
    const size_t kv_head,
    const float (&k_golden)[kNumTokens][kQBatchSize][kDims],
    const float (&v_golden)[kNumTokens][kQBatchSize][kDims]) {
  const size_t qbatch_size = kv_caches.size();
  ASSERT_EQ(kQBatchSize, qbatch_size);
  const size_t start_offset = 0;
  const size_t qkv_dim = config.layer_configs[0].qkv_dim;

  auto float_near = GetFloatNearMatcher();

  hwy::AlignedFreeUniquePtr<float[]> actual_k_row =
      hwy::AllocateAligned<float>(kDims);
  hwy::AlignedFreeUniquePtr<float[]> actual_v_row =
      hwy::AllocateAligned<float>(kDims);

  const size_t cache_layer_size = config.layer_configs[layer].CacheLayerSize();
  const size_t head_offset = kv_head * qkv_dim * 2;
  const size_t kv_offset = layer * cache_layer_size + head_offset;

  for (size_t token_idx = 0; token_idx < kNumTokens; ++token_idx) {
    for (size_t qi = 0; qi < kQBatchSize; ++qi) {
      const float* cache_row =
          kv_caches[qi].kv_cache.Row(start_offset + token_idx);
      for (size_t j = 0; j < kDims; ++j) {
        actual_k_row[j] = cache_row[kv_offset + j];
        actual_v_row[j] = cache_row[kv_offset + qkv_dim + j];
      }
      EXPECT_THAT(
          absl::MakeSpan(actual_k_row.get(), kDims),
          testing::Pointwise(float_near,
                             absl::MakeSpan(k_golden[token_idx][qi], kDims)))
          << "K cache mismatch for qi=" << qi << " token_idx=" << token_idx;
      EXPECT_THAT(
          absl::MakeSpan(actual_v_row.get(), kDims),
          testing::Pointwise(float_near,
                             absl::MakeSpan(v_golden[token_idx][qi], kDims)))
          << "V cache mismatch for qi=" << qi << " token_idx=" << token_idx;
    }
  }
}

template <size_t kNumTokens, size_t kQBatchSize, size_t kDims>
void CompareQVecsWithGolden(
    const ModelConfig& config, const AttentionActivationsPtrs& attention,
    const size_t head,
    const float (&q_golden)[kNumTokens][kQBatchSize][kDims]) {
  const size_t qkv_dim = config.layer_configs[0].qkv_dim;

  auto float_near = GetFloatNearMatcher();

  hwy::AlignedFreeUniquePtr<float[]> actual_q_row =
      hwy::AllocateAligned<float>(kDims);

  const size_t head_offset = head * qkv_dim;
  for (size_t token_idx = 0; token_idx < kNumTokens; ++token_idx) {
    for (size_t qi = 0; qi < kQBatchSize; ++qi) {
      const size_t i = token_idx * kQBatchSize + qi;
      const float* q_row = attention.q.Row(i);
      for (size_t j = 0; j < kDims; ++j) {
        actual_q_row[j] = q_row[head_offset + j];
      }
      EXPECT_THAT(
          absl::MakeSpan(actual_q_row.get(), kDims),
          testing::Pointwise(float_near,
                             absl::MakeSpan(q_golden[token_idx][qi], kDims)))
          << "Q vec mismatch for token_idx=" << token_idx << " qi=" << qi
          << " head=" << head;
    }
  }
}

const size_t kNumTokens = 10;
const size_t kQBatchSize = 2;

// Only compare the first 10 dimensions of the attention output to keep the
// test manageable.  It's unlikely that the first few dimensions would have
// different behavior than the later dimensions.
const size_t kDimsToCompare = 10;

// Use a Coding Agent to update the golden values from test
// failure output (which includes the new values).
// Instructions to the Coding Agent: You must never update the
// EMU128 tolerance. Always use the value from the EMU128 test to update the
// Goldens.  If necessary, add relaxed tolerance for other instruction sets.

// Layer 0
const float kGoldenAttSums[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{46.5, 56.5, 10.0625, 65.5, -2.109375, 135, 15.8125, 51, -100, 52.5},
     {-30.375, -17.875, 51.75, -78, -84, 6.40625, 15.375, 70, -22.875, 20.125}},
    {{-32.75, 38.25, 78.5, 107.5, 20.25, 197, -136, 42.5, -84, 25.625},
     {-18.5, -18, 135, -13.4375, -6.625, -45.75, 29.625, 93, 18.625, 75.5}},
    {{-16.375, -61.5, -58.25, -27.375, -28, 71, -109.5, 60.25, 3.125, -29.125},
     {-19, -16.75, 129, 0.59765625, -82, 123.5, 60.75, -36.75, -77, 26.625}},
    {{3.984375, 83, -41.75, 39.5, -203, 110, -76, 131, 0.4609375, -44.5},
     {-47, -19.5, 58, 81.5, 21.75, -30, -118, 44.25, -149, 22.5}},
    {{64, -31, -89, -92.5, -11.1875, -54.75, -302, 3.453125, -108, 39.25},
     {7.6875, -80, -40, 32.25, -30.25, 90, -41, 44.25, -140, -2.4375}},
    {{39.75, 17.875, 115, 38.75, -44, 139, -53.25, -23.875, -13.0625, 38.5},
     {143, 249, 5.09375, 0.83984375, 27.875, -5.84375, 30.25, -101.5, 65.5,
      13.5}},
    {{-30.125, -169, -150, 58, -35.75, 22.75, 36.5, -32.25, -8.9375, 55.25},
     {137, 5.25, 61.25, 37, -42.75, 240, 62, -164, 11.3125, 173}},
    {{-103, -47.5, 39, -48, -67.5, 121, -136, 99, 80, -47.5},
     {29.875, 7.34375, -36.75, -14.5, -27.5, 44.75, -67.5, -40.75, 71.5, 172}},
    {{-37.25, 109.5, -26.125, -115.5, 108, 57.25, 1.3671875, 72, -122.5, 59.25},
     {40.25, 53.25, -142, 78.5, 38, 4.3125, -27.75, -134, -85, 107.5}},
    {{-8.4375, -35, -35.5, 131, -33.25, 106, 109.5, -92, -135, 80},
     {-77, 40.75, -10.125, 33.25, -33, 104, -7.6875, 85.5, -40, 93}},
};

// Layer 0, *K*V Head 0
const float kGoldenK[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{-4.51717567, 6.93118095, 6.48003578, 9.12825584, 2.38755274, 11.8121576,
      1.65376127, 5.04456615, -7.19549274, 2.57609844},
     {0.152208567, 3.14520073, -8.35154343, 5.44226503, -6.74000502,
      -1.43484437, -4.72092056, -9.48932, -6.12409401, -1.55352509}},
    {{4.378829, 5.05565643, -7.63948059, -5.74608946, 2.90109587, 0.155819178,
      4.56115055, 1.37885749, 1.48427355, -1.07145202},
     {-0.702698231, 1.49563932, 6.42149782, -6.68306589, 1.85317755,
      -7.70267582, 2.07357907, -7.60303402, -0.514724255, 0.308567047}},
    {{-1.20844436, 4.14724302, 6.04515219, 8.7753458, -0.975198627, 0.564640105,
      5.39941597, 4.64036179, 0.366614938, 3.48258138},
     {4.57078934, -4.60315752, -3.3364439, 1.29875994, -3.40833569, -6.95262,
      -6.39040232, -6.60212612, 6.63269806, -0.815209687}},
    {{0.602011144, 2.22505236, 3.62411499, -4.07026958, 12.8036356, 3.76139069,
      6.99502087, 7.02500725, -2.51568675, 4.2489934},
     {-6.37204599, -3.34989691, 2.10935307, 4.23634195, 5.79134035, 13.502944,
      -2.19158888, -1.55771351, -1.22244942, 3.36499929}},
    {{-8.85739231, -4.08585882, -0.618261, 6.52911091, 5.14922285, 7.6869874,
      0.750387549, -0.812200725, 2.7509625, 6.29693508},
     {-8.2555, 2.84032059, -1.03791106, 2.07648611, -4.94546843, 1.76888537,
      -1.75901175, 11.2628574, 1.41086221, -3.58669901}},
    {{0.919141412, 1.97533965, -11.3202848, -3.3137629, -4.7161727, 5.07012081,
      1.76256621, 8.20588207, 6.05700159, -3.89765406},
     {9.11918, 2.11261511, -5.87290621, 11.6033278, -4.66597795, -7.13774204,
      -9.10563755, -2.48294282, 3.35282946, -3.75122213}},
    {{4.17705393, -4.95192289, -10.5068378, 3.90004015, -3.51306129, 5.38068056,
      0.901511431, 11.222868, 2.67285442, 9.18779},
     {3.61795235, -7.00262165, 2.08284521, -6.70515728, 1.93205631, 2.84467721,
      3.94591737, -6.18882942, -1.78465152, -9.39100933}},
    {{-0.361971855, -1.57735932, 5.07296801, -1.55669761, -1.44996238,
      7.29838896, 5.23075104, -0.512441278, -3.59834242, 2.38584423},
     {-0.795628309, 7.30230665, -1.71035647, -16.6999454, 3.05102086,
      -4.9243927, 4.28508186, -0.694577456, 6.58464718, 4.40330124}},
    {{2.57555652, -3.5651083, 0.784440041, -4.7043705, 2.37520599, -3.62385964,
      -3.48913693, -7.28049421, -5.48726082, 1.95519221},
     {5.56260443, -5.7683115, 1.26402235, -17.507719, 4.18873024, -3.20694613,
      -4.42512083, 1.78077614, -3.25167561, 0.864362717}},
    {{-9.43858051, 0.391518891, -2.74012518, 4.9842453, 7.48263216, -16.3434925,
      -4.75156116, -1.99114823, 3.99918842, -5.95400572},
     {-1.22347319, 9.57339382, -1.31736016, -5.02770805, -4.81617355,
      -1.96618557, -0.456317186, 12.6451035, -1.50221801, 6.7991147}},
};

// Layer 0, K*V* Head 0
const float kGoldenV[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{2.77553034, -7.67514181, -1.60433948, 4.67795134, -1.75084186, 8.57896423,
      -1.15065813, -3.75088787, -4.7442131, -1.68890858},
     {-4.79950905, -1.66658688, 4.14471292, -4.95649052, -5.4200325, 3.52626801,
      -10.9432049, 0.338347554, -1.53204226, 0.473476171}},
    {{-10.6566734, 4.12785721, 4.54053593, -1.39667869, -1.55028772, 0.20508635,
      -0.00620913506, 2.93214, -0.788117647, 2.78032446},
     {-0.322291255, 2.63848567, -2.30808377, -13.0153809, 2.74378228,
      3.21460533, 0.688529968, 2.37544608, 6.06825066, 4.57566404}},
    {{-3.34342527, 6.03099537, 6.335958, 0.993818045, 0.905343294, 6.93058586,
      3.9635396, 10.8044815, 7.8620863, -10.1157322},
     {16.416317, -1.62025332, 2.3161006, 3.32571959, -1.79581594, -10.2925539,
      -5.86338425, -6.36642933, 9.18872166, 5.95524168}},
    {{6.58254147, -6.96165133, -4.97437, -2.33467388, 5.83671236, -0.794236898,
      -2.03117108, -3.93387103, -5.96872902, 5.83316422},
     {7.13243389, -8.04649162, 2.53055143, 2.0771277, -0.667295456, -13.0285645,
      0.960428238, -2.11983275, 8.18105602, -6.72609901}},
    {{9.52539158, -4.3840766, -6.94514465, -2.75913763, -10.8364506,
      -3.95606327, 2.43603897, -5.78482246, -0.801304817, 8.23436832},
     {6.89768124, 2.36394405, -2.08569574, -0.682706833, 3.38872, -6.28313875,
      4.79594612, 4.93417454, -6.40791416, -10.7355442}},
    {{1.64614546, -3.93421197, -0.48935914, 5.48284435, -7.69781828, 11.8203125,
      1.81672478, -1.42535269, -5.26496315, -5.31612349},
     {7.52353811, 3.56836724, 0.414305687, 0.340799928, 2.44263697, 7.52111912,
      0.246491909, -11.1172791, -3.82061529, 3.24794388}},
    {{-2.48950672, -8.55112743, 8.04663277, -5.77116871, -0.637019753,
      -7.65882111, -7.49037457, 3.8041625, -3.57038307, 9.37715435},
     {-2.95554614, -5.18210888, 1.00015664, -4.03864431, -7.14954519,
      5.99929142, 5.86350155, 2.03810191, -4.23009968, 9.39885902}},
    {{2.86876941, -0.836064458, -0.374509573, -0.277966499, 3.20654631,
      -3.68510771, -7.76134634, 2.23905277, -8.35530376, 5.25071716},
     {-4.37938213, 4.78577232, 2.03453469, 5.48564529, -1.05589461, -1.65940428,
      4.0130887, 5.26074123, 4.67537832, 0.791350365}},
    {{0.289200783, 7.06031752, -1.15099299, -5.29136801, -1.343642, -8.36283112,
      4.13158274, -1.93137062, 3.16199875, 2.21854591},
     {-3.04716778, -2.52233481, -5.99031925, 2.80152273, 0.340899587,
      0.667474508, -2.39674735, 8.83768654, -5.45613146, -1.55994594}},
    {{-3.1383903, -7.71573353, 3.38072681, 6.07642221, -2.39587545, -7.84178352,
      -1.60108304, -8.6121521, -5.151721, 4.17612457},
     {4.77026033, -5.51171303, -7.38155365, -5.38462543, 2.95842505, 5.18372536,
      0.521988213, 7.23966122, -4.90852165, 7.18465281}},
};

// Layer 0, QHead 0
const float kGoldenQ[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{-0.574402, 0.370210946, -0.426894128, -0.543187499, -0.0266762525,
      -0.177960426, -0.00839614868, 0.411925435, 0.536462843, 0.528389931},
     {0.270543516, -0.109283224, -0.58602041, -0.358663559, -0.393124342,
      -0.0895933211, -0.632167816, 0.386703, 0.314152211, 0.0554139167}},
    {{-0.0331106335, -0.100827977, 0.322449774, 0.225943685, -0.384854138,
      -0.208085626, 0.0206767023, 0.287796348, -0.139513299, 0.255447835},
     {-0.026678158, -0.453293741, 0.560033202, 0.105156109, -0.35259974,
      0.711447656, -0.253611118, 0.0487166606, -0.0861926, -0.0338740163}},
    {{-0.321974963, -0.466039389, 0.207254, -0.126807243, -0.192775548,
      -0.0953653902, 0.209789664, 0.405356169, -0.00627979636, -0.059096083},
     {-0.345578849, 0.394073665, 0.299743384, -0.00751776947, -0.288939655,
      0.127782926, -0.20755057, 0.0655022934, -0.705084443, -0.241842657}},
    {{0.216089681, 0.0918798074, 0.0560657121, -0.157523692, -0.00141696166,
      0.51770097, 0.596379519, -0.271057934, 0.241035387, -0.275827795},
     {-0.0744233802, 0.180814296, 0.170143232, -0.337861359, -0.175804436,
      0.213403717, -0.173699334, 0.109528266, -0.385727286, 0.109684013}},
    {{-0.350524545, -0.142550975, -0.212269634, -0.0589753427, -0.434021264,
      0.384472728, 0.445421219, -0.635599554, -0.246593416, 0.120986834},
     {-0.227436215, 0.357608676, -0.253396749, -0.0683219433, -0.179259315,
      0.236576155, 0.559984267, 0.165754244, -0.0402980633, -0.101906672}},
    {{-0.246389568, -0.266164333, -0.0967710093, -0.401160359, 0.242542222,
      0.0869856179, 0.201580375, 0.207793966, -0.0875666812, -0.242263734},
     {0.703284621, 0.0382430181, 0.43997851, -0.858277559, 0.342218578,
      0.414044619, 0.403636098, -0.579880178, -1.12243, -0.112913512}},
    {{0.138267457, 0.483219147, 0.230450034, -0.568304598, 0.204461277,
      -0.286731184, -0.416590065, -0.483460307, -0.561008453, 0.395195067},
     {0.282603294, 0.0723766834, -0.206548199, 0.561849773, 0.482716858,
      0.135281503, -0.438842, 0.472577244, -0.346201897, -0.0211652946}},
    {{-0.0537007526, -0.227346301, -0.28714636, 0.24774681, -0.0975415707,
      -0.0123391608, 0.0612513833, -0.374673873, 0.283457726, 0.40945363},
     {0.0688965, -0.149037778, -0.246169269, 0.0599289, -0.456733376,
      0.0808929652, 0.115154937, 0.0997389406, -0.408117682, 0.576600909}},
    {{0.402076691, -0.11815206, 0.542395, 0.0382412523, -0.614984, 0.286177,
      0.318540573, -0.299300939, -0.177486524, 0.394140184},
     {0.105372488, -0.145784974, 0.0695323348, -0.106080391, -0.755512118,
      0.975362539, -0.15056029, 0.58882606, -0.059625227, -0.810613}},
    {{0.243249148, 0.0904035419, -0.472183734, -0.176162, 0.314925164,
      -0.191137731, 0.492265761, -0.0120046511, 0.824757636, 0.298175},
     {0.0523641109, 0.224086404, 0.0143201668, 0.0090854, 0.304901183,
      -0.391372293, 0.267655343, 0.117368169, 0.645064473, 0.336050332}},
};

void RunAttentionTest(AttentionImpl attention_impl) {
  TestState state;
  TestModelState model_state(state);
  TestAttentionState attention_state(state, model_state, kNumTokens,
                                     kQBatchSize, attention_impl);

  GemmaAttention(attention_state.tokens.size(), 0, model_state.layers,
                 attention_state.attention, *attention_state.qbatch, state.env,
                 AttentionImplToFlags(attention_impl, HWY_NATIVE_DOT_BF16));

  CompareAttSumsWithGolden(attention_state.attention, kGoldenAttSums);
  CompareKVCacheWithGolden(model_state.config,
                           hwy::Span<KVCache>(attention_state.kv_caches.data(),
                                              attention_state.kv_caches.size()),
                           /*layer=*/0, /*kv_head=*/0, kGoldenK, kGoldenV);
  CompareQVecsWithGolden(model_state.config, attention_state.attention,
                         /*head=*/0, kGoldenQ);
}

void TestGemmaAttentionOld() { RunAttentionTest(AttentionImpl::kOld); }

void TestGemmaAttentionFlash() { RunAttentionTest(AttentionImpl::kFlash); }

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(AttentionTest);
HWY_EXPORT_AND_TEST_P(AttentionTest, TestGemmaAttentionOld);
HWY_EXPORT_AND_TEST_P(AttentionTest, TestGemmaAttentionFlash);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
