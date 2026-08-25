/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/gemma4/gemma4_config.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::tensor::examples::gemma4 {
namespace {

TEST(Gemma4ConfigTest, TestE4BConfig) {
  Config config = Config::E4B();
  EXPECT_EQ(config.vocab_size, 262144);
  EXPECT_EQ(config.embed_dim, 2560);
  EXPECT_EQ(config.hidden_dim, 10240);
  EXPECT_EQ(config.head_dim, 256);
  EXPECT_EQ(config.num_heads, 8);
  EXPECT_EQ(config.num_kv_heads, 2);
  EXPECT_EQ(config.num_layers, 42);
  EXPECT_FLOAT_EQ(config.final_logit_softcap, 30.0f);
  EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
  EXPECT_FALSE(config.attn_logits_soft_cap.has_value());
  EXPECT_EQ(config.sliding_window_size, 512);
  EXPECT_EQ(config.global_key_size, 512);
  EXPECT_TRUE(config.use_post_attn_norm);
  EXPECT_TRUE(config.use_post_ffw_norm);
  EXPECT_EQ(config.per_layer_input_dim, 256);
  EXPECT_FLOAT_EQ(config.local_base_frequency, 10000.0f);
  EXPECT_FLOAT_EQ(config.global_base_frequency, 1000000.0f);
  EXPECT_FLOAT_EQ(config.global_rope_proportion, 0.25f);
  EXPECT_FLOAT_EQ(config.local_rope_proportion, 1.0f);
  EXPECT_FLOAT_EQ(config.frac_shared_layers, 18.0f / 42.0f);
  EXPECT_TRUE(config.share_global);
  EXPECT_TRUE(config.share_local);
  EXPECT_EQ(config.attention_pattern_size, 6);

  for (size_t i = 0; i < config.num_layers; ++i) {
    if ((i + 1) % config.attention_pattern_size) {
      EXPECT_THAT(config.GetLayerType(i), Config::LayerType::kLocalSliding)
          << "Layer " << i;
    } else {
      EXPECT_THAT(config.GetLayerType(i), Config::LayerType::kGlobal)
          << "Layer " << i;
    }
  }
}

TEST(Gemma4ConfigTest, TestE2BConfig) {
  Config config = Config::E2B();
  EXPECT_EQ(config.vocab_size, 262144);
  EXPECT_EQ(config.embed_dim, 1536);
  EXPECT_EQ(config.hidden_dim, 6144);
  EXPECT_EQ(config.head_dim, 256);
  EXPECT_EQ(config.num_heads, 8);
  EXPECT_EQ(config.num_kv_heads, 1);
  EXPECT_EQ(config.num_layers, 35);
  EXPECT_FLOAT_EQ(config.final_logit_softcap, 30.0f);
  EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
  EXPECT_FALSE(config.attn_logits_soft_cap.has_value());
  EXPECT_EQ(config.sliding_window_size, 512);
  EXPECT_EQ(config.global_key_size, 512);
  EXPECT_TRUE(config.use_post_attn_norm);
  EXPECT_TRUE(config.use_post_ffw_norm);
  EXPECT_EQ(config.per_layer_input_dim, 256);
  EXPECT_FLOAT_EQ(config.local_base_frequency, 10000.0f);
  EXPECT_FLOAT_EQ(config.global_base_frequency, 1000000.0f);
  EXPECT_FLOAT_EQ(config.global_rope_proportion, 0.25f);
  EXPECT_FLOAT_EQ(config.local_rope_proportion, 1.0f);
  EXPECT_FLOAT_EQ(config.frac_shared_layers, 20.0f / 35.0f);
  EXPECT_TRUE(config.share_global);
  EXPECT_TRUE(config.share_local);
  EXPECT_EQ(config.attention_pattern_size, 5);

  for (size_t i = 0; i < config.num_layers; ++i) {
    if ((i + 1) % config.attention_pattern_size) {
      EXPECT_THAT(config.GetLayerType(i), Config::LayerType::kLocalSliding)
          << "Layer " << i;
    } else {
      EXPECT_THAT(config.GetLayerType(i), Config::LayerType::kGlobal)
          << "Layer " << i;
    }
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
