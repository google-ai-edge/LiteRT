// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_NO_ABSL
#error "This test must be compiled with LITERT_NO_ABSL."
#endif

#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "litert/cc/litert_any.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/cc/options/litert_magic_number_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/cc/options/litert_samsung_options.h"
#include "litert/cc/options/litert_webnn_options.h"

static_assert(std::is_same_v<litert::StringView, std::string_view>);
static_assert(std::is_same_v<litert::Span<const int>, std::span<const int>>);
static_assert(std::is_same_v<litert::Dimensions, std::vector<int32_t>>);
static_assert(std::is_same_v<litert::Strides, std::vector<uint32_t>>);

static_assert(std::is_same_v<decltype(litert::Model::DefaultSignatureKey()),
                             std::string_view>);
static_assert(std::is_same_v<
              decltype(std::declval<litert::Model&>().GetSignatureKeys()),
              litert::Expected<std::vector<std::string_view>>>);

using TensorBufferMap =
    std::unordered_map<std::string_view, litert::TensorBuffer>;
static_assert(std::is_same_v<
              litert::FlatHashMap<litert::StringView, litert::TensorBuffer>,
              TensorBufferMap>);
static_assert(std::is_same_v<
              decltype(std::declval<litert::CompiledModel&>().Run(
                  std::declval<std::string_view>(),
                  std::declval<const TensorBufferMap&>(),
                  std::declval<const TensorBufferMap&>())),
              litert::Expected<void>>);

static_assert(std::is_same_v<
              decltype(std::declval<litert::EnvironmentOptions&>()
                           .GetOptions()),
              std::span<const litert::EnvironmentOptions::Option>>);
static_assert(std::is_same_v<
              decltype(std::declval<litert::TensorBufferRequirements&>()
                           .Strides()),
              litert::Expected<std::span<const uint32_t>>>);

int main() { return 0; }
