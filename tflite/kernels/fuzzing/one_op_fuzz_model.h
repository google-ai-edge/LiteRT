/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_KERNELS_FUZZING_ONE_OP_FUZZ_MODEL_H_
#define TENSORFLOW_LITE_KERNELS_FUZZING_ONE_OP_FUZZ_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "tflite/core/c/common.h"
#include "tflite/core/interpreter.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace fuzzing {

struct RuntimeTensor {
  int tensor_index;
  std::optional<std::vector<int32_t>> shape;
  std::optional<std::vector<uint8_t>> data;
};

struct OneOpModelSpec {
  const char* description = "one_op_fuzz";
  BuiltinOperator builtin_operator = BuiltinOperator_ADD;
  int version = 1;
  BuiltinOptions builtin_options_type = BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options;
  BuiltinOptions2 builtin_options_2_type = BuiltinOptions2_NONE;
  flatbuffers::Offset<void> builtin_options_2;
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  std::vector<flatbuffers::Offset<Buffer>> buffers;
  std::vector<int32_t> model_inputs;
  std::vector<int32_t> model_outputs;
  std::vector<int32_t> op_inputs;
  std::vector<int32_t> op_outputs;
};

struct OneOpRunSpec {
  TfLiteRegistration* registration = nullptr;
  int min_version = 1;
  int max_version = 1;
  size_t max_live_allocation_bytes = 0;
  std::vector<RuntimeTensor> runtime_tensors;
  std::vector<int> persistent_ro_tensors;
  bool invoke = false;
  std::function<RunResult(Interpreter*)> post_allocate;
};

RunResult BuildAndRunOneOpModel(flatbuffers::FlatBufferBuilder* builder,
                                const OneOpModelSpec& model_spec,
                                const OneOpRunSpec& run_spec);

}  // namespace fuzzing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_FUZZING_ONE_OP_FUZZ_MODEL_H_
