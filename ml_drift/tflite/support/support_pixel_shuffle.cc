// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/odml/litert/ml_drift/tflite/support/support_pixel_shuffle.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

namespace litert::ml_drift::ir {

bool IsPixelShuffleSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int output_id = outputs->data[0];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      // go/keep-sorted end
      // clang-format on});
  };
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "input", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "output", *error)) {
    return false;
  }

  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }

  // Check shapes
  if (input.dims->size < 3 || input.dims->size > 5) {
    *error = absl::StrCat("Invalid number of input dims: ", input.dims->size,
                          ", should be between 3 and 5");
    return false;
  }
  if (output.dims->size != input.dims->size) {
    *error = absl::StrCat("Output and input dimensions must match: ",
                          output.dims->size, " != ", input.dims->size);
    return false;
  }

  // Parse params
  const uint8_t* buffer_t =
      reinterpret_cast<const uint8_t*>(node->custom_initial_data);
  if (buffer_t == nullptr) {
    *error = "PixelShuffle is missing params.";
    return false;
  }
  size_t length = node->custom_initial_data_size;
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  if (m["num_groups"].IsNull()) {
    *error = "PixelShuffle is missing num_groups.";
    return false;
  }
  const int num_groups = m["num_groups"].AsInt32();
  if (num_groups <= 0) {
    *error = absl::StrCat("Invalid number of groups: ", num_groups,
                          ", should be greater than 0");
    return false;
  }

  // Given r = num_groups, the input and output dimensions should be:
  //   input_dims = (*, C * r^2, H, W)
  //   output_dims = (*, C, H * r, W * r)
  const int n_dims = input.dims->size;
  if (input.dims->data[n_dims - 3] !=
      output.dims->data[n_dims - 3] * num_groups * num_groups) {
    *error = absl::StrCat(
        "Invalid shape of 3rd to last dim. Should be in[-3] = "
        "out[-3] * num_groups^2. Got in[-3]: ",
        input.dims->data[n_dims - 3], " num_groups: ", num_groups,
        " and out[-3]: ", output.dims->data[n_dims - 3]);
    return false;
  }

  if (input.dims->data[n_dims - 2] * num_groups !=
      output.dims->data[n_dims - 2]) {
    *error = absl::StrCat(
        "Invalid shape of 2nd to last dim. Should be in[-2] * num_groups = "
        "out[-2]. Got in[-2]: ",
        input.dims->data[n_dims - 2], " num_groups: ", num_groups,
        " and out[-2]: ", output.dims->data[n_dims - 2]);
    return false;
  }
  if (input.dims->data[n_dims - 1] * num_groups !=
      output.dims->data[n_dims - 1]) {
    *error = absl::StrCat(
        "Invalid shape of last dim. Should be in[-1] * num_groups = "
        "out[-1]. Got in[-1]: ",
        input.dims->data[n_dims - 1], " num_groups: ", num_groups,
        " and out[-1]: ", output.dims->data[n_dims - 1]);
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
