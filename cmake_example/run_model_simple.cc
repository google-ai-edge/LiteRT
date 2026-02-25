// Copyright 2025 Google LLC.
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

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

ABSL_FLAG(std::string, graph, "", "Model filename to use for testing.");
ABSL_FLAG(std::string, accelerator, "cpu",
          "Which backend to use. Comma delimited string of accelerators (e.g. "
          "cpu,gpu). Will delegate to GPU, then CPU if they are "
          "specified in this flag.");
ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(bool, compare_numerical, false,
          "Decide if random value should be filled into tensor buffer to "
          "perform numerical check.");
ABSL_FLAG(size_t, sample_size, 5,
          "Number of sample elements to print from beginning, middle, and end "
          "of tensor.");

namespace litert {
namespace {

litert::HwAcceleratorSet GetAccelerator() {
  const std::string accelerator_str = absl::GetFlag(FLAGS_accelerator);
  litert::HwAcceleratorSet accelerators(
      static_cast<int>(litert::HwAccelerators::kNone));
  for (absl::string_view accelerator : absl::StrSplit(accelerator_str, ',')) {
    if (accelerator == "gpu") {
      accelerators |= litert::HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= litert::HwAccelerators::kNpu;
    } else if (accelerator == "cpu") {
      accelerators |= litert::HwAccelerators::kCpu;
    }
  }
  return accelerators;
}

Expected<Environment> GetEnvironment() { return Environment::Create({}); }

Expected<Options> GetOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(GetAccelerator());
  return options;
}

// Helper function to get the total number of elements in a tensor.
Expected<size_t> GetTotalElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto& layout = type.Layout();
  size_t total_elements = 1;
  for (size_t d = 0; d < layout.Rank(); ++d) {
    total_elements *= layout.Dimensions()[d];
  }
  return total_elements;
}

// Fills a tensor buffer with sample data based on element type
Expected<void> FillInputBuffer(TensorBuffer& buffer) {
  if (!absl::GetFlag(FLAGS_compare_numerical)) {
    return {};
  }

  LITERT_ASSIGN_OR_RETURN(const size_t total_elements,
                          GetTotalElements(buffer));

  // Always treat input as float and fill with rotating values from 0.0 to 0.9
  std::vector<float> data(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    // Rotate through 0.0, 0.1, 0.2, ..., 0.9, 0.0, 0.1, ...
    data[i] = static_cast<float>(i % 10) * 0.1f;
  }

  // Write the data to the tensor buffer
  return buffer.Write<float>(absl::MakeConstSpan(data));
}

Expected<void> RunModel() {
  if (absl::GetFlag(FLAGS_graph).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Model filename is empty. Use --graph to provide it.");
  }

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
  LITERT_ASSIGN_OR_RETURN(auto options, GetOptions());

  ABSL_LOG(INFO) << "Model: " << absl::GetFlag(FLAGS_graph);
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, absl::GetFlag(FLAGS_graph), options));

  size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  ABSL_LOG(INFO) << "Signature index: " << signature_index;

  ABSL_LOG(INFO) << "Prepare input buffers";

  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model.CreateInputBuffers(signature_index));

    // Fill input buffers with sample data
    for (size_t i = 0; i < input_buffers.size(); ++i) {
      auto& buffer = input_buffers[i];
      LITERT_RETURN_IF_ERROR(FillInputBuffer(buffer));
    }

  ABSL_LOG(INFO) << "Prepare output buffers";

  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model.CreateOutputBuffers(signature_index));

  litert::Expected<void> status;
    status = compiled_model.Run(signature_index, input_buffers, output_buffers);

  ABSL_LOG(INFO) << "Model run completed";

  return status;
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto res = litert::RunModel();
  if (!res) {
    ABSL_LOG(ERROR) << res.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
