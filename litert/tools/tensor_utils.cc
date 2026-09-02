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

#include "litert/tools/tensor_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <numeric>
#include <string>
#include <system_error>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace tensor_utils {

Expected<void> FillInputBuffersWithCustomData(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& input_buffers, absl::string_view input_dir,
    bool quantize_inputs) {
  ABSL_LOG(INFO) << "Using inputs from: " << input_dir;
  LITERT_ASSIGN_OR_RETURN(
      const auto input_names,
      compiled_model.GetSignatureInputNames(signature_index));
  if (input_buffers.size() != input_names.size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("Number of input buffers (%d) does not match number "
                        "of model inputs (%d) for signature %d.",
                        input_buffers.size(), input_names.size(),
                        signature_index));
  }
  for (size_t i = 0; i < input_names.size(); ++i) {
    const auto& input_name = input_names[i];
    auto& input_buffer = input_buffers[i];
    const auto input_file_path =
        std::filesystem::path(std::string(input_dir)) /
        (std::string(input_name.data(), input_name.size()) + ".raw");
    LITERT_ASSIGN_OR_RETURN(auto data, tensor_utils::ReadTensorDataFromRawFile(
                                           input_file_path.string()));
    if (quantize_inputs) {
      LITERT_ASSIGN_OR_RETURN(auto q_type, compiled_model.GetInputTensorQTypeId(
                                               signature_index, input_name));
      LITERT_ASSIGN_OR_RETURN(auto type, input_buffer.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto buffer_size, input_buffer.Size());
      const auto& layout = type.Layout();
      size_t total_elements = std::accumulate(layout.Dimensions().begin(),
                                              layout.Dimensions().end(), 1,
                                              std::multiplies<size_t>());
      const size_t expected_fp32_size = total_elements * sizeof(float);

      if (q_type == QuantizationTypeId::PerTensor) {
        if (data.size() == expected_fp32_size) {
          LITERT_ASSIGN_OR_RETURN(
              auto q_params, compiled_model.GetInputTensorPerTensorQuantization(
                                 signature_index, input_name));
          if (q_params.scale <= 0.0f || !std::isfinite(q_params.scale)) {
            return Unexpected(
                kLiteRtStatusErrorRuntimeFailure,
                absl::StrFormat(
                    "Invalid quantization scale %f for input tensor '%s'.",
                    q_params.scale, input_name));
          }
          absl::Span<const float> float_data(
              reinterpret_cast<const float*>(data.data()), total_elements);

          ABSL_LOG(INFO) << "Quantizing input tensor '" << input_name
                         << "' from FP32 to type "
                         << static_cast<int>(type.ElementType())
                         << " (scale=" << q_params.scale
                         << ", zero_point=" << q_params.zero_point << ")";

          switch (type.ElementType()) {
            case ElementType::Int8: {
              auto q_vec = QuantizeData<int8_t>(float_data, q_params.scale,
                                                q_params.zero_point);
              LITERT_RETURN_IF_ERROR(
                  input_buffer.Write<int8_t>(absl::MakeConstSpan(q_vec)));
              continue;
            }
            case ElementType::UInt8: {
              auto q_vec = QuantizeData<uint8_t>(float_data, q_params.scale,
                                                 q_params.zero_point);
              LITERT_RETURN_IF_ERROR(
                  input_buffer.Write<uint8_t>(absl::MakeConstSpan(q_vec)));
              continue;
            }
            case ElementType::Int16: {
              auto q_vec = QuantizeData<int16_t>(float_data, q_params.scale,
                                                 q_params.zero_point);
              LITERT_RETURN_IF_ERROR(
                  input_buffer.Write<int16_t>(absl::MakeConstSpan(q_vec)));
              continue;
            }
            case ElementType::UInt16: {
              auto q_vec = QuantizeData<uint16_t>(float_data, q_params.scale,
                                                  q_params.zero_point);
              LITERT_RETURN_IF_ERROR(
                  input_buffer.Write<uint16_t>(absl::MakeConstSpan(q_vec)));
              continue;
            }
            case ElementType::Int32: {
              auto q_vec = QuantizeData<int32_t>(float_data, q_params.scale,
                                                 q_params.zero_point);
              LITERT_RETURN_IF_ERROR(
                  input_buffer.Write<int32_t>(absl::MakeConstSpan(q_vec)));
              continue;
            }
            default:
              return Unexpected(
                  kLiteRtStatusErrorRuntimeFailure,
                  absl::StrFormat("Auto-quantization is not supported for "
                                  "element type %d on tensor '%s'.",
                                  static_cast<int>(type.ElementType()),
                                  input_name));
          }
        } else if (data.size() != buffer_size) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              absl::StrFormat(
                  "Mismatched input size for '%s'. Expected %d bytes "
                  "(for FP32 auto-quantization) or %d bytes (raw "
                  "quantized buffer), but got %d bytes.",
                  input_name, expected_fp32_size, buffer_size, data.size()));
        }
      } else if (q_type != QuantizationTypeId::None) {
        ABSL_LOG(WARNING) << "Auto-quantization requested, but tensor '"
                          << input_name
                          << "' has unsupported quantization type "
                          << static_cast<int>(q_type)
                          << "; attempting raw fill.";
      }
    }
    LITERT_RETURN_IF_ERROR(
        tensor_utils::FillBufferWithCustomData(input_buffer, data));
  }
  return {};
}

Expected<void> WriteOutputBuffersToFiles(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& output_buffers, absl::string_view output_dir) {
  ABSL_LOG(INFO) << "Writing outputs to: " << output_dir;
  LITERT_ASSIGN_OR_RETURN(
      const auto output_names,
      compiled_model.GetSignatureOutputNames(signature_index));
  if (output_names.size() != output_buffers.size()) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Mismatched output count: signature has %d outputs "
                        "but got %d output buffers.",
                        output_names.size(), output_buffers.size()));
  }
  std::error_code ec;
  if (!std::filesystem::is_directory(output_dir, ec)) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Output directory %s does not exist or is not a "
                        "directory.",
                        output_dir));
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    const auto output_name = output_names[i];
    auto& output_buffer = output_buffers[i];
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, output_buffer.Size());
    LITERT_ASSIGN_OR_RETURN(void* host_mem_addr,
                            output_buffer.Lock(TensorBuffer::LockMode::kRead));
    absl::Cleanup unlock = [&output_buffer] { output_buffer.Unlock(); };
    const auto output_file_path =
        std::filesystem::path(output_dir) / absl::StrCat(output_name, ".raw");
    std::ofstream file(output_file_path, std::ios::binary);
    if (!file.is_open()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        absl::StrFormat("Failed to open output file %s.",
                                        output_file_path.string()));
    }
    file.write(static_cast<const char*>(host_mem_addr), buffer_size);
    file.close();
    if (!file) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        absl::StrFormat("Failed to write output file %s.",
                                        output_file_path.string()));
    }
    ABSL_LOG(INFO) << "Wrote output " << output_name << " (" << buffer_size
                   << " bytes) to " << output_file_path;
  }
  return {};
}

}  // namespace tensor_utils
}  // namespace litert
