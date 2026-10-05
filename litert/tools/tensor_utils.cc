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

#include <algorithm>
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
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
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
namespace {

struct InputListEntry {
  absl::string_view name;
  absl::string_view path;
};

Expected<InputListEntry> ParseInputListEntry(absl::string_view entry) {
  const size_t separator = entry.find(":=");
  if (separator == absl::string_view::npos) {
    return InputListEntry{"", entry};
  }
  const absl::string_view name = entry.substr(0, separator);
  if (name.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Input list tensor name is empty.");
  }
  return InputListEntry{name, entry.substr(separator + 2)};
}

template <typename InputNames>
size_t ResolveInputIndex(const InputNames& input_names,
                         const InputListEntry& entry, size_t positional_index) {
  if (entry.name.empty()) {
    return positional_index;
  }
  auto input_it = std::find(input_names.begin(), input_names.end(), entry.name);
  return input_it == input_names.end()
             ? positional_index
             : static_cast<size_t>(input_it - input_names.begin());
}

}  // namespace

Expected<void> FillSingleInputBufferFromFile(
    const CompiledModel& compiled_model, size_t signature_index,
    absl::string_view input_name, TensorBuffer& input_buffer,
    absl::string_view file_path, bool quantize_inputs) {
  LITERT_ASSIGN_OR_RETURN(auto data,
                          tensor_utils::ReadTensorDataFromRawFile(file_path));
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
            return input_buffer.Write<int8_t>(absl::MakeConstSpan(q_vec));
          }
          case ElementType::UInt8: {
            auto q_vec = QuantizeData<uint8_t>(float_data, q_params.scale,
                                               q_params.zero_point);
            return input_buffer.Write<uint8_t>(absl::MakeConstSpan(q_vec));
          }
          case ElementType::Int16: {
            auto q_vec = QuantizeData<int16_t>(float_data, q_params.scale,
                                               q_params.zero_point);
            return input_buffer.Write<int16_t>(absl::MakeConstSpan(q_vec));
          }
          case ElementType::UInt16: {
            auto q_vec = QuantizeData<uint16_t>(float_data, q_params.scale,
                                                q_params.zero_point);
            return input_buffer.Write<uint16_t>(absl::MakeConstSpan(q_vec));
          }
          case ElementType::Int32: {
            auto q_vec = QuantizeData<int32_t>(float_data, q_params.scale,
                                               q_params.zero_point);
            return input_buffer.Write<int32_t>(absl::MakeConstSpan(q_vec));
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
  return tensor_utils::FillBufferWithCustomData(input_buffer, data);
}

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
    LITERT_RETURN_IF_ERROR(FillSingleInputBufferFromFile(
        compiled_model, signature_index, input_name, input_buffer,
        input_file_path.string(), quantize_inputs));
  }
  return {};
}

Expected<std::vector<std::vector<std::string>>> ParseInputListFile(
    absl::string_view input_list_path) {
  const std::string path_str(input_list_path);
  std::ifstream file(path_str);
  if (!file.is_open()) {
    return Unexpected(
        kLiteRtStatusErrorNotFound,
        absl::StrFormat("Failed to open input list file %s.",
                        input_list_path));
  }
  std::vector<std::vector<std::string>> lines;
  std::string line;
  bool is_leading_metadata = true;
  while (std::getline(file, line)) {
    const absl::string_view trimmed_line = absl::StripAsciiWhitespace(line);
    if (trimmed_line.empty()) {
      continue;
    }
    if (is_leading_metadata &&
        (absl::StartsWith(trimmed_line, "#") ||
         absl::StartsWith(trimmed_line, "%"))) {
      continue;
    }
    is_leading_metadata = false;
    std::vector<std::string> file_paths =
        absl::StrSplit(trimmed_line, absl::ByAnyChar(" \t"),
                       absl::SkipEmpty());
    if (file_paths.empty()) {
      continue;
    }
    lines.push_back(std::move(file_paths));
  }
  if (lines.empty()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("Input list file %s contains no entries.",
                        input_list_path));
  }
  return lines;
}

Expected<void> FillInputBuffersFromFileList(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& input_buffers,
    absl::Span<const std::string> file_paths, bool quantize_inputs) {
  if (file_paths.size() != input_buffers.size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("Number of files on input list line (%d) does not "
                        "match number of model inputs (%d) for signature "
                        "%d.",
                        file_paths.size(), input_buffers.size(),
                        signature_index));
  }
  LITERT_ASSIGN_OR_RETURN(
      const auto input_names,
      compiled_model.GetSignatureInputNames(signature_index));
  if (input_names.size() != input_buffers.size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("Number of input buffers (%d) does not match number "
                        "of model inputs (%d) for signature %d.",
                        input_buffers.size(), input_names.size(),
                        signature_index));
  }

  for (size_t i = 0; i < input_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(const InputListEntry entry,
                            ParseInputListEntry(file_paths[i]));
    const size_t input_index = ResolveInputIndex(input_names, entry, i);
    LITERT_RETURN_IF_ERROR(FillSingleInputBufferFromFile(
        compiled_model, signature_index, input_names[input_index],
        input_buffers[input_index], entry.path, quantize_inputs));
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
  const std::filesystem::path output_dir_path = std::string(output_dir);
  std::error_code ec;
  if (!std::filesystem::is_directory(output_dir_path, ec)) {
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
        output_dir_path / absl::StrCat(output_name, ".raw");
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
