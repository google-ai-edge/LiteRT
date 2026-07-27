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

#include "litert/tools/build_custom_npu_model/build_custom_npu_model.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"

namespace litert::tools {

namespace {

Expected<OwningBufferRef<uint8_t>> ReadFile(absl::string_view path) {
  std::ifstream infile(std::string(path), std::ios::binary | std::ios::ate);
  if (!infile) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      absl::StrCat("Cannot open binary file: ", path));
  }
  std::streamsize file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  OwningBufferRef<uint8_t> buffer(file_size);
  if (!infile.read(reinterpret_cast<char*>(buffer.Data()), file_size)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      absl::StrCat("Failed to read binary file: ", path));
  }
  return buffer;
}

Expected<void> WriteFile(absl::string_view path, BufferRef<uint8_t> data) {
  std::ofstream outfile(std::string(path), std::ios::binary);
  if (!outfile) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      absl::StrCat("Cannot open output file: ", path));
  }
  outfile.write(reinterpret_cast<const char*>(data.Data()), data.Size());
  if (!outfile.good()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      absl::StrCat("Failed to write to output file: ", path));
  }
  return {};
}

}  // namespace

Expected<LiteRtElementType> ParseElementType(absl::string_view dtype_str) {
  std::string s = absl::AsciiStrToLower(absl::StripAsciiWhitespace(dtype_str));
  if (s == "f32") {
    return kLiteRtElementTypeFloat32;
  } else if (s == "i32") {
    return kLiteRtElementTypeInt32;
  } else if (s == "u8") {
    return kLiteRtElementTypeUInt8;
  } else if (s == "i8") {
    return kLiteRtElementTypeInt8;
  } else if (s == "i16") {
    return kLiteRtElementTypeInt16;
  } else if (s == "f16") {
    return kLiteRtElementTypeFloat16;
  } else if (s == "bool") {
    return kLiteRtElementTypeBool;
  }
  return Unexpected(kLiteRtStatusErrorInvalidArgument,
                    absl::StrCat("Unsupported data type string: ", dtype_str));
}

Expected<std::vector<int32_t>> ParseDimensions(absl::string_view shape_str) {
  std::vector<int32_t> dims;
  std::string clean_str =
      absl::AsciiStrToLower(absl::StripAsciiWhitespace(shape_str));
  if (clean_str.empty()) {
    return dims;
  }

  // Enforce AxBxCxD format using 'x' delimiter
  std::vector<absl::string_view> tokens =
      absl::StrSplit(clean_str, 'x', absl::SkipEmpty());

  for (auto tok : tokens) {
    int32_t dim_val = 0;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(tok), &dim_val) ||
        dim_val <= 0) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrCat("Invalid dimension value in AxBxCxD format: ", tok));
    }
    dims.push_back(dim_val);
  }
  return dims;
}

Expected<std::vector<TensorInfo>> ParseTensorInfoList(
    absl::string_view shapes_flag, absl::string_view dtypes_flag,
    absl::string_view names_flag, absl::string_view default_prefix) {
  std::vector<TensorInfo> tensors;
  if (shapes_flag.empty()) {
    return tensors;
  }

  // Split multiple tensor shapes by ';' or ',' (e.g. "1x224x224x3,1x10" or
  // "1x224x224x3;1x10")
  std::vector<absl::string_view> shape_tokens =
      absl::StrSplit(shapes_flag, absl::ByAnyChar(";,"), absl::SkipEmpty());

  std::vector<absl::string_view> dtype_tokens;
  if (!dtypes_flag.empty()) {
    dtype_tokens = absl::StrSplit(dtypes_flag, absl::ByAnyChar(";,"));
  }

  std::vector<absl::string_view> name_tokens;
  if (!names_flag.empty()) {
    name_tokens = absl::StrSplit(names_flag, absl::ByAnyChar(";,"));
  }

  for (size_t i = 0; i < shape_tokens.size(); ++i) {
    TensorInfo info;
    if (i < name_tokens.size() && !name_tokens[i].empty()) {
      info.name = std::string(absl::StripAsciiWhitespace(name_tokens[i]));
    } else {
      info.name = absl::StrCat(default_prefix, "_", i);
    }

    LITERT_ASSIGN_OR_RETURN(info.dimensions, ParseDimensions(shape_tokens[i]));

    if (i < dtype_tokens.size() && !dtype_tokens[i].empty()) {
      LITERT_ASSIGN_OR_RETURN(info.element_type,
                              ParseElementType(dtype_tokens[i]));
    } else {
      info.element_type = kLiteRtElementTypeFloat32;
    }
    tensors.push_back(std::move(info));
  }

  return tensors;
}

Expected<OwningBufferRef<uint8_t>> BuildCustomNpuModelMemory(
    const BuildCustomNpuModelOptions& options,
    BufferRef<uint8_t> bytecode_data) {
  if (bytecode_data.Size() == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "NPU bytecode buffer is empty.");
  }
  if (options.input_tensors.empty() || options.output_tensors.empty()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Input and output tensor specifications must be provided.");
  }

  ::LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // 1. Emplace Input Tensors
  for (size_t i = 0; i < options.input_tensors.size(); ++i) {
    const auto& in_info = options.input_tensors[i];
    auto& tensor = subgraph.EmplaceTensor();
    tensor.SetName(in_info.name);
    tensor.SetType(MakeRankedTensorType(in_info.element_type,
                                        absl::MakeSpan(in_info.dimensions)));
    subgraph.Inputs().push_back(&tensor);
    input_names.push_back(in_info.name);
  }

  // 2. Emplace Output Tensors
  for (size_t i = 0; i < options.output_tensors.size(); ++i) {
    const auto& out_info = options.output_tensors[i];
    auto& tensor = subgraph.EmplaceTensor();
    tensor.SetName(out_info.name);
    tensor.SetType(MakeRankedTensorType(out_info.element_type,
                                        absl::MakeSpan(out_info.dimensions)));
    subgraph.Outputs().push_back(&tensor);
    output_names.push_back(out_info.name);
  }

  // 3. Emplace & Configure Custom Dispatch Op
  auto& dispatch_op = subgraph.EmplaceOp();
  ::MakeDispatchOp(dispatch_op);
  dispatch_op.Inputs() = subgraph.Inputs();
  dispatch_op.Outputs() = subgraph.Outputs();

  // 4. Register NPU Bytecode Asset in Model Buffers
  OwningBufferRef<uint8_t> bytecode_copy(bytecode_data.Data(),
                                         bytecode_data.Size());
  auto buf_id = model.Buffers()->RegisterOwnedBuffer(std::move(bytecode_copy));
  std::string entry_point =
      options.entry_point_name.empty() ? "main" : options.entry_point_name;
  model.AttachAssetToOp(&dispatch_op, buf_id, entry_point);

  // 5. Emplace Signature Definition with custom/default signature key name
  std::string sig_key =
      options.signature_key.empty() ? "serving_default" : options.signature_key;
  model.EmplaceSignature(&subgraph, std::move(input_names), subgraph.Inputs(),
                         std::move(output_names), subgraph.Outputs(), sig_key);

  // 6. Serialize to TFLite FlatBuffer format
  return litert::internal::SerializeModel(std::move(model));
}

Expected<void> BuildCustomNpuModel(const BuildCustomNpuModelOptions& options) {
  if (options.npu_bytecode_path.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "NPU bytecode path must be provided.");
  }
  if (options.output_model_path.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Output model path must be provided.");
  }

  LITERT_ASSIGN_OR_RETURN(auto bytecode_buf,
                          ReadFile(options.npu_bytecode_path));

  if (options.input_tensors.empty() || options.output_tensors.empty()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Input and output tensor specifications must be provided via "
        "--input_shapes and --output_shapes.");
  }

  LITERT_ASSIGN_OR_RETURN(auto serialized_model,
                          BuildCustomNpuModelMemory(options, bytecode_buf));

  LITERT_RETURN_IF_ERROR(
      WriteFile(options.output_model_path, serialized_model));

  ABSL_LOG(INFO) << "Successfully built custom NPU model at: "
                 << options.output_model_path << " (" << serialized_model.Size()
                 << " bytes)";
  return {};
}

}  // namespace litert::tools
