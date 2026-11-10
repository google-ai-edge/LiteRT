// Copyright 2024 Google LLC.
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

#include "litert/runtime/tfl_utils.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/internal/litert_dispatch_delegate_vendor_options.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/runtime/tensor_identifier.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

#if defined(LITERT_NO_BUILTIN_OPS)
#include <memory>

#include "tflite/mutable_op_resolver.h"

namespace tflite {
__attribute__((weak)) std::unique_ptr<MutableOpResolver> CreateOpResolver() {
  return std::make_unique<MutableOpResolver>();
}
}  // namespace tflite
#endif  // defined(LITERT_NO_BUILTIN_OPS)

namespace litert::internal {

namespace {

std::optional<absl::string_view> FindCanonicalSignatureInputName(
    const LiteRtSignatureT& signature, LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return std::nullopt;
  }
  for (const std::string& name : signature.InputNames()) {
    Expected<LiteRtTensor> sig_tensor = signature.FindInputTensor(name);
    if (sig_tensor.HasValue() && *sig_tensor == tensor) {
      return name;
    }
  }
  return std::nullopt;
}

std::optional<absl::string_view> FindCanonicalSignatureOutputName(
    const LiteRtSignatureT& signature, LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return std::nullopt;
  }
  for (const std::string& name : signature.OutputNames()) {
    Expected<LiteRtTensor> sig_tensor = signature.FindOutputTensor(name);
    if (sig_tensor.HasValue() && *sig_tensor == tensor) {
      return name;
    }
  }
  return std::nullopt;
}

}  // namespace

TfLiteStatus SetCustomAllocationForInputTensor(
    tflite::Interpreter* interpreter,
    const LiteRtExternalTensorBinding& binding) {
  if (!interpreter) {
    return kTfLiteError;
  }
  if (!binding.data) {
    return kTfLiteError;
  }

  const char* signature_name =
      binding.signature_name.empty() ? nullptr : binding.signature_name.c_str();
  auto* signature_runner = interpreter->GetSignatureRunner(signature_name);
  if (!signature_runner) {
    LITERT_LOG(LITERT_INFO, "Signature %s not found in interpreter",
               binding.signature_name.c_str());
    return kTfLiteError;
  }

  const TfLiteCustomAllocation custom_allocation = {binding.data,
                                                    binding.size_bytes};
  signature_runner->SetCustomAllocationForInputTensor(
      binding.tensor_name.c_str(), custom_allocation);

  return kTfLiteOk;
}

Expected<LiteRtLayout> ConvertTensorLayout(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  size_t rank = TfLiteOpaqueTensorNumDims(tfl_opaque_tensor);
  auto layout = LiteRtLayout{};
  layout.rank = rank;
  layout.has_strides = false;
  for (size_t i = 0; i < rank; ++i) {
    layout.dimensions[i] = TfLiteOpaqueTensorDim(tfl_opaque_tensor, i);
  }
  return layout;
}

Expected<LiteRtRankedTensorType> ConvertTensorType(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  auto tfl_type = TfLiteOpaqueTensorType(tfl_opaque_tensor);
  auto element_type = static_cast<LiteRtElementType>(tfl_type);
  auto layout = ConvertTensorLayout(tfl_opaque_tensor);
  return LiteRtRankedTensorType{/*element_type = */ element_type,
                                /* layout = */ layout.Value()};
}

Expected<void> ResizeTensor(const LiteRtLayout& layout,
                            TfLiteOpaqueContext* tfl_context,
                            TfLiteOpaqueTensor* tfl_opaque_tensor) {
  // TFL tensors don't support strides.
  if (layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unexpected layout with strides");
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(layout.rank);
  for (auto i = 0; i < layout.rank; ++i) {
    output_size->data[i] = layout.dimensions[i];
  }
  if (auto status = TfLiteOpaqueContextResizeTensor(
          tfl_context, tfl_opaque_tensor, output_size);
      status != kTfLiteOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("Failed to resize TFL tensor ",
                     TfLiteOpaqueTensorName(tfl_opaque_tensor), ": ", status));
  }

  return {};
}

Expected<TfLiteTensorIdentifier> GetTensorIdentifier(
    const tflite::Interpreter& interpreter, const TfLiteTensor* target_tensor) {
  for (int i = 0; i < interpreter.subgraphs_size(); ++i) {
    const auto* subgraph = interpreter.subgraph(i);
    // Instead of iterating over all tensors, we can start from the first one
    // and look for the target tensor in the array of tensors. This assumes
    // that the target tensor is linear located after the base tensor in memory.
    const auto* base_tensor = subgraph->tensor(0);
    const std::ptrdiff_t diff = target_tensor - base_tensor;
    if (diff < 0 || diff >= subgraph->tensors_size()) {
      continue;
    }
    if (subgraph->tensor(diff) == target_tensor) {
      return TfLiteTensorIdentifier{i, static_cast<int>(diff)};
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound,
                    "Tensor not found in interpreter");
}

Expected<DispatchDelegateVendorOptions> BuildDispatchDelegateVendorOptions(
    const LiteRtModelT& model) {
  DispatchDelegateVendorOptions vendor_options;

  // It is OK not to have any signatures. We are building the map only for
  // signature-aware models.
  if (model.Signatures().empty()) {
    return vendor_options;
  }

  for (LiteRtSignature signature : model.Signatures()) {
    if (!signature) {
      continue;
    }
    absl::string_view sig_key = signature->Key();
    const LiteRtSubgraphT& subgraph = signature->GetSubgraph();
    // Go over all ops in the subgraph and pull out the function names.
    for (LiteRtOp op : subgraph.Ops()) {
      if (op->OpCode() != kLiteRtOpCodeTflCustom) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(absl::string_view custom_code, op->CustomCode());
      if (custom_code == litert::internal::kLiteRtDispatchOpCustomName &&
          op->CustomOptions().Size() > 0) {
        litert::internal::DispatchOpOptions op_opts =
            litert::internal::GetDispatchOpOptions(op->CustomOptions());
        if (!op_opts.name.empty()) {
          FunctionMapping mapping;
          mapping.function_name = op_opts.name;
          mapping.signature_name = std::string(sig_key);

          mapping.input_tensor_ports.reserve(op->Inputs().size());
          for (int port_index = 0; port_index < op->Inputs().size();
               ++port_index) {
            LiteRtTensor input_tensor = op->Inputs()[port_index];
            std::optional<absl::string_view> canonical_name =
                FindCanonicalSignatureInputName(*signature, input_tensor);
            if (canonical_name.has_value() && !canonical_name->empty()) {
              mapping.input_tensor_ports.push_back(
                  {std::string(*canonical_name), port_index});
            }
          }

          mapping.output_tensor_ports.reserve(op->Outputs().size());
          for (int port_index = 0; port_index < op->Outputs().size();
               ++port_index) {
            LiteRtTensor output_tensor = op->Outputs()[port_index];
            std::optional<absl::string_view> canonical_name =
                FindCanonicalSignatureOutputName(*signature, output_tensor);
            if (canonical_name.has_value() && !canonical_name->empty()) {
              mapping.output_tensor_ports.push_back(
                  {std::string(*canonical_name), port_index});
            }
          }

          vendor_options.AddFunctionMapping(std::move(mapping));
        }
      }
    }
  }
  return vendor_options;
}

}  // namespace litert::internal
