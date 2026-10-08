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

#include "litert/runtime/litert_interpreter_builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/allocation.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/core/api/flatbuffer_conversions.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/core/subgraph.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_options.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/stderr_reporter.h"
#include "tflite/util.h"

#if __cplusplus >= 201703L && __STDC_VERSION__ >= 201112L
#if !defined(__ANDROID__) || __ANDROID_API__ >= 28
#if !defined(__APPLE__) && !defined(_WIN32)
#define LITERT_TFLITE_USE_STD_ALIGNED_ALLOC
#endif
#endif
#endif

namespace tflite {
namespace interpreter_wrapper {
class InterpreterWrapper {
 public:
  template <typename Interp>
  static void AddSubgraphs(Interp* interpreter, int subgraphs_to_add) {
    interpreter->AddSubgraphs(subgraphs_to_add);
  }
  template <typename Interp>
  static void EmplaceSignatureDef(Interp* interpreter,
                                  std::map<std::string, uint32_t> inputs,
                                  std::map<std::string, uint32_t> outputs,
                                  std::string signature_key,
                                  uint32_t subgraph_index) {
    auto& sig_def = interpreter->signature_defs_.emplace_back();
    sig_def.inputs = std::move(inputs);
    sig_def.outputs = std::move(outputs);
    sig_def.signature_key = std::move(signature_key);
    sig_def.subgraph_index = subgraph_index;
  }
  template <typename Interp, typename MetadataMap>
  static TfLiteStatus SetMetadata(Interp* interpreter,
                                  const MetadataMap& metadata) {
    return interpreter->SetMetadata(metadata);
  }
  template <typename Interp>
  static void ApplySignatureTensorNames(Interp* interpreter) {
    for (auto& signature_def : interpreter->signature_defs_) {
      auto* subgraph = interpreter->subgraph(signature_def.subgraph_index);
      for (auto& [name, tensor_index] : signature_def.inputs) {
        auto* tensor = subgraph->tensor(tensor_index);
        tensor->name = name.c_str();
      }
      for (auto& [name, tensor_index] : signature_def.outputs) {
        auto* tensor = subgraph->tensor(tensor_index);
        tensor->name = name.c_str();
      }
    }
  }
  template <typename Interp, typename Opts>
  static TfLiteStatus ApplyOptionsImpl(Interp* interpreter, Opts* options) {
    return interpreter->ApplyOptionsImpl(options);
  }
};
}  // namespace interpreter_wrapper

class SingleOpModel {
 public:
  template <typename SubgraphT>
  static void SetAllocation(SubgraphT* subgraph, const Allocation* allocation) {
    subgraph->allocation_ = allocation;
  }
  template <typename SubgraphT>
  static void ReserveNodes(SubgraphT* subgraph, int count) {
    subgraph->ReserveNodes(count);
  }
};
}  // namespace tflite

namespace litert::internal {
namespace {

class LiteRtMallocDataAllocator : public tflite::BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
#ifdef LITERT_TFLITE_USE_STD_ALIGNED_ALLOC
    size_t used_alignment = std::max(alignment_hint, sizeof(void*));
    size_t used_size =
        ((size + used_alignment - 1) / used_alignment) * used_alignment;
    return aligned_alloc(used_alignment, used_size);
#else
    return malloc(size);
#endif
  }
  void Deallocate(void* data) override { free(data); }
};

template <typename GetTensorIndexFn>
TfLiteStatus ParseQuantizationFromLiteRtTensor(
    const LiteRtTensorT& litert_tensor, const std::vector<int>& dims,
    GetTensorIndexFn get_tensor_index,
    const tflite::InterpreterOptions& options,
    tflite::ErrorReporter* error_reporter, TfLiteQuantization* quantization) {
  *quantization = {kTfLiteNoQuantization, nullptr};
  const auto& qparams = litert_tensor.Qparams();
  if (qparams.first == kLiteRtQuantizationNone) {
    return kTfLiteOk;
  }

  if (qparams.first == kLiteRtQuantizationBlockWise) {
    const auto& bw = qparams.second.block_wise;
    int32_t scales_index = get_tensor_index(bw.scales);
    if (scales_index < 0) {
      return kTfLiteError;
    }
    int32_t zp_index = -1;
    if (bw.zero_points != nullptr) {
      zp_index = get_tensor_index(bw.zero_points);
      if (zp_index < 0) {
        return kTfLiteError;
      }
    }
    auto* params = static_cast<TfLiteBlockwiseQuantization*>(
        malloc(sizeof(TfLiteBlockwiseQuantization)));
    *params = {
        .scale = scales_index,
        .zero_point = zp_index,
        .blocksize = bw.block_size,
        .quantized_dimension = 0,
    };
    *quantization = {kTfLiteBlockwiseQuantization, params};
    return kTfLiteOk;
  }

  if (qparams.first == kLiteRtQuantizationPerTensor) {
    const auto& pt = qparams.second.per_tensor;
    auto* affine = static_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    affine->scale = TfLiteFloatArrayCreate(1);
    affine->scale->data[0] = pt.scale;
    affine->zero_point = TfLiteIntArrayCreate(1);
    affine->zero_point->data[0] = static_cast<int32_t>(pt.zero_point);
    affine->quantized_dimension = 0;
    *quantization = {kTfLiteAffineQuantization, affine};
    return kTfLiteOk;
  }

  if (qparams.first == kLiteRtQuantizationPerChannel) {
    const auto& pc = qparams.second.per_channel;
    const size_t num_scales = pc.num_channels;
    if (pc.quantized_dimension < 0 ||
        (!dims.empty() &&
         static_cast<size_t>(pc.quantized_dimension) >= dims.size()) ||
        (num_scales != 1 && !dims.empty() &&
         num_scales != static_cast<size_t>(dims[pc.quantized_dimension]))) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Invalid per-channel quantization parameters.");
      return kTfLiteError;
    }

    bool compress_zp =
        options.GetCompressQuantizationZeroPoints() && num_scales > 0;
    for (size_t i = 1; compress_zp && i < num_scales; ++i) {
      compress_zp = (pc.zero_points[i] == pc.zero_points[0]);
    }

    auto* affine = static_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    affine->scale = TfLiteFloatArrayCreate(num_scales);
    for (size_t i = 0; i < num_scales; ++i) {
      affine->scale->data[i] = pc.scales[i];
    }
    const size_t num_zp = compress_zp ? 1 : num_scales;
    affine->zero_point = TfLiteIntArrayCreate(num_zp);
    for (size_t i = 0; i < num_zp; ++i) {
      affine->zero_point->data[i] = static_cast<int32_t>(pc.zero_points[i]);
    }
    affine->quantized_dimension = pc.quantized_dimension;
    *quantization = {kTfLiteAffineQuantization, affine};
    return kTfLiteOk;
  }

  return kTfLiteError;
}

}  // namespace

TfLiteStatus BuildInterpreterFromLiteRtModel(
    LiteRtModelT& model, const tflite::OpResolver& op_resolver,
    tflite::ErrorReporter* error_reporter,
    const tflite::InterpreterOptions& options,
    const tflite::Allocation* allocation, int num_threads,
    std::unique_ptr<tflite::Interpreter>* interpreter_out) {
  interpreter_out->reset();
  tflite::ErrorReporter* reporter =
      error_reporter ? error_reporter : tflite::DefaultErrorReporter();

  const auto& subgraphs = model.Subgraphs();
  if (subgraphs.empty()) {
    return kTfLiteError;
  }

  const auto& op_codes = GetTflOpCodes(model);
  std::vector<const TfLiteRegistration*> op_index_to_registration(
      op_codes.size(), nullptr);
  std::vector<TfLiteRegistration> unresolved_custom_ops;
  size_t max_custom_ops = op_codes.size();
  for (const auto* sg : subgraphs) {
    max_custom_ops += sg->Ops().size();
  }
  unresolved_custom_ops.reserve(max_custom_ops);

  auto resolve_op =
      [&](tflite::BuiltinOperator builtin_code, const char* custom_code,
          int32_t version,
          const TfLiteRegistration** registration) -> TfLiteStatus {
    if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
      *registration = op_resolver.FindOp(custom_code, version);
      if (*registration == nullptr) {
        unresolved_custom_ops.push_back(
            tflite::CreateUnresolvedCustomOp(custom_code));
        *registration = &unresolved_custom_ops.back();
      }
      return kTfLiteOk;
    }
    *registration = op_resolver.FindOp(builtin_code, version);
    if (*registration == nullptr) {
      TF_LITE_REPORT_ERROR(
          reporter, "Didn't find op for builtin opcode '%s' version '%d'.\n",
          tflite::EnumNameBuiltinOperator(builtin_code), version);
      return kTfLiteError;
    }
    return kTfLiteOk;
  };

  for (size_t i = 0; i < op_codes.size(); ++i) {
    const auto& opcode = *op_codes[i];
    tflite::BuiltinOperator builtin_code =
        opcode.builtin_code ==
                tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
            ? static_cast<tflite::BuiltinOperator>(
                  opcode.deprecated_builtin_code)
            : opcode.builtin_code;
    if (resolve_op(builtin_code, opcode.custom_code.c_str(), opcode.version,
                   &op_index_to_registration[i]) != kTfLiteOk) {
      return kTfLiteError;
    }
  }

  auto tmp_interpreter = std::make_unique<tflite::Interpreter>(reporter);
  if (subgraphs.size() > 1) {
    tflite::interpreter_wrapper::InterpreterWrapper::AddSubgraphs(
        tmp_interpreter.get(), subgraphs.size() - 1);
  }
  tmp_interpreter->SetNumThreads(num_threads);
  tflite::interpreter_wrapper::InterpreterWrapper::ApplyOptionsImpl(
      tmp_interpreter.get(), const_cast<tflite::InterpreterOptions*>(&options));

  LiteRtMallocDataAllocator malloc_allocator;
  flatbuffers::FlatBufferBuilder op_fbb(128);
  absl::flat_hash_map<const LiteRtSubgraphT*, int> subgraph_to_index;
  subgraph_to_index.reserve(subgraphs.size());
  std::vector<int> dims, dims_signature, inputs, outputs;

  for (size_t subgraph_index = 0; subgraph_index < subgraphs.size();
       ++subgraph_index) {
    const LiteRtSubgraphT* litert_subgraph = subgraphs[subgraph_index];
    subgraph_to_index.emplace(litert_subgraph,
                              static_cast<int>(subgraph_index));
    tflite::Subgraph* modified_subgraph =
        tmp_interpreter->subgraph(subgraph_index);
    tflite::SingleOpModel::SetAllocation(modified_subgraph, allocation);
    if (!litert_subgraph->Name().empty()) {
      modified_subgraph->SetName(std::string(litert_subgraph->Name()).c_str());
    }

    const auto& tensors = litert_subgraph->Tensors();
    const auto& operators = litert_subgraph->Ops();
    if (modified_subgraph->AddTensors(tensors.size()) != kTfLiteOk) {
      return kTfLiteError;
    }

    absl::flat_hash_map<const LiteRtTensorT*, int32_t> tensor_to_index;
    auto get_tensor_index = [&](const LiteRtTensorT* t) -> int32_t {
      if (!t) return kTfLiteOptionalTensor;
      uint32_t idx = t->TensorIndex();
      if (idx < tensors.size() && tensors[idx] == t) {
        return static_cast<int32_t>(idx);
      }
      if (tensor_to_index.empty()) {
        tensor_to_index.reserve(tensors.size());
        for (size_t j = 0; j < tensors.size(); ++j) {
          tensor_to_index.emplace(tensors[j], static_cast<int32_t>(j));
        }
      }
      auto it = tensor_to_index.find(t);
      return it != tensor_to_index.end() ? it->second : kTfLiteOptionalTensor;
    };

    auto map_tensor_indices = [&](absl::Span<LiteRtTensor const> list,
                                  std::vector<int>& out) {
      out.resize(list.size());
      for (size_t j = 0; j < list.size(); ++j) {
        out[j] = get_tensor_index(list[j]);
      }
    };

    map_tensor_indices(litert_subgraph->Inputs(), inputs);
    modified_subgraph->SetInputs(inputs);
    map_tensor_indices(litert_subgraph->Outputs(), outputs);
    modified_subgraph->SetOutputs(outputs);

    for (size_t i = 0; i < tensors.size(); ++i) {
      const LiteRtTensorT& litert_tensor = *tensors[i];
      const auto& tensor_type = litert_tensor.Type();
      LiteRtElementType element_type = kLiteRtElementTypeNone;
      dims.clear();
      dims_signature.clear();

      if (tensor_type.first == kLiteRtRankedTensorType) {
        const auto& ranked = tensor_type.second.ranked_tensor_type;
        element_type = ranked.element_type;
        dims.resize(ranked.layout.rank);
        bool has_dynamic_dim = false;
        for (size_t d = 0; d < ranked.layout.rank; ++d) {
          int32_t dim = ranked.layout.dimensions[d];
          has_dynamic_dim |= (dim < 0);
          dims[d] = dim < 0 ? 1 : dim;
        }
        if (has_dynamic_dim) {
          dims_signature.assign(ranked.layout.dimensions,
                                ranked.layout.dimensions + ranked.layout.rank);
        }
      } else if (tensor_type.first == kLiteRtUnrankedTensorType) {
        element_type = tensor_type.second.unranked_tensor_type.element_type;
      }
      if (element_type == kLiteRtElementTypeNone) {
        return kTfLiteError;
      }

      const uint32_t external_buffer_id = litert_tensor.ExternalBufferId();
      const auto& weights = litert_tensor.Weights();
      const char* buffer_ptr = nullptr;
      size_t buffer_size = 0;
      size_t buffer_identifier = 0;
      if (weights.GetBufferManager() != nullptr) {
        auto buf = weights.Buffer();
        if (buf.Data() != nullptr) {
          buffer_ptr = buf.StrData();
          buffer_size = buf.Size();
          buffer_identifier = weights.GetBufferId();
        }
      }
      if (buffer_ptr != nullptr && external_buffer_id != 0) {
        return kTfLiteError;
      }

      TfLiteQuantization quantization{};
      if (ParseQuantizationFromLiteRtTensor(litert_tensor, dims,
                                            get_tensor_index, options, reporter,
                                            &quantization) != kTfLiteOk) {
        return kTfLiteError;
      }

      const TfLiteType type = static_cast<TfLiteType>(element_type);
      const char* name_cstr = litert_tensor.Name().data();
      if (buffer_ptr != nullptr) {
        if (modified_subgraph->SetTensorParametersReadOnly(
                i, type, name_cstr, dims, quantization, buffer_ptr, buffer_size,
                allocation, /*sparsity=*/nullptr, buffer_identifier,
                external_buffer_id) != kTfLiteOk) {
          return kTfLiteError;
        }
      } else if (modified_subgraph->SetTensorParametersReadWrite(
                     i, type, name_cstr, dims, quantization,
                     /*is_variable=*/false, dims_signature,
                     external_buffer_id) != kTfLiteOk) {
        return kTfLiteError;
      }
    }

    tflite::SingleOpModel::ReserveNodes(modified_subgraph, operators.size());
    for (size_t i = 0; i < operators.size(); ++i) {
      const LiteRtOpT& op = *operators[i];
      const TfLiteRegistration* registration = nullptr;
      int32_t op_code_ind = GetTflOpCodeInd(op);
      tflite::BuiltinOperator expected_builtin =
          static_cast<tflite::BuiltinOperator>(op.OpCode());
      if (op_code_ind >= 0 &&
          static_cast<size_t>(op_code_ind) < op_index_to_registration.size() &&
          op_codes[op_code_ind]->builtin_code == expected_builtin) {
        registration = op_index_to_registration[op_code_ind];
      } else {
        const char* custom_code = "";
        if (expected_builtin == tflite::BuiltinOperator_CUSTOM) {
          auto custom = op.CustomCode();
          // Both op.CustomCode() (backed by std::string in LiteRtOpT) and
          // kLiteRtDispatchOpCustomName (constexpr string_view) are
          // null-terminated and outlive the interpreter.
          custom_code = (custom.HasValue() && !custom->empty())
                            ? custom->data()
                            : kLiteRtDispatchOpCustomName.data();
        }
        if (resolve_op(expected_builtin, custom_code, /*version=*/1,
                       &registration) != kTfLiteOk) {
          return kTfLiteError;
        }
      }

      tflite::BuiltinOperator op_type =
          static_cast<tflite::BuiltinOperator>(registration->builtin_code);
      void* builtin_data = nullptr;
      const char* init_data = nullptr;
      size_t init_data_size = 0;
      if (op_type == tflite::BuiltinOperator_CUSTOM) {
        auto custom_opts = op.CustomOptions();
        if (custom_opts.Data() != nullptr && custom_opts.Size() > 0) {
          init_data = custom_opts.StrData();
          init_data_size = custom_opts.Size();
        }
      } else {
        const tflite::Operator* fb_op = op.FbOp();
        const bool repacked = (fb_op == nullptr);
        const auto& opts = GetTflOptions(op);
        const auto& opts2 = GetTflOptions2(op);
        if (repacked) {
          op_fbb.Clear();
          auto fb_op_offset = tflite::CreateOperator(
              op_fbb, /*opcode_index=*/0, /*inputs=*/0, /*outputs=*/0,
              opts.type, opts.Pack(op_fbb), /*custom_options=*/0,
              tflite::CustomOptionsFormat_FLEXBUFFERS,
              /*mutating_variable_inputs=*/0, /*intermediates=*/0,
              /*large_custom_options_offset=*/0,
              /*large_custom_options_size=*/0, opts2.type, opts2.Pack(op_fbb));
          op_fbb.Finish(fb_op_offset);
          fb_op =
              flatbuffers::GetRoot<tflite::Operator>(op_fbb.GetBufferPointer());
        }
        TF_LITE_ENSURE_STATUS(tflite::ParseOpData(
            fb_op, op_type, reporter, &malloc_allocator, &builtin_data));
        if (repacked && builtin_data != nullptr) {
          if (const auto* comp = opts2.AsStableHLOCompositeOptions()) {
            auto* p =
                static_cast<TfLiteStablehloCompositeParams*>(builtin_data);
            p->name = comp->name.c_str();
            p->attributes = comp->composite_attributes.data();
          } else if (const auto* fc = opts.AsFullyConnectedOptions()) {
            auto* p = static_cast<TfLiteFullyConnectedParams*>(builtin_data);
            p->quant_spec =
                fc->quant_spec.empty() ? nullptr : fc->quant_spec.data();
          } else if (const auto* bkt = opts.AsBucketizeOptions()) {
            auto* p = static_cast<TfLiteBucketizeParams*>(builtin_data);
            p->boundaries = bkt->boundaries.data();
          } else if (const auto* vh = opts.AsVarHandleOptions()) {
            auto* p = static_cast<TfLiteVarHandleParams*>(builtin_data);
            p->container =
                vh->container.empty() ? nullptr : vh->container.c_str();
            p->shared_name =
                vh->shared_name.empty() ? nullptr : vh->shared_name.c_str();
          }
        }
      }

      const auto* fb_inputs = op.FbOp() ? op.FbOp()->inputs() : nullptr;
      if (fb_inputs && fb_inputs->size() != op.Inputs().size() &&
          std::count(fb_inputs->begin(), fb_inputs->end(),
                     kTfLiteOptionalTensor) +
                  op.Inputs().size() ==
              fb_inputs->size()) {
        inputs.resize(fb_inputs->size());
        size_t idx = 0;
        for (size_t j = 0; j < fb_inputs->size(); ++j) {
          inputs[j] = fb_inputs->Get(j) == kTfLiteOptionalTensor
                          ? kTfLiteOptionalTensor
                          : get_tensor_index(op.Inputs()[idx++]);
        }
      } else {
        map_tensor_indices(op.Inputs(), inputs);
      }
      map_tensor_indices(op.Outputs(), outputs);
      modified_subgraph->AddNodeWithParameters(
          inputs, outputs, /*intermediates=*/{}, init_data, init_data_size,
          builtin_data, registration);
    }
  }

  const auto& model_signatures = model.Signatures();
  const auto* fb_model = GetTflFlatbuffer(model).FlatbufferModelPtr();
  const auto* packed_model = fb_model ? fb_model->GetModel() : nullptr;
  for (const LiteRtSignatureT* sig : model_signatures) {
    if (model_signatures.size() == 1 &&
        sig->Key() == litert::kDefaultSignatureKey &&
        (packed_model == nullptr || packed_model->signature_defs() == nullptr ||
         packed_model->signature_defs()->empty())) {
      continue;
    }
    auto sg_it = subgraph_to_index.find(&sig->GetSubgraph());
    if (sg_it == subgraph_to_index.end()) {
      continue;
    }
    const auto& sig_tensors = sig->GetSubgraph().Tensors();
    auto find_sig_tensor_idx = [&](const LiteRtTensorT* t) -> int32_t {
      if (!t) return -1;
      uint32_t idx = t->TensorIndex();
      if (idx < sig_tensors.size() && sig_tensors[idx] == t) {
        return static_cast<int32_t>(idx);
      }
      for (size_t i = 0; i < sig_tensors.size(); ++i) {
        if (sig_tensors[i] == t) return static_cast<int32_t>(i);
      }
      return -1;
    };
    std::map<std::string, uint32_t> sig_inputs, sig_outputs;
    for (size_t i = 0; i < sig->InputNames().size(); ++i) {
      if (int32_t idx = find_sig_tensor_idx(sig->GetInputTensor(i)); idx >= 0) {
        sig_inputs[sig->InputNames()[i]] = static_cast<uint32_t>(idx);
      }
    }
    for (size_t i = 0; i < sig->OutputNames().size(); ++i) {
      if (int32_t idx = find_sig_tensor_idx(sig->GetOutputTensor(i));
          idx >= 0) {
        sig_outputs[sig->OutputNames()[i]] = static_cast<uint32_t>(idx);
      }
    }
    tflite::interpreter_wrapper::InterpreterWrapper::EmplaceSignatureDef(
        tmp_interpreter.get(), std::move(sig_inputs), std::move(sig_outputs),
        std::string(sig->Key()), static_cast<uint32_t>(sg_it->second));
  }

  if (options.GetUseSignatureTensorNames()) {
    tflite::interpreter_wrapper::InterpreterWrapper::ApplySignatureTensorNames(
        tmp_interpreter.get());
  }

  std::map<std::string, std::string> metadata;
  for (auto it = model.MetadataBegin(); it != model.MetadataEnd(); ++it) {
    if (auto buf = model.FindMetadata(it->first); buf.HasValue()) {
      metadata.emplace(it->first, std::string(buf->StrData(), buf->Size()));
    }
  }
  if (tflite::interpreter_wrapper::InterpreterWrapper::SetMetadata(
          tmp_interpreter.get(), metadata) != kTfLiteOk) {
    return kTfLiteError;
  }

  *interpreter_out = std::move(tmp_interpreter);
  return kTfLiteOk;
}

}  // namespace litert::internal
