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

#include "litert/runtime/magic_number_utils.h"

#include <cstdint>
#include <cstring>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "litert/c/litert_any.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"
#include "tflite/model_builder.h"

namespace litert::internal {
namespace {

const LiteRtSignatureT* GetSignature(const LiteRtModelT& model,
                                     absl::string_view signature_key) {
  for (const auto* signature : model.Signatures()) {
    if (signature->Key() == signature_key) {
      return signature;
    }
  }
  return nullptr;
}

int GetSubgraphIndex(const LiteRtModelT& model,
                     const LiteRtSubgraphT& subgraph) {
  int index = 0;
  for (const auto* s : model.Subgraphs()) {
    if (s == &subgraph) {
      return index;
    }
    ++index;
  }
  return -1;
}

// Returns the index of the shape parameter in a given op, or -1 if op doesn't
// have a shape parameter.
int GetShapeParamIndex(const LiteRtOpT& op) {
  switch (op.OpCode()) {
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflBroadcastTo:
      return 1;
    case kLiteRtOpCodeTflSlice:
      return 2;
    default:
      return -1;
  }
}

// Logs debug info about the model and magic number configs which is only called
// when magic number configs are set.
void LogDebugInfo(
    const LiteRtModelT& model,
    const LiteRtMagicNumberConfigs& magic_number_configs,
    const LiteRtMagicNumberVerifications* magic_number_verifications) {
  LITERT_LOG(LITERT_INFO, "Loaded: num_subgraphs=%d, num_signatures=%d",
             model.NumSubgraphs(), model.Signatures().size());
  for (const auto* signature : model.Signatures()) {
    const auto& subgraph = signature->GetSubgraph();
    LITERT_LOG(LITERT_INFO,
               "  signature=%s, subgraph_index=%d, num_tensors=%d, "
               "num_inputs=%d, num_outputs=%d, num_ops=%d",
               signature->Key().data(), GetSubgraphIndex(model, subgraph),
               subgraph.Tensors().size(), subgraph.NumInputs(),
               subgraph.NumOutputs(), subgraph.Ops().size());
  }

  LITERT_LOG(LITERT_INFO, "Magic number configs: num_configs=%d",
             magic_number_configs.num_configs);
  for (int i = 0; i < magic_number_configs.num_configs; ++i) {
    const auto& config = magic_number_configs.configs[i];
    LITERT_LOG(LITERT_INFO,
               "  config[%d]: magic_number=%d, target_number=%d, "
               "signature_prefix=%s",
               i, config.magic_number, config.target_number,
               config.signature_prefix ? config.signature_prefix : "null");
  }

  if (magic_number_verifications != nullptr) {
    LITERT_LOG(LITERT_INFO, "Magic number verifications: num_verifications=%d",
               magic_number_verifications->num_verifications);
    for (int i = 0; i < magic_number_verifications->num_verifications; ++i) {
      const auto& verification = magic_number_verifications->verifications[i];
      LITERT_LOG(LITERT_INFO,
                 "  verification[%d]: signature=%s, test_signature=%s", i,
                 verification.signature, verification.test_signature);
    }
  }
}

// Returns the factor of magic_number that divides value, or 0 if no such
// factor exists.
int64_t GetMagicNumberFactor(int64_t value, int64_t magic_number) {
  int64_t factor = value / magic_number;
  return factor * magic_number == value ? factor : 0;
}

// Updates the dimensions of a given tensor whose number is a magic number.
// Returns the number of dimensions updated.
Expected<int> UpdateMagicNumberInDimensions(
    int64_t magic_number, int64_t target_number, LiteRtTensorT& tensor,
    const tflite::FlatBufferModel& fb_model,
    const tflite::SubGraph& tflite_subgraph) {
  if (tensor.Type().first != kLiteRtRankedTensorType) {
    return 0;
  }

  int tidx = tensor.TensorIndex();
  const auto* tflite_tensor = tflite_subgraph.tensors()->Get(tidx);
  LITERT_RETURN_IF_ERROR(tflite_tensor != nullptr);

  int num_updated = 0;
  auto& layout = tensor.Type().second.ranked_tensor_type.layout;
  for (int i = 0; i < layout.rank; ++i) {
    int64_t factor = GetMagicNumberFactor(layout.dimensions[i], magic_number);
    if (factor == 0) {
      continue;
    }

    LITERT_LOG(LITERT_DEBUG,
               "Update shape[%d] of tensor=%d from %d to %d, factor=%d", i,
               tensor.TensorIndex(), magic_number, target_number, factor);
    layout.dimensions[i] = target_number * factor;
    ++num_updated;

    // Update the flatbuffer accordingly.
    const_cast<::flatbuffers::Vector<int32_t>*>(tflite_tensor->shape())
        ->Mutate(i, layout.dimensions[i]);
  }

  return num_updated;
}

// A util template to update magic number in the flatbuffer for a given type.
// Returns the number of elements updated.
template <typename T>
int UpdateMagicNumber(T magic_number, T target_number, LiteRtOpCode op_code,
                      unsigned char* data, const unsigned char* data_end) {
  int num_updated = 0;
  for (int i = 0; data < data_end; i += sizeof(T), data += sizeof(T)) {
    T value = 0;
    memcpy(&value, data, sizeof(T));
    int64_t factor = GetMagicNumberFactor(value, magic_number);
    if (factor == 0) {
      continue;
    }

    value = target_number * factor;
    LITERT_LOG(LITERT_DEBUG,
               "Update shape param data[%d] of op=%d from %d to %d, factor=%d",
               i, op_code, magic_number, target_number, factor);
    memcpy(data, &value, sizeof(T));
    ++num_updated;
  }
  return num_updated;
}

// Updates the shape parameter of a given tensor in the flatbuffer.
// Returns the number of elements updated.
Expected<int> UpdateMagicNumberInParam(
    int64_t magic_number, int64_t target_number, LiteRtOpCode op_code,
    const LiteRtTensorT& tensor, const tflite::FlatBufferModel& fb_model,
    const tflite::SubGraph& tflite_subgraph) {
  int shape_param_tensor_index = tensor.TensorIndex();
  const auto* shape_param_tensor =
      tflite_subgraph.tensors()->Get(shape_param_tensor_index);
  LITERT_RETURN_IF_ERROR(shape_param_tensor != nullptr);

  auto buffer_index = shape_param_tensor->buffer();
  const auto* buffer = fb_model->buffers()->Get(buffer_index);
  LITERT_RETURN_IF_ERROR(buffer != nullptr);

  unsigned char* data = const_cast<unsigned char*>(buffer->data()->data());
  const unsigned char* data_end = data + buffer->data()->size();
  if (shape_param_tensor->type() == tflite::TensorType_INT32) {
    return UpdateMagicNumber<int32_t>(magic_number, target_number, op_code,
                                      data, data_end);
  } else if (shape_param_tensor->type() == tflite::TensorType_INT64) {
    return UpdateMagicNumber<int64_t>(magic_number, target_number, op_code,
                                      data, data_end);
  }
  return 0;
}

Expected<int> ReplaceMagicNumberInSubgraph(
    int64_t magic_number, int64_t target_number, LiteRtModelT& model,
    const tflite::FlatBufferModel& fb_model,
    const tflite::SubGraph& tflite_subgraph, LiteRtSubgraphT& subgraph) {
  int updated_tensors = 0;
  for (auto* tensor : subgraph.Tensors()) {
    LITERT_ASSIGN_OR_RETURN(
        int num_updated,
        UpdateMagicNumberInDimensions(magic_number, target_number, *tensor,
                                      fb_model, tflite_subgraph));
    updated_tensors += num_updated;
  }

  for (auto* op : subgraph.Ops()) {
    // Update subgraphs of this subgraph.
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      const auto* opts =
          litert::internal::GetTflOptions2(*op).AsStableHLOCompositeOptions();
      LITERT_RETURN_IF_ERROR(opts != nullptr);
      auto decomp_index = opts->decomposition_subgraph_index;
      LITERT_RETURN_IF_ERROR(decomp_index < model.NumSubgraphs());
      const auto* decomp_subgraph = fb_model->subgraphs()->Get(decomp_index);
      LITERT_RETURN_IF_ERROR(decomp_subgraph != nullptr);
      LITERT_ASSIGN_OR_RETURN(
          auto num_tensors_updated,
          ReplaceMagicNumberInSubgraph(magic_number, target_number, model,
                                       fb_model, *decomp_subgraph,
                                       model.Subgraph(decomp_index)));
      if (num_tensors_updated > 0) {
        LITERT_LOG(
            LITERT_DEBUG,
            "%d tensors of subgraph %d have been updated for magic number %d.",
            num_tensors_updated, decomp_index, magic_number);
      }
      updated_tensors += num_tensors_updated;
      continue;
    }

    // Update shape parameters if any.
    int shape_param_index = GetShapeParamIndex(*op);
    if (shape_param_index == -1) {
      continue;
    }
    LITERT_RETURN_IF_ERROR(shape_param_index < op->NumInputs());
    LITERT_ASSIGN_OR_RETURN(
        int num_updated,
        UpdateMagicNumberInParam(magic_number, target_number, op->OpCode(),
                                 op->Input(shape_param_index), fb_model,
                                 tflite_subgraph));
    updated_tensors += num_updated;
  }

  return updated_tensors;
}

Expected<int> UpdateMagicNumbersInModel(
    const LiteRtMagicNumberConfigs& magic_number_configs, LiteRtModelT& model) {
  const auto& fb_model =
      litert::internal::GetTflFlatbuffer(model).FlatbufferModel();
  int total_updated_tensors = 0;
  for (auto* signature : model.Signatures()) {
    auto& subgraph = signature->GetSubgraph();
    int subgraph_index = GetSubgraphIndex(model, subgraph);
    const auto* tflite_subgraph = fb_model->subgraphs()->Get(subgraph_index);
    LITERT_RETURN_IF_ERROR(tflite_subgraph != nullptr);

    for (int i = 0; i < magic_number_configs.num_configs; ++i) {
      const auto& config = magic_number_configs.configs[i];
      auto signature_prefix = absl::NullSafeStringView(config.signature_prefix);
      if (!signature_prefix.empty() &&
          !absl::StartsWith(signature->Key(), signature_prefix)) {
        continue;
      }

      LITERT_LOG(LITERT_DEBUG, "Replacing magic number %d in signature=%s",
                 config.magic_number, signature->Key().data());
      LITERT_ASSIGN_OR_RETURN(int updated_tensors,
                              ReplaceMagicNumberInSubgraph(
                                  config.magic_number, config.target_number,
                                  model, fb_model, *tflite_subgraph, subgraph));
      if (updated_tensors == 0) {
        LITERT_LOG(LITERT_DEBUG, "No magic number %d found in signature=%s",
                   config.magic_number, signature->Key().data());
        continue;
      }
      total_updated_tensors += updated_tensors;
      LITERT_LOG(
          LITERT_INFO,
          "%d tensors of signature %s have been updated for magic number %d.",
          updated_tensors, signature->Key().data(), config.magic_number);
    }
  }

  return total_updated_tensors;
}

Expected<void> VerifySameShape(const LiteRtTensorT& t1, const LiteRtTensorT& t2,
                               const tflite::FlatBufferModel& fb_model,
                               const tflite::SubGraph& tflite_subgraph) {
  LITERT_RETURN_IF_ERROR(t1.Type().first == t2.Type().first);
  if (t1.Type().first == kLiteRtUnrankedTensorType) {
    return {};
  }
  LITERT_RETURN_IF_ERROR(t1.Type().first == kLiteRtRankedTensorType);

  const auto& layout_1 = t1.Type().second.ranked_tensor_type.layout;
  const auto& layout_2 = t2.Type().second.ranked_tensor_type.layout;
  LITERT_RETURN_IF_ERROR(layout_1.rank == layout_2.rank);

  for (int i = 0; i < layout_1.rank; ++i) {
    LITERT_RETURN_IF_ERROR(layout_1.dimensions[i] == layout_2.dimensions[i]);
  }

  int tidx1 = t1.TensorIndex();
  const auto* tflite_t1 = tflite_subgraph.tensors()->Get(tidx1);
  LITERT_RETURN_IF_ERROR(tflite_t1 != nullptr);

  int tidx2 = t2.TensorIndex();
  const auto* tflite_t2 = tflite_subgraph.tensors()->Get(tidx2);
  LITERT_RETURN_IF_ERROR(tflite_t2 != nullptr);

  LITERT_RETURN_IF_ERROR(tflite_t1->type() == tflite_t2->type());

  auto shape1 = tflite_t1->shape();
  auto shape2 = tflite_t2->shape();
  LITERT_RETURN_IF_ERROR(shape1->size() == shape2->size());
  for (int i = 0; i < shape1->size(); ++i) {
    LITERT_RETURN_IF_ERROR(shape1->Get(i) == shape2->Get(i));
  }

  return {};
}

Expected<void> VerifySameData(const LiteRtTensorT& t1, const LiteRtTensorT& t2,
                              const tflite::FlatBufferModel& fb_model,
                              const tflite::SubGraph& tflite_subgraph) {
  int tidx1 = t1.TensorIndex();
  const auto* tflite_t1 = tflite_subgraph.tensors()->Get(tidx1);
  LITERT_RETURN_IF_ERROR(tflite_t1 != nullptr);

  int tidx2 = t2.TensorIndex();
  const auto* tflite_t2 = tflite_subgraph.tensors()->Get(tidx2);
  LITERT_RETURN_IF_ERROR(tflite_t2 != nullptr);

  LITERT_RETURN_IF_ERROR(tflite_t1->type() == tflite_t2->type());

  auto bidx1 = tflite_t1->buffer();
  const auto* b1 = fb_model->buffers()->Get(bidx1);
  LITERT_RETURN_IF_ERROR(b1 != nullptr);

  auto bidx2 = tflite_t2->buffer();
  const auto* b2 = fb_model->buffers()->Get(bidx2);
  LITERT_RETURN_IF_ERROR(b2 != nullptr);

  LITERT_RETURN_IF_ERROR(b1->data()->size() == b2->data()->size());

  LITERT_RETURN_IF_ERROR(
      memcmp(b1->data()->data(), b2->data()->data(), b1->data()->size()) == 0);
  return {};
}

Expected<void> VerifyWithTestSignatures(
    const LiteRtModelT& model,
    const LiteRtMagicNumberVerifications& verifications) {
  const auto& fb_model =
      litert::internal::GetTflFlatbuffer(model).FlatbufferModel();
  for (int i = 0; i < verifications.num_verifications; ++i) {
    LITERT_LOG(LITERT_INFO, "Verifying signature %s with test signature %s.",
               verifications.verifications[i].signature,
               verifications.verifications[i].test_signature);
    const auto* signature =
        GetSignature(model, verifications.verifications[i].signature);
    LITERT_RETURN_IF_ERROR(signature != nullptr);
    const auto& subgraph = signature->GetSubgraph();
    int subgraph_index = GetSubgraphIndex(model, subgraph);
    const auto* tflite_subgraph = fb_model->subgraphs()->Get(subgraph_index);
    LITERT_RETURN_IF_ERROR(tflite_subgraph != nullptr);

    const auto* test_signature =
        GetSignature(model, verifications.verifications[i].test_signature);
    LITERT_RETURN_IF_ERROR(test_signature != nullptr);
    const auto& test_subgraph = test_signature->GetSubgraph();
    LITERT_RETURN_IF_ERROR(
        subgraph.Tensors().size() == test_subgraph.Tensors().size(),
        _ << "Tensor size mismatch: " << subgraph.Tensors().size() << " vs "
          << test_subgraph.Tensors().size());
    LITERT_RETURN_IF_ERROR(subgraph.NumInputs() == test_subgraph.NumInputs());
    LITERT_RETURN_IF_ERROR(subgraph.NumOutputs() == test_subgraph.NumOutputs());
    LITERT_RETURN_IF_ERROR(subgraph.Ops().size() == test_subgraph.Ops().size());

    for (int i = 0; i < subgraph.Tensors().size(); ++i) {
      LITERT_RETURN_IF_ERROR(VerifySameShape(subgraph.Tensor(i),
                                             test_subgraph.Tensor(i), fb_model,
                                             *tflite_subgraph));
    }

    for (int i = 0; i < subgraph.Ops().size(); ++i) {
      const auto& op = subgraph.Op(i);
      const auto& test_op = test_subgraph.Op(i);
      LITERT_RETURN_IF_ERROR(op.OpCode() == test_op.OpCode());
      LITERT_RETURN_IF_ERROR(op.NumInputs() == test_op.NumInputs());
      LITERT_RETURN_IF_ERROR(op.NumOutputs() == test_op.NumOutputs());
      for (int j = 0; j < op.NumInputs(); ++j) {
        LITERT_RETURN_IF_ERROR(VerifySameShape(op.Input(j), test_op.Input(j),
                                               fb_model, *tflite_subgraph));
      }
      for (int j = 0; j < op.NumOutputs(); ++j) {
        LITERT_RETURN_IF_ERROR(VerifySameShape(op.Output(j), test_op.Output(j),
                                               fb_model, *tflite_subgraph));
      }
      int shape_param_index = GetShapeParamIndex(op);
      if (shape_param_index >= 0) {
        LITERT_RETURN_IF_ERROR(shape_param_index < op.NumInputs());
        LITERT_RETURN_IF_ERROR(VerifySameData(op.Input(shape_param_index),
                                              test_op.Input(shape_param_index),
                                              fb_model, *tflite_subgraph));
      }
    }
    LITERT_LOG(LITERT_INFO, "Verified signature %s with test signature %s.",
               verifications.verifications[i].signature,
               verifications.verifications[i].test_signature);
  }
  return {};
}

}  // namespace

Expected<int> ReplaceMagicNumbersIfAny(const LiteRtEnvironmentT& env,
                                       LiteRtModelT& model) {
  auto option_configs = env.GetOption(kLiteRtEnvOptionTagMagicNumberConfigs);
  if (!option_configs) {
    LITERT_LOG(LITERT_DEBUG, "No magic number configs found.");
    return 0;
  }

  const auto* magic_number_configs =
      reinterpret_cast<const LiteRtMagicNumberConfigs*>(
          option_configs->ptr_value);
  if (magic_number_configs->num_configs == 0) {
    LITERT_LOG(LITERT_WARNING, "Magic number configs are empty.");
    return 0;
  }

  const LiteRtMagicNumberVerifications* magic_number_verifications = nullptr;
  auto option_verifications =
      env.GetOption(kLiteRtEnvOptionTagMagicNumberVerifications);
  if (!option_verifications) {
    LITERT_LOG(LITERT_DEBUG, "No magic number verifications found.");
  } else {
    magic_number_verifications =
        reinterpret_cast<const LiteRtMagicNumberVerifications*>(
            option_verifications->ptr_value);
    if (magic_number_verifications->num_verifications == 0) {
      LITERT_LOG(LITERT_WARNING, "Magic number verifications are empty.");
      magic_number_verifications = nullptr;
    }
  }

  LogDebugInfo(model, *magic_number_configs, magic_number_verifications);

  LITERT_ASSIGN_OR_RETURN(
      int total_updated_tensors,
      UpdateMagicNumbersInModel(*magic_number_configs, model));

  if (magic_number_verifications != nullptr) {
    LITERT_RETURN_IF_ERROR(
        VerifyWithTestSignatures(model, *magic_number_verifications));
  }

  return total_updated_tensors;
}

}  // namespace litert::internal
