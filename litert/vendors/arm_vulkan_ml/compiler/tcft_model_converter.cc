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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/arm_vulkan_ml/compiler/tcft_model_converter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/vendors/arm_vulkan_ml/capabilities.h"
#include "model_converter/model_converter.h"
#include "tosa_converter_for_tflite/tcft.h"

namespace litert::arm_vulkan_ml {

namespace {

int CaptureOutput(const void* data, size_t size, void* user_data) {
  if ((data == nullptr && size != 0) || user_data == nullptr) {
    return 1;
  }
  if (size == 0) {
    return 0;
  }
  auto* output = static_cast<std::vector<char>*>(user_data);
  const auto* bytes = static_cast<const char*>(data);
  try {
    output->insert(output->end(), bytes, bytes + size);
  } catch (...) {
    return 1;
  }
  return 0;
}

const char* StatusMessageOrUnknown(const char* message) {
  return message != nullptr ? message : "unknown status";
}

LiteRtStatus SerializeModelToFlatbuffer(LiteRtModel model,
                                        OwningBufferRef<uint8_t>* serialized) {
  if (model == nullptr || serialized == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::OwningBufferRef<uint8_t> owned_buf;
  auto [buf, size, offset] = owned_buf.GetWeak();
  LiteRtModelSerializationOptions options = {};
  const LiteRtStatus status =
      LiteRtSerializeModel(model, &buf, &size, &offset, false, options);
  if (status != kLiteRtStatusOk) {
    return status;
  }
  if (buf == nullptr || offset > size) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *serialized = std::move(owned_buf);
  return kLiteRtStatusOk;
}

LiteRtStatus IsolatePartition(
    const BufferRef<uint8_t>& serialized_partitions_model,
    size_t partition_index, OwningBufferRef<uint8_t>* serialized_partition) {
  LiteRtModel cloned_model = nullptr;
  const LiteRtStatus load_status = LiteRtCreateModelFromBuffer(
      /*environment=*/nullptr, serialized_partitions_model.Data(),
      serialized_partitions_model.Size(), &cloned_model);
  if (load_status != kLiteRtStatusOk) {
    return load_status;
  }
  std::unique_ptr<LiteRtModelT, decltype(&LiteRtDestroyModel)> owned_model(
      cloned_model, &LiteRtDestroyModel);
  if (partition_index >= owned_model->NumSubgraphs()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  auto isolated_partition =
      std::make_unique<LiteRtModelT>(owned_model->Yank({partition_index}));
  return SerializeModelToFlatbuffer(isolated_partition.get(),
                                    serialized_partition);
}

LiteRtStatus ConvertTfliteToTosa(const BufferRef<uint8_t>& tflite,
                                 std::vector<char>* tosa) {
  const tcft_status tcft_result = tcft_convert_tflite_to_tosa(
      tflite.Data(), tflite.Size(), CaptureOutput, tosa);

  if (tcft_result != TCFT_STATUS_OK || tosa->empty()) {
    LITERT_LOG(LITERT_ERROR, "TCFT conversion failed: %s",
               StatusMessageOrUnknown(tcft_status_message(tcft_result)));
    return kLiteRtStatusErrorCompilation;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertTosaToVgf(const std::vector<char>& tosa,
                              const mc_options& options,
                              std::vector<char>* vgf) {
  vgf->clear();
  const mc_status mc_result = mc_convert_tosa_to_vgf(
      tosa.data(), tosa.size(), &options, CaptureOutput, vgf);

  if (mc_result != MC_STATUS_OK || vgf->empty()) {
    LITERT_LOG(LITERT_ERROR, "Model Converter conversion failed: %s",
               StatusMessageOrUnknown(mc_status_message(mc_result)));
    vgf->clear();
    return kLiteRtStatusErrorCompilation;
  }

  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus PartitionForTcftModelConverter(LiteRtSubgraph subgraph,
                                            LiteRtOpList selected_ops) {
  if (subgraph == nullptr || selected_ops == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  using ElementType = litert::ElementType;

  litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!IsSupportedOpCode(op.Code())) {
      continue;
    }

    const bool tcft_complex_op = op.Code() == kLiteRtOpCodeTflReal ||
                                 op.Code() == kLiteRtOpCodeTflImag ||
                                 op.Code() == kLiteRtOpCodeTflRfft2d;
    const auto is_supported_type = [tcft_complex_op](ElementType type) {
      return IsSupportedType(type) ||
             (tcft_complex_op && type == ElementType::Complex64);
    };
    bool is_supported = true;
    for (const auto& input : op.Inputs()) {
      if (!is_supported_type(input.ElementType())) {
        is_supported = false;
        break;
      }
    }

    if (!is_supported) {
      continue;
    }

    for (const auto& output : op.Outputs()) {
      if (!is_supported_type(output.ElementType())) {
        is_supported = false;
        break;
      }
    }

    if (!is_supported) {
      continue;
    }

    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus CompileWithTcftModelConverter(
    LiteRtModel partitions, TcftModelConverterCompiledModel* compiled_model) {
  if (partitions == nullptr || compiled_model == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const size_t num_partitions = partitions->NumSubgraphs();
  if (num_partitions == 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  mc_options options MC_OPTIONS_INIT;

  OwningBufferRef<uint8_t> serialized_partitions;
  LITERT_RETURN_IF_ERROR(
      SerializeModelToFlatbuffer(partitions, &serialized_partitions));

  TcftModelConverterCompiledModel result;
  result.byte_codes.reserve(num_partitions);
  result.call_infos.reserve(num_partitions);
  result.call_byte_code_indices.reserve(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    // LiteRT normally hands the plugin one fully delegated partition. In that
    // common case, serialized_partitions is already the exact single subgraph
    // model TCFT needs. Avoid loading, yanking, and serializing the same model
    // again, which otherwise keeps two complete FlatBuffers alive while TCFT
    // and Model Converter allocate their intermediate representations.
    const BufferRef<uint8_t>* partition_input = &serialized_partitions;
    OwningBufferRef<uint8_t> isolated_partition;

    if (num_partitions != 1) {
      LITERT_RETURN_IF_ERROR(
          IsolatePartition(serialized_partitions, i, &isolated_partition));
      partition_input = &isolated_partition;
    }

    std::vector<char> tosa;
    LITERT_RETURN_IF_ERROR(ConvertTfliteToTosa(*partition_input, &tosa));

    // TCFT has finished consuming the serialized TFLite buffer. Release it.
    if (num_partitions == 1) {
      serialized_partitions.Reset();
    } else {
      isolated_partition.Reset();
    }

    std::vector<char> vgf;
    LITERT_RETURN_IF_ERROR(ConvertTosaToVgf(tosa, options, &vgf));
    result.byte_codes.emplace_back(std::move(vgf));
    result.call_infos.emplace_back("partition_" + std::to_string(i));
    result.call_byte_code_indices.emplace_back(i);
  }
  *compiled_model = std::move(result);
  return kLiteRtStatusOk;
}

}  // namespace litert::arm_vulkan_ml
