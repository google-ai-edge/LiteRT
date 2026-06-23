/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::tensor {

namespace {

static constexpr int kTfLiteSchemaVersion = 3;

// When a flatbuffer field is created, if no value is given to it, it will be
// elided from the final buffer and we won't be able to update it after the
// buffer is finalized. We use this value to force a field to be added to the
// buffer.
static constexpr size_t kFlatbufferPlaceholderValue = 0xfafafafafafafafa;

static constexpr size_t kFlatbufferAppendedDataAlignment = 64;

// Converts an operation type to its TFLite equivalent.
absl::StatusOr<::tflite::TensorType> ToTfLite(const Type type) {
  switch (type) {
    case Type::kUnknown:
      return absl::FailedPreconditionError(absl::StrFormat(
          "Serialisation of a tensor with '%s' type is not supported",
          ToString(type)));
    case Type::kBOOL:
      return ::tflite::TensorType_BOOL;
    case Type::kI2:
      return ::tflite::TensorType_INT2;
    case Type::kI4:
      return ::tflite::TensorType_INT4;
    case Type::kI8:
      return ::tflite::TensorType_INT8;
    case Type::kI16:
      return ::tflite::TensorType_INT16;
    case Type::kI32:
      return ::tflite::TensorType_INT32;
    case Type::kI64:
      return ::tflite::TensorType_INT64;
    case Type::kU4:
      return absl::FailedPreconditionError(absl::StrFormat(
          "Serialisation of a tensor with '%s' type is not supported",
          ToString(type)));
    case Type::kU8:
      return ::tflite::TensorType_UINT8;
    case Type::kU16:
      return ::tflite::TensorType_UINT16;
    case Type::kU32:
      return ::tflite::TensorType_UINT32;
    case Type::kU64:
      return ::tflite::TensorType_UINT64;
    case Type::kFP16:
      return ::tflite::TensorType_FLOAT16;
    case Type::kFP32:
      return ::tflite::TensorType_FLOAT32;
    case Type::kFP64:
      return ::tflite::TensorType_FLOAT64;
    case Type::kBF16:
      return absl::FailedPreconditionError(absl::StrFormat(
          "Serialisation of a tensor with '%s' type is not supported",
          ToString(type)));
  }
  return absl::UnimplementedError(
      "Type was not handled in the conversion to TFLite flatbuffer value.");
}

absl::StatusOr<Type> FromTfLite(const TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return Type::kUnknown;
    case kTfLiteFloat32:
      return Type::kFP32;
    case kTfLiteInt32:
      return Type::kI32;
    case kTfLiteUInt8:
      return Type::kU8;
    case kTfLiteInt64:
      return Type::kI64;
    case kTfLiteString:
      return absl::FailedPreconditionError("String type is not supported.");
    case kTfLiteBool:
      return absl::FailedPreconditionError("Bool type is not supported.");
    case kTfLiteInt16:
      return Type::kI16;
    case kTfLiteComplex64:
      return absl::FailedPreconditionError("Complex64 type is not supported.");
    case kTfLiteInt8:
      return Type::kI8;
    case kTfLiteFloat16:
      return Type::kFP16;
    case kTfLiteFloat64:
      return Type::kFP64;
    case kTfLiteComplex128:
      return absl::FailedPreconditionError("Complex128 type is not supported.");
    case kTfLiteUInt64:
      return Type::kU64;
    case kTfLiteResource:
      return absl::FailedPreconditionError("Resource type is not supported.");
    case kTfLiteVariant:
      return absl::FailedPreconditionError("Variant type is not supported.");
    case kTfLiteUInt32:
      return Type::kU32;
    case kTfLiteUInt16:
      return Type::kU16;
    case kTfLiteUInt4:
      return Type::kU4;
    case kTfLiteInt4:
      return Type::kI4;
    case kTfLiteBFloat16:
      return Type::kBF16;
    case kTfLiteInt2:
      return Type::kI2;
    case kTfLiteFloat8E4M3FN:
      return absl::FailedPreconditionError(
          "Float8E4M3FN type is not supported.");
    case kTfLiteFloat8E5M2:
      return absl::FailedPreconditionError(
          "Float8E5M2 type is not supported.");
  }
  return absl::UnimplementedError(
      "Type was not handled in the conversion from TFLite value.");
}

std::string DebugInfo(const graph::Tensor& tensor) {
  std::string dbg_str = "tensor {";
  const char* sep = "";

  if (const auto& info = GetInfo(tensor); info.ok()) {
    if (!info->name.empty()) {
      absl::StrAppend(&dbg_str, "name: ", info->name);
      sep = ", ";
    }
    {
      absl::StrAppend(&dbg_str, sep, "type: ", ToString(info->type));
      sep = ", ";
    }
    if (!info->shape.empty()) {
      absl::StrAppend(&dbg_str, sep, "shape: [");
      sep = "";
      for (int dim : info->shape) {
        absl::StrAppend(&dbg_str, sep, dim);
        sep = ", ";
      }
      absl::StrAppend(&dbg_str, "]");
      sep = ", ";
    }
  }
  {
    absl::StrAppend(&dbg_str, sep, "idx: ", tensor.index);
    sep = ", ";
  }
  if (tensor.group) {
    absl::StrAppend(&dbg_str, sep,
                    "created_at: ", tensor.group->loc.file_name(), ":",
                    tensor.group->loc.line());
    sep = ", ";
  }

  if (const auto& maybe_producer = GetProducer(tensor);

      maybe_producer.ok() && *maybe_producer) {
    absl::StrAppend(&dbg_str, sep, "prod: ", (*maybe_producer)->GetName());
    sep = ", ";
  }

  absl::StrAppend(&dbg_str, "}");
  return dbg_str;
}

}  // namespace

ModelFactory::ModelFactory() {
  model_.version = kTfLiteSchemaVersion;
  model_.buffers.push_back(std::make_unique<tflite::BufferT>());
}

// Explores the graph that is reachable from the given output tensors.
absl::Status ModelFactory::Explore(std::vector<TensorHandle> outputs) {
  tensors_.clear();
  operations_.clear();
  execution_plan_.clear();

  if (outputs.empty()) {
    return absl::FailedPreconditionError(
        "No output tensor provided. No graph to explore.");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(execution_plan_,
                              GetExecutionPlan(absl::MakeConstSpan(outputs)));

  std::vector<graph::Tensor> visited_tensors;
  for (const auto& op : execution_plan_) {
    // Find the shared_ptr from the original map.
    for (auto const& [op_ptr, op_info] : operations_) {
      if (op_ptr.get() == op) {
        operations_.insert({op_ptr, OpSerializationInfo{}});
        break;
      }
    }
    for (const auto& input : op->inputs) {
      visited_tensors.push_back(input);
    }
    auto outputs_group = op->outputs_group.lock();
    if (!outputs_group) {
      return absl::NotFoundError("Null pointer to an outputs group.");
    }
    for (int i = 0; i < outputs_group->tensor_infos.size(); ++i) {
      visited_tensors.push_back(graph::GetTensor(i, outputs_group));
    }
  }

  for (const auto& tensor : visited_tensors) {
    tensors_.insert({tensor, TensorSerializationInfo{}});
  }

  for (TensorHandle& o : outputs) {
    tensors_[o.GetRaw()].is_output = true;
  }

  return absl::OkStatus();
}

absl::Status ModelFactory::Build() {
  model_.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
  tflite::SubGraphT& subgraph = *model_.subgraphs.back();

  size_t unnamed_input_idx = 0;
  size_t unnamed_output_idx = 0;

  for (auto& [tensor, build_info] : tensors_) {
    // The const cast is safe: we hash on the tensor group shared_ptr and index
    // and we want to modify the stored info.
    LRT_TENSOR_ASSIGN_OR_RETURN(graph::TensorInformation & tensor_info,
                                GetInfo(const_cast<graph::Tensor&>(tensor)));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const std::shared_ptr<const graph::Operation> producer,
        GetProducer(tensor), _ << DebugInfo(tensor));
    subgraph.tensors.push_back(std::make_unique<tflite::TensorT>());
    build_info.index = subgraph.tensors.size() - 1;
    if (producer == nullptr && tensor_info.buffer == nullptr) {
      subgraph.inputs.push_back(build_info.index);
      if (tensor_info.name.empty()) {
        tensor_info.name = absl::StrCat("unnamed_input_", unnamed_input_idx++);
      }
    }
    if (build_info.is_output || tensor_info.consumers.empty()) {
      subgraph.outputs.push_back(build_info.index);
      if (tensor_info.name.empty()) {
        tensor_info.name =
            absl::StrCat("unnamed_output_", unnamed_output_idx++);
      }
    }
    tflite::TensorT& t = *subgraph.tensors.back();
    t.shape = tensor_info.shape;
    t.name = tensor_info.name;
    // If the producer isn't null, it means that the buffer was set by and eager
    // execution and we ignore it.
    if (tensor_info.buffer && producer == nullptr) {
      auto [it, inserted] =
          buffers_.insert({tensor_info.buffer, BufferSerializationInfo{}});
      if (inserted) {
        it->second.index = buffer_list_.size() + 1;
        buffer_list_.push_back(tensor_info.buffer);
      }
      t.buffer = it->second.index;
    } else {
      t.buffer = 0;  // 0 means no buffer associated.
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(t.type, ToTfLite(tensor_info.type),
                                _ << DebugInfo(tensor));
    if (tensor_info.quantization) {
      if (auto pcq = tensor_info.quantization
                         ->As<const PerChannelAffineQuantization>();
          pcq.ok()) {
        t.quantization = std::make_unique<tflite::QuantizationParametersT>();
        t.quantization->scale = pcq->scales;
        t.quantization->zero_point = pcq->zero_points;
        t.quantization->quantized_dimension = pcq->quantized_dimension;
      } else if (auto bwq = tensor_info.quantization
                                ->As<const BlockwiseQuantization>();
                 bwq.ok()) {
        t.quantization = std::make_unique<tflite::QuantizationParametersT>();
        t.quantization->quantized_dimension = bwq->quantized_dimension;

        // Block quantization scales and zero points are passed as tensors.
        std::vector<fp16_t> scales_f16(bwq->scales.begin(), bwq->scales.end());
        auto scale_buffer = OwningCpuBuffer::Copy<Type::kFP16>(scales_f16);
        auto [it, inserted] =
            buffers_.insert({scale_buffer, BufferSerializationInfo{}});
        if (inserted) {
          it->second.index = buffer_list_.size() + 1;
          buffer_list_.push_back(scale_buffer);
        }
        auto scale_tensor = std::make_unique<tflite::TensorT>();
        scale_tensor->shape = {static_cast<int>(bwq->scales.size())};
        scale_tensor->type = tflite::TensorType_FLOAT16;
        scale_tensor->name = absl::StrCat(tensor_info.name, "_scales");
        scale_tensor->buffer = it->second.index;
        subgraph.tensors.push_back(std::move(scale_tensor));
        int scale_tensor_idx = subgraph.tensors.size() - 1;

        int zero_point_tensor_idx = -1;  // Default to 0 zero point.
        if (!bwq->zero_points.empty()) {
          auto zp_buffer = OwningCpuBuffer::Copy<Type::kI64>(bwq->zero_points);
          auto [it, inserted] =
              buffers_.insert({zp_buffer, BufferSerializationInfo{}});
          if (inserted) {
            it->second.index = buffer_list_.size() + 1;
            buffer_list_.push_back(zp_buffer);
          }
          auto zp_tensor = std::make_unique<tflite::TensorT>();
          zp_tensor->shape = {static_cast<int>(bwq->zero_points.size())};
          zp_tensor->type = tflite::TensorType_INT64;
          zp_tensor->name = absl::StrCat(tensor_info.name, "_zero_points");
          zp_tensor->buffer = it->second.index;
          subgraph.tensors.push_back(std::move(zp_tensor));
          zero_point_tensor_idx = subgraph.tensors.size() - 1;
        }

        tflite::BlockwiseQuantizationT q_details;
        q_details.scales = scale_tensor_idx;
        q_details.zero_points = zero_point_tensor_idx;
        q_details.block_size = bwq->block_size;
        t.quantization->details.Set(std::move(q_details));
      } else {
        return absl::InvalidArgumentError(
            "Unsupported quantization type for TFLite serialization.");
      }
    }
  }

  // Maps an operator code to its index in `model.operator_codes`.
  std::unordered_map<tflite::BuiltinOperator, int> operator_codes;
  for (const graph::Operation* operation : execution_plan_) {
    // [[maybe_unused]] OpSerializationInfo& build_info =
    // operations_[operation];

    const auto* tflite_op = operation->GetExtension<graph::TfLiteOperation>();
    if (tflite_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Operation ", operation->GetName(),
                       " does not implement TfLiteOperation."));
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(graph::TfLiteOpBuildInfo build_info,
                                tflite_op->ToTfLite(*operation));

    auto [it, inserted] = operator_codes.emplace(build_info.builtin_code, -1);
    if (inserted) {
      it->second = model_.operator_codes.size();
      model_.operator_codes.push_back(
          std::make_unique<tflite::OperatorCodeT>());
      tflite::OperatorCodeT& operator_code = *model_.operator_codes.back();
      operator_code.builtin_code = it->first;
      operator_code.deprecated_builtin_code = it->first;
      if (build_info.builtin_code == tflite::BuiltinOperator_CUSTOM) {
        operator_code.custom_code = *build_info.custom_code;
      }
    }
    subgraph.operators.push_back(std::make_unique<tflite::OperatorT>());
    tflite::OperatorT& op = *subgraph.operators.back();
    op.opcode_index = it->second;
    if (build_info.builtin_options.has_value()) {
      op.builtin_options = *build_info.builtin_options;
    }
    if (build_info.builtin_code == tflite::BuiltinOperator_CUSTOM) {
      if (build_info.custom_options != nullptr) {
        op.custom_options = *build_info.custom_options;
      }
    }
    for (auto& input : operation->inputs) {
      const TensorSerializationInfo& info = tensors_[input];
      op.inputs.push_back(info.index);
    }
    std::shared_ptr<graph::TensorGroup> outputs_group =
        operation->outputs_group.lock();
    if (!outputs_group) {
      return absl::FailedPreconditionError(
          "Operation output group pointer is null.");
    }
    for (size_t o = 0; o < outputs_group->tensor_infos.size(); ++o) {
      const TensorSerializationInfo& info =
          tensors_[{.group = outputs_group, .index = o}];
      op.outputs.push_back(info.index);
    }
  }

  // Note: We have to defer setting the buffers' info until we write the file.
  // This is because the data will be appended after the flatbuffer. We need
  // to make sure that we have the final size of the flatbuffer to compute
  // their offsets.
  //
  // We're not sure that this is finished as other subgraphs could be added to
  // the builder.
  model_.buffers.reserve(buffer_list_.size() + 1);
  for (size_t i = model_.buffers.size(); i < buffer_list_.size() + 1; ++i) {
    model_.buffers.push_back(std::make_unique<tflite::BufferT>());
    // If you leave these to 0, the builder won't include them in the final
    // buffer and we won't be able to update them.
    model_.buffers.back()->offset = kFlatbufferPlaceholderValue;
    model_.buffers.back()->size = kFlatbufferPlaceholderValue;
  }

  return absl::OkStatus();
}

static size_t Align(size_t value, size_t alignment) {
  size_t misalign = value % alignment;
  return misalign ? value + (alignment - misalign) : value;
}

// Updates the FINISHED flatbuffer builder TFLite buffer data with the
// corresponding sizes and offsets.
absl::Status ModelFactory::UpdateBufferData(
    flatbuffers::FlatBufferBuilder& fbb) {
  allocation_size_ = fbb.GetSize();

  auto* model = tflite::GetMutableModel(fbb.GetBufferPointer());
  if (!model) {
    return absl::InternalError(
        "Could not build a model from the flatbuffer builder..");
  }

  auto* const buffers = model->buffers();
  if (!buffers) {
    return absl::InternalError(
        "Could not get buffers from the flatbuffer builder.");
  }

  auto current_buffer = buffer_list_.begin();
  for (size_t i = 1; i < buffers->size(); ++i, ++current_buffer) {
    BufferSerializationInfo& buffer_build_info = buffers_[*current_buffer];
    const size_t buffer_size = current_buffer->get()->Lock().size();
    tflite::Buffer* fbb_buffer = buffers->GetMutableObject(i);
    allocation_size_ =
        Align(allocation_size_, kFlatbufferAppendedDataAlignment);
    buffer_build_info.serialization_offset = allocation_size_;
    fbb_buffer->mutate_offset(allocation_size_);
    fbb_buffer->mutate_size(buffer_size);
    allocation_size_ += buffer_size;
  }
  return absl::OkStatus();
}

absl::Status ModelFactory::WriteBufferData(std::ofstream& output_file) {
  if (!output_file) {
    return absl::InternalError(
        "Can't write buffer data to an invalid output file handle.");
  }
  for (const auto& [buffer, build_info] : buffers_) {
    LockedBufferSpan<const char> data = buffer.get()->Lock().As<const char>();
    output_file.seekp(build_info.serialization_offset);
    output_file.write(data.data(), data.size());
  }
  // Extend the file to make sure that the last buffer has at least
  // `XNN_EXTRA_BYTES`.
  output_file.seekp(0, std::ios_base::end);
  const char extra_bytes[XNN_EXTRA_BYTES] = {};
  output_file.write(extra_bytes, XNN_EXTRA_BYTES);
  return absl::OkStatus();
}

absl::Status ModelFactory::AddSubgraph(std::vector<TensorHandle> outputs) {
  LRT_TENSOR_RETURN_IF_ERROR(Explore(std::move(outputs)));
  LRT_TENSOR_RETURN_IF_ERROR(Build());
  return absl::OkStatus();
}

absl::Status ModelFactory::AddSignature(std::vector<TensorHandle> outputs,
                                        std::string name) {
  LRT_TENSOR_RETURN_IF_ERROR(AddSubgraph(std::move(outputs)));
  auto signature_def = std::make_unique<tflite::SignatureDefT>();
  signature_def->signature_key = std::move(name);
  signature_def->subgraph_index = model_.subgraphs.size() - 1;
  for (const int& input_idx : model_.subgraphs.back()->inputs) {
    const std::unique_ptr<tflite::TensorT>& tensor =
        model_.subgraphs.back()->tensors[input_idx];
    auto tensor_map = std::make_unique<tflite::TensorMapT>();
    tensor_map->name = tensor->name;
    tensor_map->tensor_index = input_idx;
    signature_def->inputs.emplace_back(std::move(tensor_map));
  }
  for (const int& output_idx : model_.subgraphs.back()->outputs) {
    const std::unique_ptr<tflite::TensorT>& tensor =
        model_.subgraphs.back()->tensors[output_idx];
    auto tensor_map = std::make_unique<tflite::TensorMapT>();
    tensor_map->name = tensor->name;
    tensor_map->tensor_index = output_idx;
    signature_def->outputs.emplace_back(std::move(tensor_map));
  }
  model_.signature_defs.emplace_back(std::move(signature_def));
  return absl::OkStatus();
}

absl::Status ModelFactory::Save(absl::string_view path) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(tflite::Model::Pack(fbb, &model_), tflite::ModelIdentifier());
  LRT_TENSOR_RETURN_IF_ERROR(UpdateBufferData(fbb));
  std::ofstream output_file(std::string(path),
                            std::ios::binary | std::ios::trunc);
  if (!output_file.is_open()) {
    return absl::InvalidArgumentError("Could not open output file");
  }
  output_file.write(reinterpret_cast<char*>(fbb.GetBufferPointer()),
                    fbb.GetSize());
  LRT_TENSOR_RETURN_IF_ERROR(WriteBufferData(output_file));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<char>> ModelFactory::CreateFlatbuffer() {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(tflite::Model::Pack(fbb, &model_), tflite::ModelIdentifier());
  LRT_TENSOR_RETURN_IF_ERROR(UpdateBufferData(fbb));

  std::vector<char> fb;
  fb.resize(allocation_size_ + XNN_EXTRA_BYTES);
  std::memcpy(fb.data(), fbb.GetBufferPointer(), fbb.GetSize());
  for (const auto& [buffer, build_info] : buffers_) {
    LockedBufferSpan<const char> data = buffer.get()->Lock().As<const char>();
    std::memcpy(fb.data() + build_info.serialization_offset, data.data(),
                data.size());
  }
  return fb;
}

absl::Status Save(std::vector<TensorHandle> outputs, absl::string_view path) {
  ModelFactory serialization;
  LRT_TENSOR_RETURN_IF_ERROR(serialization.AddSubgraph(std::move(outputs)));
  LRT_TENSOR_RETURN_IF_ERROR(serialization.Save(path));
  return absl::OkStatus();
}

absl::Status Save(std::vector<TensorHandle> outputs, std::vector<char>& fb) {
  ModelFactory serialization;
  LRT_TENSOR_RETURN_IF_ERROR(serialization.AddSubgraph(std::move(outputs)));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto fb_temp, serialization.CreateFlatbuffer());
  fb.insert(fb.end(), fb_temp.begin(), fb_temp.end());
  return absl::OkStatus();
}

absl::Status Run(std::vector<TensorHandle> outputs) {
  ModelFactory serialization;
  LRT_TENSOR_RETURN_IF_ERROR(serialization.AddSubgraph(outputs));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<char> fb,
                              serialization.CreateFlatbuffer());
  auto model_ptr =
      tflite::FlatBufferModel::BuildFromBuffer(fb.data(), fb.size());
  if (!model_ptr) {
    return absl::InternalError(
        "Could not build a model from the generated flatbuffer.");
  }
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  LRT_TENSOR_RETURN_IF_ERROR(tflite::InterpreterBuilder(*model_ptr, resolver)(
                                 &interpreter) == kTfLiteOk)
      << "Failed to build the interpreter";
  if (!interpreter) {
    return absl::InternalError("Failed to create the interpreter.");
  }

  // We rely on the fact that setting an input tensor buffer will not make it an
  // input to the model (since it's a constant buffer).
  if (!interpreter->inputs().empty()) {
    return absl::FailedPreconditionError(
        "To eagerly run a model, all the inputs must have an associated "
        "buffer.");
  }

  LRT_TENSOR_RETURN_IF_ERROR(interpreter->AllocateTensors() == kTfLiteOk)
      << "Failed to allocate tensors";
  LRT_TENSOR_RETURN_IF_ERROR(interpreter->Invoke() == kTfLiteOk)
      << "Failed to invoke interpreter";

  for (const int output_idx : interpreter->outputs()) {
    const TfLiteTensor* const output = interpreter->tensor(output_idx);
    for (TensorHandle& output_tensor : outputs) {
      if (output_tensor.GetName() == output->name) {
        LRT_TENSOR_ASSIGN_OR_RETURN(graph::TensorInformation & info,
                                    GetInfo(output_tensor.GetRaw()));
        LRT_TENSOR_ASSIGN_OR_RETURN(info.type, FromTfLite(output->type));
        info.shape.assign(output->dims->data,
                          output->dims->data + output->dims->size);
        info.buffer =
            OwningCpuBuffer::Copy(output->data.raw_const, output->bytes);
        break;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::tensor
