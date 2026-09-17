/* Copyright 2026 Google LLC.

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

#include "tensor/runners/model_chain.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "tensor/runners/litert/litert_buffer.h"

namespace litert::tensor {

namespace {

litert::TensorBufferType GetPreferredBufferType(
    absl::Span<const litert::TensorBufferType> supported_types) {
  bool has_ahwb = false;
  bool has_opencl = false;
  bool has_host = false;
  for (auto t : supported_types) {
    if (t == litert::TensorBufferType::kAhwb) has_ahwb = true;
    if (t == litert::TensorBufferType::kOpenClBuffer) has_opencl = true;
    if (t == litert::TensorBufferType::kHostMemory) has_host = true;
  }
  if (has_ahwb) return litert::TensorBufferType::kAhwb;
  if (has_opencl) return litert::TensorBufferType::kOpenClBuffer;
  if (has_host) return litert::TensorBufferType::kHostMemory;
  return supported_types.empty() ? litert::TensorBufferType::kHostMemory
                                 : supported_types.front();
}

struct ConsumerTarget {
  std::shared_ptr<ModelStage> to_stage;
  std::string input_name;
};

absl::StatusOr<HardwareBufferDescriptor> NegotiatePortDescriptor(
    const std::shared_ptr<ModelStage>& from_stage,
    absl::string_view output_name, absl::Span<const ConsumerTarget> consumers) {
  auto prod_desc_or = from_stage->GetOutputDescriptor(output_name);
  if (!prod_desc_or.ok()) return prod_desc_or.status();

  HardwareBufferDescriptor harmonized = *prod_desc_or;
  for (const auto& consumer : consumers) {
    auto cons_desc_or =
        consumer.to_stage->GetInputDescriptor(consumer.input_name);
    if (!cons_desc_or.ok()) return cons_desc_or.status();

    auto new_harmonized_or = BoundaryLayoutNegotiator::HarmonizeStageBoundary(
        harmonized, *cons_desc_or);
    if (!new_harmonized_or.ok()) return new_harmonized_or.status();
    harmonized = *new_harmonized_or;
  }
  return harmonized;
}

}  // namespace

// ============================================================================
// FunctionalModelStage Implementation
// ============================================================================

FunctionalModelStage::FunctionalModelStage(
    std::string name,
    absl::flat_hash_map<std::string, HardwareBufferDescriptor>
        input_descriptors,
    absl::flat_hash_map<std::string, HardwareBufferDescriptor>
        output_descriptors,
    ExecuteFn execute_fn)
    : name_(std::move(name)),
      input_descriptors_(std::move(input_descriptors)),
      output_descriptors_(std::move(output_descriptors)),
      execute_fn_(std::move(execute_fn)) {}

std::vector<std::string> FunctionalModelStage::InputNames() const {
  std::vector<std::string> names;
  names.reserve(input_descriptors_.size());
  for (const auto& [name, _] : input_descriptors_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> FunctionalModelStage::OutputNames() const {
  std::vector<std::string> names;
  names.reserve(output_descriptors_.size());
  for (const auto& [name, _] : output_descriptors_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

absl::StatusOr<HardwareBufferDescriptor>
FunctionalModelStage::GetInputDescriptor(absl::string_view name) const {
  auto it = input_descriptors_.find(name);
  if (it == input_descriptors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no input named '", name, "'."));
  }
  return it->second;
}

absl::StatusOr<HardwareBufferDescriptor>
FunctionalModelStage::GetOutputDescriptor(absl::string_view name) const {
  auto it = output_descriptors_.find(name);
  if (it == output_descriptors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no output named '", name, "'."));
  }
  return it->second;
}

absl::Status FunctionalModelStage::SetInputBuffer(
    absl::string_view name, std::shared_ptr<LitertBuffer> buffer) {
  if (!input_descriptors_.contains(name)) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no input named '", name, "'."));
  }
  input_buffers_.insert_or_assign(name, std::move(buffer));
  return absl::OkStatus();
}

absl::Status FunctionalModelStage::SetOutputBuffer(
    absl::string_view name, std::shared_ptr<LitertBuffer> buffer) {
  if (!output_descriptors_.contains(name)) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no output named '", name, "'."));
  }
  output_buffers_.insert_or_assign(name, std::move(buffer));
  return absl::OkStatus();
}

std::shared_ptr<LitertBuffer> FunctionalModelStage::GetInputBuffer(
    absl::string_view name) const {
  auto it = input_buffers_.find(name);
  return (it != input_buffers_.end()) ? it->second : nullptr;
}

std::shared_ptr<LitertBuffer> FunctionalModelStage::GetOutputBuffer(
    absl::string_view name) const {
  auto it = output_buffers_.find(name);
  return (it != output_buffers_.end()) ? it->second : nullptr;
}

absl::Status FunctionalModelStage::Run() {
  // Ensure all required inputs and outputs are bound.
  for (const auto& [in_name, _] : input_descriptors_) {
    auto it = input_buffers_.find(in_name);
    if (it == input_buffers_.end() || it->second == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Stage '", name_, "': required input buffer '", in_name,
                       "' is not bound."));
    }
  }
  for (const auto& [out_name, desc] : output_descriptors_) {
    auto it = output_buffers_.find(out_name);
    if (it == output_buffers_.end() || it->second == nullptr) {
      // If the stage is part of a ModelChain, env_ is guaranteed to be set
      // in ModelChain::Builder::Build(). The fallback below handles standalone
      // stage execution outside of a ModelChain.
      if (!env_) {
        auto env_or = litert::Environment::Create({});
        if (!env_or.HasValue()) {
          return absl::InternalError("Failed to create LiteRT environment");
        }
        env_ = std::make_shared<litert::Environment>(std::move(*env_or));
      }
      auto buf_or = LitertBuffer::CreateManaged(env_, desc.buffer_type,
                                                desc.ToRankedTensorType(),
                                                desc.PackedBytes());
      if (!buf_or.ok()) {
        return buf_or.status();
      }
      if (it != output_buffers_.end()) {
        it->second = *buf_or;
      } else {
        output_buffers_.insert_or_assign(out_name, *buf_or);
      }
    }
  }
  return execute_fn_(input_buffers_, output_buffers_);
}

// ============================================================================
// CompiledModelStage Implementation
// ============================================================================

CompiledModelStage::CompiledModelStage(
    std::shared_ptr<litert::Environment> env, std::string name,
    CompiledModel compiled_model, size_t signature_index,
    std::vector<std::string> input_names, std::vector<std::string> output_names,
    absl::flat_hash_map<std::string, HardwareBufferDescriptor>
        input_descriptors,
    absl::flat_hash_map<std::string, HardwareBufferDescriptor>
        output_descriptors,
    std::string model_path, std::vector<uint8_t> model_buffer,
    std::optional<litert::Options> options)
    : env_(std::move(env)),
      name_(std::move(name)),
      compiled_model_(std::move(compiled_model)),
      signature_index_(signature_index),
      input_names_(std::move(input_names)),
      output_names_(std::move(output_names)),
      input_descriptors_(std::move(input_descriptors)),
      output_descriptors_(std::move(output_descriptors)),
      model_path_(std::move(model_path)),
      model_buffer_(std::move(model_buffer)),
      options_(std::move(options)) {}

absl::Status CompiledModelStage::RefreshDescriptorsFromCompiledModel() {
  auto in_names_res = compiled_model_.GetSignatureInputNames(signature_index_);
  if (!in_names_res.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to get signature input names for stage '", name_,
                     "': ", in_names_res.Error().Message()));
  }

  auto out_names_res =
      compiled_model_.GetSignatureOutputNames(signature_index_);
  if (!out_names_res.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to get signature output names for stage '", name_,
                     "': ", out_names_res.Error().Message()));
  }

  input_names_.clear();
  input_descriptors_.clear();
  for (size_t i = 0; i < in_names_res->size(); ++i) {
    std::string in_name = std::string((*in_names_res)[i]);
    input_names_.push_back(in_name);

    auto tensor_type_res =
        compiled_model_.GetInputTensorType(signature_index_, i);
    if (!tensor_type_res.HasValue()) {
      return absl::InternalError(absl::StrCat(
          "Failed to get input tensor type for '", in_name, "' in stage '",
          name_, "': ", tensor_type_res.Error().Message()));
    }

    HardwareBufferDescriptor desc;
    desc.element_type = tensor_type_res->ElementType();
    auto dims = tensor_type_res->Layout().Dimensions();
    desc.shape.assign(dims.begin(), dims.end());
    auto bytes_res = tensor_type_res->Bytes();
    if (bytes_res.HasValue()) {
      desc.size_bytes = *bytes_res;
    }

    auto req_res =
        compiled_model_.GetInputBufferRequirements(signature_index_, i);
    if (req_res.HasValue()) {
      if (auto sz = req_res->BufferSize();
          sz.HasValue() && *sz > desc.size_bytes) {
        desc.size_bytes = *sz;
      }
      if (auto align = req_res->Alignment(); align.HasValue() && *align > 0) {
        desc.alignment = *align;
      }
      if (auto types = req_res->SupportedTypes();
          types.HasValue() && !types->empty()) {
        desc.buffer_type = GetPreferredBufferType(*types);
      }
    }

    input_descriptors_[in_name] = std::move(desc);
  }

  output_names_.clear();
  output_descriptors_.clear();
  for (size_t i = 0; i < out_names_res->size(); ++i) {
    std::string out_name = std::string((*out_names_res)[i]);
    output_names_.push_back(out_name);

    auto tensor_type_res =
        compiled_model_.GetOutputTensorType(signature_index_, i);
    if (!tensor_type_res.HasValue()) {
      return absl::InternalError(absl::StrCat(
          "Failed to get output tensor type for '", out_name, "' in stage '",
          name_, "': ", tensor_type_res.Error().Message()));
    }

    HardwareBufferDescriptor desc;
    desc.element_type = tensor_type_res->ElementType();
    auto dims = tensor_type_res->Layout().Dimensions();
    desc.shape.assign(dims.begin(), dims.end());
    auto bytes_res = tensor_type_res->Bytes();
    if (bytes_res.HasValue()) {
      desc.size_bytes = *bytes_res;
    }

    auto req_res =
        compiled_model_.GetOutputBufferRequirements(signature_index_, i);
    if (req_res.HasValue()) {
      if (auto sz = req_res->BufferSize();
          sz.HasValue() && *sz > desc.size_bytes) {
        desc.size_bytes = *sz;
      }
      if (auto align = req_res->Alignment(); align.HasValue() && *align > 0) {
        desc.alignment = *align;
      }
      if (auto types = req_res->SupportedTypes();
          types.HasValue() && !types->empty()) {
        desc.buffer_type = GetPreferredBufferType(*types);
      }
    }

    output_descriptors_[out_name] = std::move(desc);
  }

  return absl::OkStatus();
}

absl::Status CompiledModelStage::PrepareStageBoundary(
    const absl::flat_hash_map<std::string, HardwareBufferDescriptor>&
        negotiated_inputs,
    const absl::flat_hash_map<std::string, HardwareBufferDescriptor>&
        negotiated_outputs) {
  // NOTE: Re-compiling a model in PrepareStageBoundary can be expensive (e.g.,
  // AOT compilation for NPUs or GPUs). Users who already know upfront that
  // they will use GPU or external buffers are encouraged to configure
  // GpuOptions::EnableExternalTensorsMode in their Options before creating
  // the stage.
  auto IsExternalGpuBuffer = [](litert::TensorBufferType t) {
    return t == litert::TensorBufferType::kAhwb ||
           t == litert::TensorBufferType::kOpenClBuffer ||
           litert::IsWebGpuMemory(t);
  };

  std::vector<std::string> external_tensors;
  for (const auto& [name, desc] : negotiated_inputs) {
    if (IsExternalGpuBuffer(desc.buffer_type)) {
      external_tensors.push_back(name);
    }
  }
  for (const auto& [name, desc] : negotiated_outputs) {
    if (IsExternalGpuBuffer(desc.buffer_type)) {
      external_tensors.push_back(name);
    }
  }

  if (external_tensors.empty()) {
    return absl::OkStatus();
  }

  if (model_path_.empty() && model_buffer_.empty()) {
    ABSL_LOG(WARNING)
        << "Stage '" << name_ << "' was connected to external GPU buffers ("
        << absl::StrJoin(external_tensors, ", ")
        << "), but lacks model source (path/buffer) to recompile with external "
           "tensors mode because it was constructed from an existing "
           "CompiledModel. Execution may fail or fall back to CPU copies.";
    return absl::OkStatus();
  }

  if (!options_.has_value()) {
    ABSL_LOG(WARNING)
        << "Stage '" << name_ << "' was connected to external GPU buffers ("
        << absl::StrJoin(external_tensors, ", ")
        << "), but lacks Options to recompile with external tensors mode.";
    return absl::OkStatus();
  }

  auto gpu_options_or = options_->GetGpuOptions();
  if (!gpu_options_or.HasValue()) {
    return absl::OkStatus();
  }

  gpu_options_or->EnableExternalTensorsMode(true);
  for (const auto& tensor_name : external_tensors) {
    gpu_options_or->AddExternalTensorPattern(tensor_name.c_str());
  }

  if (!model_path_.empty()) {
    auto model_res = CompiledModel::Create(*env_, model_path_, *options_);
    if (!model_res.HasValue()) {
      return absl::InternalError(absl::StrCat(
          "Failed to recompile stage '", name_,
          "' with external GPU tensors mode: ", model_res.Error().Message()));
    }
    compiled_model_ = std::move(*model_res);
  } else {
    BufferRef<uint8_t> buf_ref(model_buffer_.data(), model_buffer_.size());
    auto model_res = CompiledModel::Create(*env_, buf_ref, *options_);
    if (!model_res.HasValue()) {
      return absl::InternalError(absl::StrCat(
          "Failed to recompile stage '", name_,
          "' with external GPU tensors mode: ", model_res.Error().Message()));
    }
    compiled_model_ = std::move(*model_res);
  }

  return RefreshDescriptorsFromCompiledModel();
}

absl::StatusOr<std::shared_ptr<CompiledModelStage>> CompiledModelStage::Create(
    std::shared_ptr<litert::Environment> env, std::string name,
    CompiledModel compiled_model, size_t signature_index) {
  auto stage = std::shared_ptr<CompiledModelStage>(new CompiledModelStage(
      std::move(env), std::move(name), std::move(compiled_model),
      signature_index, {}, {}, {}, {}));
  auto status = stage->RefreshDescriptorsFromCompiledModel();
  if (!status.ok()) return status;
  return stage;
}

absl::StatusOr<std::shared_ptr<CompiledModelStage>> CompiledModelStage::Create(
    std::shared_ptr<litert::Environment> env, std::string name,
    const std::string& model_path, litert::Options options,
    size_t signature_index) {
  if (!env) {
    auto env_or = litert::Environment::Create({});
    if (!env_or.HasValue()) {
      return absl::InternalError("Failed to create LiteRT environment");
    }
    env = std::make_shared<litert::Environment>(std::move(*env_or));
  }
  auto model_res = CompiledModel::Create(*env, model_path, options);
  if (!model_res.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to load and compile model from '", model_path,
                     "': ", model_res.Error().Message()));
  }
  auto stage_or = Create(std::move(env), std::move(name), std::move(*model_res),
                         signature_index);
  if (!stage_or.ok()) return stage_or;
  (*stage_or)->model_path_ = model_path;
  (*stage_or)->options_ = std::move(options);
  return stage_or;
}

absl::StatusOr<std::shared_ptr<CompiledModelStage>> CompiledModelStage::Create(
    std::shared_ptr<litert::Environment> env, std::string name,
    const std::string& model_path, litert::HwAccelerators accelerators,
    size_t signature_index) {
  auto options_or = litert::Options::Create();
  if (!options_or.HasValue()) {
    return absl::InternalError("Failed to create LiteRT options");
  }
  options_or->SetHardwareAccelerators(accelerators);
  return Create(std::move(env), std::move(name), model_path,
                std::move(*options_or), signature_index);
}

absl::StatusOr<std::shared_ptr<CompiledModelStage>> CompiledModelStage::Create(
    std::shared_ptr<litert::Environment> env, std::string name,
    absl::Span<const uint8_t> model_buffer, litert::Options options,
    size_t signature_index) {
  if (!env) {
    auto env_or = litert::Environment::Create({});
    if (!env_or.HasValue()) {
      return absl::InternalError("Failed to create LiteRT environment");
    }
    env = std::make_shared<litert::Environment>(std::move(*env_or));
  }
  BufferRef<uint8_t> buf_ref(model_buffer.data(), model_buffer.size());
  auto model_res = CompiledModel::Create(*env, buf_ref, options);
  if (!model_res.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to load and compile model from buffer: ",
                     model_res.Error().Message()));
  }
  auto stage_or = Create(std::move(env), std::move(name), std::move(*model_res),
                         signature_index);
  if (!stage_or.ok()) return stage_or;
  (*stage_or)->model_buffer_.assign(model_buffer.begin(), model_buffer.end());
  (*stage_or)->options_ = std::move(options);
  return stage_or;
}

std::vector<std::string> CompiledModelStage::InputNames() const {
  return input_names_;
}

std::vector<std::string> CompiledModelStage::OutputNames() const {
  return output_names_;
}

absl::StatusOr<HardwareBufferDescriptor> CompiledModelStage::GetInputDescriptor(
    absl::string_view name) const {
  auto it = input_descriptors_.find(name);
  if (it == input_descriptors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no input named '", name, "'."));
  }
  return it->second;
}

absl::StatusOr<HardwareBufferDescriptor>
CompiledModelStage::GetOutputDescriptor(absl::string_view name) const {
  auto it = output_descriptors_.find(name);
  if (it == output_descriptors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no output named '", name, "'."));
  }
  return it->second;
}

absl::Status CompiledModelStage::SetInputBuffer(
    absl::string_view name, std::shared_ptr<LitertBuffer> buffer) {
  if (!input_descriptors_.contains(name)) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no input named '", name, "'."));
  }
  input_buffers_.insert_or_assign(name, std::move(buffer));
  return absl::OkStatus();
}

absl::Status CompiledModelStage::SetOutputBuffer(
    absl::string_view name, std::shared_ptr<LitertBuffer> buffer) {
  if (!output_descriptors_.contains(name)) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name_, "' has no output named '", name, "'."));
  }
  output_buffers_.insert_or_assign(name, std::move(buffer));
  return absl::OkStatus();
}

std::shared_ptr<LitertBuffer> CompiledModelStage::GetInputBuffer(
    absl::string_view name) const {
  auto it = input_buffers_.find(name);
  return (it != input_buffers_.end()) ? it->second : nullptr;
}

std::shared_ptr<LitertBuffer> CompiledModelStage::GetOutputBuffer(
    absl::string_view name) const {
  auto it = output_buffers_.find(name);
  return (it != output_buffers_.end()) ? it->second : nullptr;
}

absl::Status CompiledModelStage::Run() {
  std::vector<litert::TensorBuffer> in_bufs;
  in_bufs.reserve(input_names_.size());
  for (const auto& in_name : input_names_) {
    auto it = input_buffers_.find(in_name);
    if (it == input_buffers_.end() || it->second == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Stage '", name_, "': required input buffer '", in_name,
                       "' is not bound."));
    }
    // Note: LiteRT TensorBuffer inherits from BaseHandle (move-only wrapper
    // around unique_ptr). Duplicate() creates an additional reference handle.
    auto dup = it->second->tensor_buffer().Duplicate();
    if (!dup.HasValue()) {
      return absl::InternalError(
          absl::StrCat("Failed to duplicate input buffer '", in_name,
                       "': ", dup.Error().Message()));
    }
    in_bufs.push_back(std::move(*dup));
  }

  std::vector<litert::TensorBuffer> out_bufs;
  out_bufs.reserve(output_names_.size());
  for (const auto& out_name : output_names_) {
    std::shared_ptr<LitertBuffer> buf;
    auto it = output_buffers_.find(out_name);
    if (it == output_buffers_.end() || it->second == nullptr) {
      const auto& desc = output_descriptors_.at(out_name);
      if (!env_) {
        auto env_or = litert::Environment::Create({});
        if (!env_or.HasValue()) {
          return absl::InternalError("Failed to create LiteRT environment");
        }
        env_ = std::make_shared<litert::Environment>(std::move(*env_or));
      }
      auto buf_or = LitertBuffer::CreateManaged(env_, desc.buffer_type,
                                                desc.ToRankedTensorType(),
                                                desc.PackedBytes());
      if (!buf_or.ok()) {
        return buf_or.status();
      }
      buf = *buf_or;
      if (it != output_buffers_.end()) {
        it->second = buf;
      } else {
        output_buffers_.insert_or_assign(out_name, buf);
      }
    } else {
      buf = it->second;
    }
    auto dup = buf->tensor_buffer().Duplicate();
    if (!dup.HasValue()) {
      return absl::InternalError(
          absl::StrCat("Failed to duplicate output buffer '", out_name,
                       "': ", dup.Error().Message()));
    }
    out_bufs.push_back(std::move(*dup));
  }

  auto run_res = compiled_model_.Run(
      signature_index_,
      litert::Span<const litert::TensorBuffer>(in_bufs.data(), in_bufs.size()),
      litert::Span<const litert::TensorBuffer>(out_bufs.data(),
                                               out_bufs.size()));
  if (!run_res.HasValue()) {
    return absl::InternalError(absl::StrCat("Failed to execute model stage '",
                                            name_,
                                            "': ", run_res.Error().Message()));
  }
  return absl::OkStatus();
}

// ============================================================================
// BoundaryLayoutNegotiator Implementation
// ============================================================================

absl::StatusOr<HardwareBufferDescriptor>
BoundaryLayoutNegotiator::HarmonizeStageBoundary(
    const HardwareBufferDescriptor& producer_desc,
    const HardwareBufferDescriptor& consumer_desc) {
  HardwareBufferDescriptor harmonized;

  // 1. Data Type Validation
  if (producer_desc.element_type != litert::ElementType::None &&
      consumer_desc.element_type != litert::ElementType::None &&
      producer_desc.element_type != consumer_desc.element_type) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Type mismatch across stage boundary: producer outputs type ",
        static_cast<int>(producer_desc.element_type),
        ", but consumer expects type ",
        static_cast<int>(consumer_desc.element_type)));
  }
  harmonized.element_type =
      (producer_desc.element_type != litert::ElementType::None)
          ? producer_desc.element_type
          : consumer_desc.element_type;

  // 2. Shape Validation (allow differences only in singleton dimensions)
  auto NumElements = [](const std::vector<int32_t>& s) -> int64_t {
    if (s.empty()) return 0;
    int64_t count = 1;
    for (int32_t dim : s) {
      count *= dim;
    }
    return count;
  };

  auto SqueezeShape = [](const std::vector<int32_t>& s) {
    std::vector<int32_t> squeezed;
    squeezed.reserve(s.size());
    for (int32_t dim : s) {
      if (dim != 1) {
        squeezed.push_back(dim);
      }
    }
    return squeezed;
  };

  if (!producer_desc.shape.empty() && !consumer_desc.shape.empty()) {
    // Explicitly validate that the total volume (element count) matches.
    int64_t prod_elements = NumElements(producer_desc.shape);
    int64_t cons_elements = NumElements(consumer_desc.shape);
    if (prod_elements != cons_elements) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Element count mismatch across stage boundary: producer has ",
          prod_elements, " elements, but consumer expects ", cons_elements,
          " elements."));
    }

    auto prod_squeezed = SqueezeShape(producer_desc.shape);
    auto cons_squeezed = SqueezeShape(consumer_desc.shape);
    if (prod_squeezed != cons_squeezed) {
      auto FormatShape = [](const std::vector<int32_t>& s) {
        std::string out;
        for (size_t i = 0; i < s.size(); ++i) {
          if (i > 0) absl::StrAppend(&out, ", ");
          absl::StrAppend(&out, s[i]);
        }
        return out;
      };
      return absl::InvalidArgumentError(absl::StrCat(
          "Shape mismatch across stage boundary (excluding singleton "
          "dimensions): producer shape [",
          FormatShape(producer_desc.shape), "] vs consumer shape [",
          FormatShape(consumer_desc.shape), "]."));
    }

    // Validate that explicit size_bytes is not under-allocated for the shape.
    auto prod_ranked = producer_desc.ToRankedTensorType();
    if (auto min_bytes = prod_ranked.Bytes();
        min_bytes.HasValue() && producer_desc.size_bytes > 0 &&
        producer_desc.size_bytes < *min_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("Producer size_bytes (", producer_desc.size_bytes,
                       ") is smaller than required packed bytes for shape (",
                       *min_bytes, ")."));
    }
    auto cons_ranked = consumer_desc.ToRankedTensorType();
    if (auto min_bytes = cons_ranked.Bytes();
        min_bytes.HasValue() && consumer_desc.size_bytes > 0 &&
        consumer_desc.size_bytes < *min_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("Consumer size_bytes (", consumer_desc.size_bytes,
                       ") is smaller than required packed bytes for shape (",
                       *min_bytes, ")."));
    }
  }

  // Preserve producer shape if available, otherwise fallback to consumer.
  // When ranks differ due to singleton dimensions (e.g. {1, 128} vs {128}),
  // total byte volume and element ordering are identical, making zero-copy
  // reinterpretation safe.
  harmonized.shape =
      !producer_desc.shape.empty() ? producer_desc.shape : consumer_desc.shape;

  // 3. Size negotiation: allocate the maximum bytes requested to ensure the
  // buffer is large enough for both producer and consumer. If consumer expects
  // more bytes than producer outputs (e.g. due to backend alignment or row
  // padding requirements), the buffer accommodates the larger footprint without
  // overflow, and the producer populates its valid region.
  harmonized.size_bytes =
      std::max(producer_desc.PackedBytes(), consumer_desc.PackedBytes());

  // 4. Alignment negotiation (satisfy the strictest alignment)
  harmonized.alignment =
      std::max(producer_desc.alignment, consumer_desc.alignment);

  // 5. Buffer type negotiation
  if (producer_desc.buffer_type == litert::TensorBufferType::kAhwb ||
      consumer_desc.buffer_type == litert::TensorBufferType::kAhwb) {
    harmonized.buffer_type = litert::TensorBufferType::kAhwb;
  } else if (producer_desc.buffer_type ==
                 litert::TensorBufferType::kOpenClBuffer ||
             consumer_desc.buffer_type ==
                 litert::TensorBufferType::kOpenClBuffer) {
    harmonized.buffer_type = litert::TensorBufferType::kOpenClBuffer;
  } else if (litert::IsWebGpuMemory(producer_desc.buffer_type) ||
             litert::IsWebGpuMemory(consumer_desc.buffer_type)) {
    harmonized.buffer_type = litert::IsWebGpuMemory(producer_desc.buffer_type)
                                 ? producer_desc.buffer_type
                                 : consumer_desc.buffer_type;
  } else {
    harmonized.buffer_type = producer_desc.buffer_type;
  }

  // 6. Hardware capability flags (union of required accesses)
  harmonized.gpu_readable =
      producer_desc.gpu_readable || consumer_desc.gpu_readable;
  harmonized.gpu_writable =
      producer_desc.gpu_writable || consumer_desc.gpu_writable;
  harmonized.npu_accessible =
      producer_desc.npu_accessible || consumer_desc.npu_accessible;
  harmonized.cpu_accessible =
      producer_desc.cpu_accessible || consumer_desc.cpu_accessible;

  return harmonized;
}

// ============================================================================
// ModelChain and Builder Implementation
// ============================================================================

struct ModelChain::Builder::Impl {
  std::vector<std::shared_ptr<ModelStage>> stages;
  std::vector<Connection> connections;
  std::shared_ptr<litert::Environment> env;
};

struct ModelChain::Impl {
  std::shared_ptr<litert::Environment> env;
  std::vector<std::shared_ptr<ModelStage>> ordered_stages;
  absl::flat_hash_map<std::string, std::shared_ptr<ModelStage>> stages_by_name;
  std::vector<std::shared_ptr<LitertBuffer>> intermediate_buffers;
  absl::flat_hash_set<std::string> connected_inputs;
  absl::flat_hash_set<std::string> connected_outputs;
};

ModelChain::Builder::Builder() : impl_(std::make_unique<Impl>()) {}
ModelChain::Builder::~Builder() = default;

ModelChain::Builder& ModelChain::Builder::WithEnvironment(
    std::shared_ptr<litert::Environment> env) {
  impl_->env = std::move(env);
  return *this;
}

ModelChain::Builder& ModelChain::Builder::WithEnvironment(
    litert::Environment env) {
  impl_->env = std::make_shared<litert::Environment>(std::move(env));
  return *this;
}

ModelChain::Builder& ModelChain::Builder::AddStage(
    std::shared_ptr<ModelStage> stage) {
  impl_->stages.push_back(std::move(stage));
  return *this;
}

ModelChain::Builder& ModelChain::Builder::Connect(
    absl::string_view from_stage, absl::string_view output_name,
    absl::string_view to_stage, absl::string_view input_name) {
  impl_->connections.push_back({std::string(from_stage),
                                std::string(output_name), std::string(to_stage),
                                std::string(input_name)});
  return *this;
}

absl::StatusOr<ModelChain> ModelChain::Builder::Build() {
  if (impl_->stages.empty()) {
    return absl::InvalidArgumentError(
        "ModelChain must contain at least one stage.");
  }

  std::shared_ptr<litert::Environment> env;
  if (impl_->env != nullptr) {
    env = impl_->env;
  } else {
    auto env_or = litert::Environment::Create({});
    if (!env_or.HasValue()) {
      return absl::InternalError("Failed to create LiteRT environment");
    }
    env = std::make_shared<litert::Environment>(std::move(*env_or));
  }

  auto chain_impl = std::make_unique<ModelChain::Impl>();

  // 1. Index stages by name and verify uniqueness
  for (const auto& stage : impl_->stages) {
    std::string name(stage->Name());
    auto [it, inserted] = chain_impl->stages_by_name.try_emplace(name, stage);
    if (!inserted) {
      return absl::AlreadyExistsError(
          absl::StrCat("Duplicate stage name in ModelChain: '", name, "'."));
    }
  }

  // 2. Auto-infer linear connections if not explicitly provided
  if (impl_->connections.empty() && impl_->stages.size() > 1) {
    for (size_t i = 0; i + 1 < impl_->stages.size(); ++i) {
      auto out_names = impl_->stages[i]->OutputNames();
      auto in_names = impl_->stages[i + 1]->InputNames();
      if (out_names.size() == 1 && in_names.size() == 1) {
        impl_->connections.push_back(
            {std::string(impl_->stages[i]->Name()), out_names[0],
             std::string(impl_->stages[i + 1]->Name()), in_names[0]});
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Cannot auto-infer connections between stage '",
            impl_->stages[i]->Name(), "' (", out_names.size(),
            " outputs) and '", impl_->stages[i + 1]->Name(), "' (",
            in_names.size(), " inputs). Explicit Connect(...) is required."));
      }
    }
  }

  // 3. Build adjacency graph for topological sorting
  absl::flat_hash_map<std::string, std::vector<std::string>> adj;
  absl::flat_hash_map<std::string, int> in_degree;
  for (const auto& stage : impl_->stages) {
    in_degree[stage->Name()] = 0;
  }

  for (const auto& conn : impl_->connections) {
    if (!chain_impl->stages_by_name.contains(conn.from_stage)) {
      return absl::NotFoundError(
          absl::StrCat("Connection references unknown from_stage: '",
                       conn.from_stage, "'."));
    }
    if (!chain_impl->stages_by_name.contains(conn.to_stage)) {
      return absl::NotFoundError(absl::StrCat(
          "Connection references unknown to_stage: '", conn.to_stage, "'."));
    }
    adj[conn.from_stage].push_back(conn.to_stage);
    in_degree[conn.to_stage]++;
  }

  // 4. Topological Sort (Kahn's algorithm)
  std::queue<std::string> q;
  for (const auto& [name, deg] : in_degree) {
    if (deg == 0) {
      q.push(name);
    }
  }

  while (!q.empty()) {
    std::string curr = q.front();
    q.pop();
    chain_impl->ordered_stages.push_back(chain_impl->stages_by_name.at(curr));

    for (const auto& next : adj[curr]) {
      if (--in_degree[next] == 0) {
        q.push(next);
      }
    }
  }

  if (chain_impl->ordered_stages.size() != impl_->stages.size()) {
    return absl::InvalidArgumentError(
        "Cyclic dependency detected in ModelChain connections.");
  }

  // Propagate environment to all stages
  for (auto& stage : chain_impl->ordered_stages) {
    stage->SetEnvironment(env);
  }

  // 5. Harmonize boundaries and pre-allocate LitertBuffers
  // (supporting 1-to-N fan-out)
  absl::flat_hash_map<std::pair<std::string, std::string>,
                      std::vector<ConsumerTarget>>
      fanout_map;
  std::vector<std::pair<std::string, std::string>> unique_outputs;

  for (const auto& conn : impl_->connections) {
    auto to_stage = chain_impl->stages_by_name.at(conn.to_stage);
    std::pair<std::string, std::string> out_port = {conn.from_stage,
                                                    conn.output_name};
    auto [it, inserted] = fanout_map.try_emplace(out_port);
    if (inserted) {
      unique_outputs.push_back(out_port);
    }
    it->second.push_back({to_stage, conn.input_name});
  }

  using DescriptorMap =
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>;
  absl::flat_hash_map<std::string, DescriptorMap> stage_negotiated_inputs;
  absl::flat_hash_map<std::string, DescriptorMap> stage_negotiated_outputs;

  for (const auto& out_port : unique_outputs) {
    const auto& [from_stage_name, output_name] = out_port;
    auto from_stage = chain_impl->stages_by_name.at(from_stage_name);
    const auto& consumers = fanout_map.at(out_port);

    auto harmonized_or =
        NegotiatePortDescriptor(from_stage, output_name, consumers);
    if (!harmonized_or.ok()) return harmonized_or.status();
    const auto& harmonized = *harmonized_or;

    stage_negotiated_outputs[from_stage_name][output_name] = harmonized;
    for (const auto& consumer : consumers) {
      stage_negotiated_inputs[consumer.to_stage->Name()][consumer.input_name] =
          harmonized;
    }
  }

  // Stages like CompiledModelStage may re-compile and update their descriptors
  // based on initial boundary negotiation (e.g. enabling external GPU tensor
  // mode). We notify all stages, then re-negotiate in a second pass to pick up
  // any changes.
  for (auto& stage : chain_impl->ordered_stages) {
    std::string stage_name(stage->Name());
    auto prep_status =
        stage->PrepareStageBoundary(stage_negotiated_inputs[stage_name],
                                    stage_negotiated_outputs[stage_name]);
    if (!prep_status.ok()) return prep_status;
  }

  for (const auto& out_port : unique_outputs) {
    const auto& [from_stage_name, output_name] = out_port;
    auto from_stage = chain_impl->stages_by_name.at(from_stage_name);
    const auto& consumers = fanout_map.at(out_port);

    auto harmonized_or =
        NegotiatePortDescriptor(from_stage, output_name, consumers);
    if (!harmonized_or.ok()) return harmonized_or.status();
    const auto& harmonized = *harmonized_or;

    std::shared_ptr<LitertBuffer> shared_buf;
    if (harmonized.alignment > 0) {
      litert::TensorBufferType types[] = {harmonized.buffer_type};
      auto reqs_or = litert::TensorBufferRequirements::CreateWithAlignment(
          types, harmonized.PackedBytes(), harmonized.alignment);
      if (reqs_or.HasValue()) {
        auto buf_or = LitertBuffer::CreateManaged(
            env, harmonized.ToRankedTensorType(), *reqs_or);
        if (buf_or.ok()) {
          shared_buf = *buf_or;
        }
      }
    }
    if (!shared_buf) {
      auto shared_buf_or = LitertBuffer::CreateManaged(
          env, harmonized.buffer_type, harmonized.ToRankedTensorType(),
          harmonized.PackedBytes());
      if (!shared_buf_or.ok()) return shared_buf_or.status();
      shared_buf = *shared_buf_or;
    }

    auto s1 = from_stage->SetOutputBuffer(output_name, shared_buf);
    if (!s1.ok()) return s1;

    for (const auto& consumer : consumers) {
      auto s2 =
          consumer.to_stage->SetInputBuffer(consumer.input_name, shared_buf);
      if (!s2.ok()) return s2;

      chain_impl->connected_inputs.insert(
          absl::StrCat(consumer.to_stage->Name(), ":", consumer.input_name));
    }

    chain_impl->intermediate_buffers.push_back(shared_buf);
    chain_impl->connected_outputs.insert(
        absl::StrCat(from_stage_name, ":", output_name));
  }

  // 6. Pre-allocate terminal outputs
  for (const auto& stage : chain_impl->ordered_stages) {
    for (const auto& out_name : stage->OutputNames()) {
      std::string port_key = absl::StrCat(stage->Name(), ":", out_name);
      if (!chain_impl->connected_outputs.contains(port_key)) {
        auto desc_or = stage->GetOutputDescriptor(out_name);
        if (desc_or.ok()) {
          std::shared_ptr<LitertBuffer> out_buf;
          if (desc_or->alignment > 0) {
            litert::TensorBufferType types[] = {desc_or->buffer_type};
            auto reqs_or =
                litert::TensorBufferRequirements::CreateWithAlignment(
                    types, desc_or->PackedBytes(), desc_or->alignment);
            if (reqs_or.HasValue()) {
              auto buf_or = LitertBuffer::CreateManaged(
                  env, desc_or->ToRankedTensorType(), *reqs_or);
              if (buf_or.ok()) {
                out_buf = *buf_or;
              }
            }
          }
          if (!out_buf) {
            auto out_buf_or = LitertBuffer::CreateManaged(
                env, desc_or->buffer_type, desc_or->ToRankedTensorType(),
                desc_or->PackedBytes());
            if (out_buf_or.ok()) {
              out_buf = *out_buf_or;
            }
          }
          if (out_buf) {
            (void)stage->SetOutputBuffer(out_name, out_buf);
          }
        }
      }
    }
  }

  chain_impl->env = std::move(env);
  return ModelChain(std::move(chain_impl));
}

ModelChain::ModelChain(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ModelChain::~ModelChain() = default;
ModelChain::ModelChain(ModelChain&&) noexcept = default;
ModelChain& ModelChain::operator=(ModelChain&&) noexcept = default;

absl::Status ModelChain::Execute() {
  for (auto& stage : impl_->ordered_stages) {
    auto status = stage->Run();
    if (!status.ok()) {
      return absl::Status(status.code(), absl::StrCat("Stage '", stage->Name(),
                                                      "' execution failed: ",
                                                      status.message()));
    }
  }
  return absl::OkStatus();
}

absl::Status ModelChain::SetInputBuffer(absl::string_view stage_name,
                                        absl::string_view input_name,
                                        std::shared_ptr<LitertBuffer> buffer) {
  auto it = impl_->stages_by_name.find(stage_name);
  if (it == impl_->stages_by_name.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", stage_name, "' not found in ModelChain."));
  }
  return it->second->SetInputBuffer(input_name, std::move(buffer));
}

absl::Status ModelChain::SetInputBuffer(absl::string_view input_name,
                                        std::shared_ptr<LitertBuffer> buffer) {
  std::vector<std::string> matching_stages;
  for (const auto& stage : impl_->ordered_stages) {
    std::string port_key = absl::StrCat(stage->Name(), ":", input_name);
    if (!impl_->connected_inputs.contains(port_key)) {
      for (const auto& in_name : stage->InputNames()) {
        if (in_name == input_name) {
          matching_stages.push_back(std::string(stage->Name()));
          break;
        }
      }
    }
  }
  if (matching_stages.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No entry input named '", input_name, "' found in ModelChain."));
  }
  if (matching_stages.size() > 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input name '", input_name,
        "' is ambiguous across multiple entry stages (",
        absl::StrJoin(matching_stages, ", "), "). Specify stage_name."));
  }
  return impl_->stages_by_name.at(matching_stages.front())
      ->SetInputBuffer(input_name, std::move(buffer));
}

absl::StatusOr<std::shared_ptr<LitertBuffer>> ModelChain::GetOutputBuffer(
    absl::string_view stage_name, absl::string_view output_name) const {
  auto it = impl_->stages_by_name.find(stage_name);
  if (it == impl_->stages_by_name.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", stage_name, "' not found in ModelChain."));
  }
  auto buffer = it->second->GetOutputBuffer(output_name);
  if (!buffer) {
    return absl::NotFoundError(
        absl::StrCat("Output buffer '", output_name, "' on stage '", stage_name,
                     "' is not set or not yet allocated."));
  }
  return buffer;
}

absl::StatusOr<std::shared_ptr<LitertBuffer>> ModelChain::GetOutputBuffer(
    absl::string_view output_name) const {
  std::vector<std::string> matching_stages;
  for (const auto& stage : impl_->ordered_stages) {
    std::string port_key = absl::StrCat(stage->Name(), ":", output_name);
    if (!impl_->connected_outputs.contains(port_key)) {
      for (const auto& out_name : stage->OutputNames()) {
        if (out_name == output_name) {
          matching_stages.push_back(std::string(stage->Name()));
          break;
        }
      }
    }
  }
  if (matching_stages.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No terminal output named '", output_name, "' found in ModelChain."));
  }
  if (matching_stages.size() > 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Output name '", output_name,
        "' is ambiguous across multiple terminal stages (",
        absl::StrJoin(matching_stages, ", "), "). Specify stage_name."));
  }
  auto stage = impl_->stages_by_name.at(matching_stages.front());
  auto buffer = stage->GetOutputBuffer(output_name);
  if (!buffer) {
    return absl::NotFoundError(
        absl::StrCat("Output buffer '", output_name, "' on stage '",
                     stage->Name(), "' is not set or not yet allocated."));
  }
  return buffer;
}

absl::StatusOr<std::shared_ptr<ModelStage>> ModelChain::GetStage(
    absl::string_view name) const {
  auto it = impl_->stages_by_name.find(name);
  if (it == impl_->stages_by_name.end()) {
    return absl::NotFoundError(
        absl::StrCat("Stage '", name, "' not found in ModelChain."));
  }
  return it->second;
}

const std::vector<std::shared_ptr<LitertBuffer>>&
ModelChain::GetIntermediateBuffers() const {
  return impl_->intermediate_buffers;
}

}  // namespace litert::tensor
