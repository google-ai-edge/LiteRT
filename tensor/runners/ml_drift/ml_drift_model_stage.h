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
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_ML_DRIFT_MODEL_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_ML_DRIFT_MODEL_STAGE_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/runners/model_chain.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

namespace internal {

inline litert::ElementType TypeToLitertElementType(Type type) {
  switch (type) {
    case Type::kFP32:
      return litert::ElementType::Float32;
    case Type::kFP16:
      return litert::ElementType::Float16;
    case Type::kI32:
      return litert::ElementType::Int32;
    case Type::kI16:
      return litert::ElementType::Int16;
    case Type::kI8:
      return litert::ElementType::Int8;
    case Type::kU8:
      return litert::ElementType::UInt8;
    case Type::kBOOL:
      return litert::ElementType::Bool;
    default:
      return litert::ElementType::None;
  }
}

inline size_t ElementTypeSize(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
      return 4;
    case litert::ElementType::Float16:
      return 2;
    case litert::ElementType::Int32:
      return 4;
    case litert::ElementType::Int16:
      return 2;
    case litert::ElementType::Int8:
      return 1;
    case litert::ElementType::UInt8:
      return 1;
    case litert::ElementType::Bool:
      return 1;
    default:
      ABSL_DCHECK(false) << "Unsupported ElementType: "
                         << static_cast<int>(type);
      return 0;
  }
}

}  // namespace internal

// ModelStage adapter executing a CompiledMlDriftModelRunner on GPU/OpenCL.
// Automatically derives input and output tensor descriptors from flat
// TensorHandles and interfaces zero-copy with LiteRT buffers.
template <typename Backend = MlDriftClBackend>
class MlDriftModelStage : public ModelStage {
 public:
  using Runner = CompiledMlDriftModelRunner<Backend>;

  static absl::StatusOr<std::shared_ptr<MlDriftModelStage<Backend>>> Create(
      std::string name, std::vector<TensorHandle> inputs,
      std::vector<TensorHandle> outputs, std::unique_ptr<Runner> runner,
      litert::TensorBufferType buffer_type =
          litert::TensorBufferType::kHostMemory,
      std::shared_ptr<litert::Environment> litert_env = nullptr) {
    if (runner == nullptr) {
      return absl::InvalidArgumentError("Runner must not be null.");
    }

    std::vector<std::string> input_names;
    input_names.reserve(inputs.size());
    absl::flat_hash_map<std::string, HardwareBufferDescriptor> in_descs;
    absl::flat_hash_map<std::string, Type> in_types;
    absl::flat_hash_map<std::string, Shape> in_shapes;
    for (const auto& h : inputs) {
      std::string in_name = std::string(h.GetName());
      input_names.push_back(in_name);
      in_types[in_name] = h.GetType();
      in_shapes[in_name] = h.GetShape();

      HardwareBufferDescriptor desc;
      desc.element_type = internal::TypeToLitertElementType(h.GetType());
      desc.shape =
          std::vector<int32_t>(h.GetShape().begin(), h.GetShape().end());
      size_t num_elements = 1;
      for (int d : desc.shape) {
        if (d > 0) num_elements *= d;
      }
      desc.size_bytes =
          num_elements * internal::ElementTypeSize(desc.element_type);
      desc.buffer_type = buffer_type;
      desc.cpu_accessible = true;
      desc.gpu_readable = true;
      in_descs[in_name] = std::move(desc);
    }

    std::vector<std::string> output_names;
    output_names.reserve(outputs.size());
    absl::flat_hash_map<std::string, HardwareBufferDescriptor> out_descs;
    absl::flat_hash_map<std::string, Type> out_types;
    absl::flat_hash_map<std::string, Shape> out_shapes;
    for (const auto& h : outputs) {
      std::string out_name = std::string(h.GetName());
      output_names.push_back(out_name);
      out_types[out_name] = h.GetType();
      out_shapes[out_name] = h.GetShape();

      HardwareBufferDescriptor desc;
      desc.element_type = internal::TypeToLitertElementType(h.GetType());
      desc.shape =
          std::vector<int32_t>(h.GetShape().begin(), h.GetShape().end());
      size_t num_elements = 1;
      for (int d : desc.shape) {
        if (d > 0) num_elements *= d;
      }
      desc.size_bytes =
          num_elements * internal::ElementTypeSize(desc.element_type);
      desc.buffer_type = buffer_type;
      if (desc.element_type == litert::ElementType::Float32) {
        if (desc.buffer_type == litert::TensorBufferType::kOpenClTextureFp16) {
          desc.buffer_type = litert::TensorBufferType::kOpenClTexture;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kOpenClBufferFp16) {
          desc.buffer_type = litert::TensorBufferType::kOpenClBuffer;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kOpenClImageBufferFp16) {
          desc.buffer_type = litert::TensorBufferType::kOpenClImageBuffer;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kWebGpuTextureFp16) {
          desc.buffer_type = litert::TensorBufferType::kWebGpuTexture;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kWebGpuBufferFp16) {
          desc.buffer_type = litert::TensorBufferType::kWebGpuBuffer;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kWebGpuImageBufferFp16) {
          desc.buffer_type = litert::TensorBufferType::kWebGpuImageBuffer;
        }
      } else if (desc.element_type == litert::ElementType::Float16) {
        if (desc.buffer_type == litert::TensorBufferType::kOpenClTexture) {
          desc.buffer_type = litert::TensorBufferType::kOpenClTextureFp16;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kOpenClBuffer) {
          desc.buffer_type = litert::TensorBufferType::kOpenClBufferFp16;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kOpenClImageBuffer) {
          desc.buffer_type = litert::TensorBufferType::kOpenClImageBufferFp16;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kWebGpuTexture) {
          desc.buffer_type = litert::TensorBufferType::kWebGpuTextureFp16;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kWebGpuBuffer) {
          desc.buffer_type = litert::TensorBufferType::kWebGpuBufferFp16;
        } else if (desc.buffer_type ==
                   litert::TensorBufferType::kWebGpuImageBuffer) {
          desc.buffer_type = litert::TensorBufferType::kWebGpuImageBufferFp16;
        }
      }
      desc.cpu_accessible = true;
      desc.gpu_writable = true;
      out_descs[out_name] = std::move(desc);
    }

    if (!litert_env) {
      auto env_or = litert::Environment::Create({});
      if (env_or.HasValue()) {
        litert_env =
            std::make_shared<litert::Environment>(std::move(*env_or));
      }
    }

    return std::shared_ptr<MlDriftModelStage<Backend>>(new MlDriftModelStage(
        std::move(name), std::move(inputs), std::move(outputs),
        std::move(runner), std::move(input_names), std::move(output_names),
        std::move(in_descs), std::move(out_descs), std::move(in_types),
        std::move(in_shapes), std::move(out_types), std::move(out_shapes),
        std::move(litert_env)));
  }

  static absl::StatusOr<std::shared_ptr<MlDriftModelStage<Backend>>> Create(
      std::string name, std::vector<TensorHandle> inputs,
      std::vector<TensorHandle> outputs,
      ml_drift::CalculationsPrecision precision =
          ml_drift::CalculationsPrecision::kF32,
      typename Backend::Environment* env = nullptr,
      litert::TensorBufferType buffer_type =
          litert::TensorBufferType::kHostMemory,
      std::shared_ptr<litert::Environment> litert_env = nullptr) {
    auto runner = std::make_unique<Runner>(inputs, outputs, precision, env);
    LRT_TENSOR_RETURN_IF_ERROR(runner->BuildModel());
    return Create(std::move(name), std::move(inputs), std::move(outputs),
                  std::move(runner), buffer_type, std::move(litert_env));
  }

  absl::string_view Name() const override { return name_; }

  std::vector<std::string> InputNames() const override { return input_names_; }
  std::vector<std::string> OutputNames() const override {
    return output_names_;
  }

  absl::StatusOr<HardwareBufferDescriptor> GetInputDescriptor(
      absl::string_view name) const override {
    auto it = input_descriptors_.find(name);
    if (it == input_descriptors_.end()) {
      return absl::NotFoundError(absl::StrCat("Input not found: ", name));
    }
    return it->second;
  }

  absl::StatusOr<HardwareBufferDescriptor> GetOutputDescriptor(
      absl::string_view name) const override {
    auto it = output_descriptors_.find(name);
    if (it == output_descriptors_.end()) {
      return absl::NotFoundError(absl::StrCat("Output not found: ", name));
    }
    return it->second;
  }

  absl::Status SetInputBuffer(absl::string_view name,
                             std::shared_ptr<LitertBuffer> buffer) override {
    if (!input_descriptors_.contains(name)) {
      return absl::NotFoundError(absl::StrCat("Input not found: ", name));
    }
    input_buffers_.insert_or_assign(name, std::move(buffer));
    return absl::OkStatus();
  }

  absl::Status SetOutputBuffer(absl::string_view name,
                               std::shared_ptr<LitertBuffer> buffer) override {
    if (!output_descriptors_.contains(name)) {
      return absl::NotFoundError(absl::StrCat("Output not found: ", name));
    }
    output_buffers_.insert_or_assign(name, std::move(buffer));
    return absl::OkStatus();
  }

  std::shared_ptr<LitertBuffer> GetInputBuffer(
      absl::string_view name) const override {
    auto it = input_buffers_.find(name);
    return (it != input_buffers_.end()) ? it->second : nullptr;
  }

  std::shared_ptr<LitertBuffer> GetOutputBuffer(
      absl::string_view name) const override {
    auto it = output_buffers_.find(name);
    return (it != output_buffers_.end()) ? it->second : nullptr;
  }

  void SetEnvironment(std::shared_ptr<litert::Environment> env) override {
    litert_env_ = std::move(env);
  }

  Runner& runner() { return *runner_; }
  const Runner& runner() const { return *runner_; }

  absl::Status Run() override {
    for (const auto& in_name : input_names_) {
      auto it = input_buffers_.find(in_name);
      if (it == input_buffers_.end() || it->second == nullptr) {
        return absl::FailedPreconditionError(
            absl::StrCat("Stage '", name_, "': required input '", in_name,
                         "' is not bound."));
      }
      TensorInit init;
      init.name = in_name;
      init.type = input_types_.at(in_name);
      init.shape = input_shapes_.at(in_name);
      init.buffer = it->second;
      TensorHandle handle(init);
      LRT_TENSOR_RETURN_IF_ERROR(runner_->SetInput(in_name, handle));
    }

    LRT_TENSOR_RETURN_IF_ERROR(runner_->Run());

    if (!litert_env_) {
      auto env_or = litert::Environment::Create({});
      if (!env_or.HasValue()) {
        return absl::InternalError("Failed to create LiteRT environment");
      }
      litert_env_ =
          std::make_shared<litert::Environment>(std::move(*env_or));
    }

    for (const auto& out_name : output_names_) {
      auto it = output_buffers_.find(out_name);
      if (it != output_buffers_.end() && it->second != nullptr) {
        auto status = runner_->GetOutput(out_name, *it->second);
        if (status.ok()) {
          continue;
        }
        ABSL_LOG(WARNING) << "Stage '" << name_ << "': GetOutput failed for '"
                          << out_name << "': " << status
                          << ". Falling back to CPU copy.";
        LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_buf,
                                    runner_->GetOutputBuffer(out_name));
        auto src_lock = mld_buf->Lock();
        auto dst_lock = it->second->LockMutable();
        std::memcpy(dst_lock.data(), src_lock.data(),
                    std::min(dst_lock.size(), src_lock.size()));
      } else {
        LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_buf,
                                    runner_->GetOutputBuffer(out_name));
        const auto& desc = output_descriptors_.at(out_name);
        LRT_TENSOR_ASSIGN_OR_RETURN(
            auto host_buf,
            LitertBuffer::CreateManagedHost(litert_env_, desc.shape,
                                            desc.element_type,
                                            desc.size_bytes));
        auto src_lock = mld_buf->Lock();
        auto dst_lock = host_buf->LockMutable();
        std::memcpy(dst_lock.data(), src_lock.data(),
                    std::min(dst_lock.size(), src_lock.size()));
        output_buffers_[out_name] = std::move(host_buf);
      }
    }
    return absl::OkStatus();
  }

 private:
  MlDriftModelStage(
      std::string name, std::vector<TensorHandle> inputs,
      std::vector<TensorHandle> outputs, std::unique_ptr<Runner> runner,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>
          input_descriptors,
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>
          output_descriptors,
      absl::flat_hash_map<std::string, Type> input_types,
      absl::flat_hash_map<std::string, Shape> input_shapes,
      absl::flat_hash_map<std::string, Type> output_types,
      absl::flat_hash_map<std::string, Shape> output_shapes,
      std::shared_ptr<litert::Environment> litert_env)
      : name_(std::move(name)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        runner_(std::move(runner)),
        input_names_(std::move(input_names)),
        output_names_(std::move(output_names)),
        input_descriptors_(std::move(input_descriptors)),
        output_descriptors_(std::move(output_descriptors)),
        input_types_(std::move(input_types)),
        input_shapes_(std::move(input_shapes)),
        output_types_(std::move(output_types)),
        output_shapes_(std::move(output_shapes)),
        litert_env_(std::move(litert_env)) {}

  std::string name_;
  std::vector<TensorHandle> inputs_;
  std::vector<TensorHandle> outputs_;
  std::unique_ptr<Runner> runner_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor>
      input_descriptors_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor>
      output_descriptors_;
  absl::flat_hash_map<std::string, Type> input_types_;
  absl::flat_hash_map<std::string, Shape> input_shapes_;
  absl::flat_hash_map<std::string, Type> output_types_;
  absl::flat_hash_map<std::string, Shape> output_shapes_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>
      input_buffers_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>
      output_buffers_;
  std::shared_ptr<litert::Environment> litert_env_;
};

using MlDriftClModelStage = MlDriftModelStage<MlDriftClBackend>;

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_ML_DRIFT_MODEL_STAGE_H_
