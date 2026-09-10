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

#include "third_party/odml/litert/tensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/utils/source_location.h"

namespace litert::tensor {

TensorHandle::TensorHandle(source_location loc)
    : impl_(graph::NewTensor(std::move(loc))) {}

TensorHandle::TensorHandle(graph::Tensor impl) : impl_(impl) {}

TensorHandle::TensorHandle(TensorInit init, source_location loc)
    : TensorHandle(std::move(loc)) {
  Set(std::move(init));
}

TensorHandle TensorHandle::ShallowClone() const {
  TensorHandle handle;
  ShallowCloneTo(handle);
  return handle;
}

void TensorHandle::ShallowCloneTo(TensorHandle& other) const {
  *GetInfo(other.impl_) = *GetInfo(impl_);
}

TensorHandle& TensorHandle::Set(TensorInit init, source_location loc) & {
  if (!graph::GetStatus(impl_).ok()) {
    impl_ = graph::NewTensor(loc);
  }
  graph::TensorInformation& info = *GetInfo(impl_);
  info.name = std::move(init.name);
  info.type = init.type;
  info.shape = std::move(init.shape);
  std::visit(
      [&info](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
          info.buffer = std::forward<decltype(arg)>(arg);
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
          info.buffer = OwningCpuBuffer::Copy<Type::kFP32>(arg);
        } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          info.buffer = OwningCpuBuffer::Copy<Type::kI32>(arg);
        } else if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          info.buffer = OwningCpuBuffer::Copy<Type::kI8>(arg);
        } else {
          ABSL_LOG(ERROR) << "Failed to create buffer from typed vector: "
                             "unsupported datatype.";
        }
      },
      std::move(init.buffer));
  info.quantization = init.quantization;
  return *this;
}

TensorHandle& TensorHandle::SetName(std::string name) & {
  if (absl::Status s = graph::SetName(GetRaw(), std::move(name)); !s.ok()) {
    ABSL_LOG(ERROR) << "Error when setting tensor name: " << s;
  }
  return *this;
}

TensorHandle&& TensorHandle::SetName(std::string name) && {
  return std::move(this->SetName(std::move(name)));
}

absl::string_view TensorHandle::GetName() const {
  absl::StatusOr<const std::string&> name = graph::GetName(GetRaw());
  if (name.ok()) {
    return *name;
  }
  return "";
}

TensorHandle& TensorHandle::SetType(Type type) & {
  absl::StatusOr<graph::TensorInformation&> info = graph::GetInfo(impl_);
  if (!info.ok()) {
    ABSL_LOG(ERROR) << "Error when getting tensor info: " << info.status();
    return *this;
  }
  info->type = type;
  return *this;
}

TensorHandle&& TensorHandle::SetType(Type type) && {
  return std::move(this->SetType(type));
}

Type TensorHandle::GetType() const {
  absl::StatusOr<const graph::TensorInformation&> info = graph::GetInfo(impl_);
  if (!info.ok()) {
    ABSL_LOG(ERROR) << "Error when getting tensor info: " << info.status();
    return Type::kUnknown;
  }
  return info->type;
}

TensorHandle& TensorHandle::SetQuantization(
    std::shared_ptr<Quantization> quantization) & {
  absl::StatusOr<graph::TensorInformation&> info = graph::GetInfo(impl_);
  if (!info.ok()) {
    ABSL_LOG(ERROR) << "Error when getting tensor info: " << info.status();
    return *this;
  }
  info->quantization = std::move(quantization);
  return *this;
}

TensorHandle&& TensorHandle::SetQuantization(
    std::shared_ptr<Quantization> quantization) && {
  return std::move(this->SetQuantization(std::move(quantization)));
}

std::shared_ptr<const Quantization> TensorHandle::GetQuantization() const {
  absl::StatusOr<const graph::TensorInformation&> info = graph::GetInfo(impl_);
  if (!info.ok()) {
    ABSL_LOG(ERROR) << "Error when getting tensor info: " << info.status();
    return nullptr;
  }
  return info->quantization;
}

std::shared_ptr<Quantization> TensorHandle::GetQuantization() {
  absl::StatusOr<graph::TensorInformation&> info = graph::GetInfo(impl_);
  if (!info.ok()) {
    ABSL_LOG(ERROR) << "Error when getting tensor info: " << info.status();
    return nullptr;
  }
  return info->quantization;
}

TensorHandle& TensorHandle::SetBuffer(std::shared_ptr<Buffer> buffer) & {
  if (absl::Status s = graph::SetBuffer(GetRaw(), std::move(buffer)); !s.ok()) {
    ABSL_LOG(ERROR) << "Error when setting tensor buffer: " << s;
  }
  return *this;
}

TensorHandle&& TensorHandle::SetBuffer(std::shared_ptr<Buffer> buffer) && {
  return std::move(this->SetBuffer(std::move(buffer)));
}

absl::StatusOr<Buffer&> TensorHandle::GetBuffer() const {
  // graph::GetBuffer expects a non-const graph::Tensor&, but GetRaw() returns
  // a const graph::Tensor& in this const method. Using const_cast to
  // temporarily remove the const qualifier. The underlying graph::GetBuffer
  // should ideally be updated to accept const graph::Tensor&.
  return graph::GetBuffer(const_cast<graph::Tensor&>(GetRaw()));
}

TensorHandle& TensorHandle::SetShape(Shape shape) & {
  absl::StatusOr<graph::TensorInformation&> info = graph::GetInfo(GetRaw());
  if (!info.ok()) {
    ABSL_LOG(ERROR) << "Error when setting tensor shape: " << info.status();
  }
  info->shape = std::move(shape);
  return *this;
}

TensorHandle&& TensorHandle::SetShape(Shape shape) && {
  return std::move(this->SetShape(std::move(shape)));
}

// Gets the tensor type.
const Shape& TensorHandle::GetShape() const {
  absl::StatusOr<const graph::TensorInformation&> info =
      graph::GetInfo(GetRaw());
  ABSL_CHECK(info.ok()) << "Error when getting tensor shape: " << info.status();
  return info->shape;
}

absl::Status TensorHandle::GetStatus() const {
  return graph::GetStatus(GetRaw());
}

}  // namespace litert::tensor
