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

#include "tensor/examples/utils/tensor_mapping.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples {

LazyTensorMapping::LazyTensorMapping(
    const absl::flat_hash_map<std::string, std::string>& name_mapping,
    SafetensorLoader&& loader)
    : loader_(std::move(loader)) {
  for (const auto& [checkpoint_name, model_name] : name_mapping) {
    name_mapping_.emplace(model_name, checkpoint_name);
  }
}

absl::StatusOr<TensorHandle> LazyTensorMapping::Get(
    absl::string_view model_name) {
  if (auto it = tensors_.find(model_name); it != tensors_.end()) {
    return it->second;
  }
  absl::StatusOr<TensorHandle> tensor_or = Load(model_name);
  for (const std::unique_ptr<TensorMappingHooks>& hooks : hooks_) {
    if (!absl::IsNotFound(tensor_or.status())) {
      break;
    }
    tensor_or = hooks->OnNotFound(*this, model_name);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle tensor, std::move(tensor_or));
  for (const std::unique_ptr<TensorMappingHooks>& hooks : hooks_) {
    LRT_TENSOR_RETURN_IF_ERROR(hooks->OnLoaded(model_name, tensor));
  }
  // The hooks may have added tensors to the cache, the lookup above can't be
  // reused.
  tensors_.insert_or_assign(model_name, tensor);
  return tensor;
}

absl::Status LazyTensorMapping::Set(absl::string_view model_name,
                                    TensorHandle tensor) {
  tensors_.insert_or_assign(model_name, std::move(tensor));
  return absl::OkStatus();
}

absl::StatusOr<TensorHandle> LazyTensorMapping::Load(
    absl::string_view model_name) {
  auto checkpoint_name_it = name_mapping_.find(model_name);
  if (checkpoint_name_it == name_mapping_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find a tensor mapping to ", model_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle tensor,
                              loader_.LoadTensor(checkpoint_name_it->second));
  tensor.SetName(std::string(model_name));
  return tensor;
}

}  // namespace litert::tensor::examples
