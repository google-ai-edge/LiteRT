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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_TENSOR_MAPPING_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_TENSOR_MAPPING_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples {

// Interface for tensor mappings used for graph construction.
//
// Note: we call `model_name` the name of the tensor that is queried by the
// graph building functions, as opposed to the `checkpoint_name` which is the
// name associated to a tensor.
class TensorMapping {
 public:
  virtual ~TensorMapping() = default;

  // Gets the tensor that maps to the given `model_name`.
  virtual absl::StatusOr<TensorHandle> Get(absl::string_view model_name) = 0;

  // Maps a new tensor to the given `model_name`
  virtual absl::Status Set(absl::string_view model_name,
                           TensorHandle tensor) = 0;
};

// Extension points called by a mapping when it loads tensors.
//
// They allow adapting the tensors of a mapping (e.g. converting their type or
// building tensors from other ones) without changing the mapping itself.
//
// A mapping may chain multiple hooks, which are called in registration order.
// The default implementation doesn't change the mapping behaviour.
class TensorMappingHooks {
 public:
  virtual ~TensorMappingHooks() = default;

  // Called once per tensor, before the mapping caches it.
  //
  // `tensor` may be modified in place (e.g. to convert its type). Every chained
  // hook is called. Tensors provided through `TensorMapping::Set` don't go
  // through this hook.
  virtual absl::Status OnLoaded(absl::string_view model_name,
                                TensorHandle& tensor) {
    return absl::OkStatus();
  }

  // Called when the mapping source has no tensor for `model_name`.
  //
  // May build the tensor, e.g. from other tensors obtained through
  // `mapping.Get()`, as long as it doesn't request `model_name` again. The
  // result goes through `OnLoaded()` and is cached by the mapping.
  //
  // Returning a `NotFound` error lets the next chained hook handle the tensor.
  // Any other result stops the chain.
  virtual absl::StatusOr<TensorHandle> OnNotFound(
      TensorMapping& mapping, absl::string_view model_name) {
    return absl::NotFoundError(
        absl::StrCat("Could not find a tensor mapping to ", model_name));
  }
};

// Mapping that loads tensors on demand, caches them and adapts them with hooks.
//
// When a tensor isn't cached, it is loaded from the checkpoint. If the
// checkpoint doesn't hold it, the hooks `TensorMappingHooks::OnNotFound()` are
// called until one of them provides it. The resulting tensor then goes through
// every hook `TensorMappingHooks::OnLoaded()` and is cached.
class LazyTensorMapping : public TensorMapping {
 public:
  // Creates an empty mapping.
  LazyTensorMapping() = default;

  // Creates a mapping over tensors that are already loaded.
  //
  // - `tensors`: maps the graph building names to the tensors. These tensors
  //   don't go through the hooks.
  explicit LazyTensorMapping(
      absl::flat_hash_map<std::string, TensorHandle> tensors)
      : tensors_(std::move(tensors)) {}

  // Prepares to load tensors that are listed in the `name_mapping`.
  //
  // - `name_mapping`: maps the checkpoint names to the graph building names.
  // - `loader`: the actual tensor loader.
  LazyTensorMapping(
      const absl::flat_hash_map<std::string, std::string>& name_mapping,
      SafetensorLoader&& loader);

  // Registers a `Hooks` built from `args` after the hooks that are already
  // registered.
  //
  // The hooks only apply to the tensors that are loaded after they are
  // registered.
  template <class Hooks, class... Args>
  LazyTensorMapping& Register(Args&&... args) & {
    static_assert(std::is_base_of_v<TensorMappingHooks, Hooks>,
                  "Hooks must derive from TensorMappingHooks.");
    hooks_.push_back(std::make_unique<Hooks>(std::forward<Args>(args)...));
    return *this;
  }

  // Registers a `Hooks` built from `args` after the hooks that are already
  // registered.
  //
  // The hooks only apply to the tensors that are loaded after they are
  // registered.
  template <class Hooks, class... Args>
  LazyTensorMapping&& Register(Args&&... args) && {
    Register<Hooks>(std::forward<Args>(args)...);
    return std::move(*this);
  }

  // Gets the tensor that maps to the given `model_name`.
  absl::StatusOr<TensorHandle> Get(absl::string_view model_name) override;

  // Maps a new tensor to the given `model_name`. The tensor doesn't go through
  // the hooks.
  absl::Status Set(absl::string_view model_name, TensorHandle tensor) override;

 private:
  // Loads the tensor that maps to `model_name` from the checkpoint.
  //
  // Returns a `NotFound` error if the checkpoint doesn't hold that tensor.
  absl::StatusOr<TensorHandle> Load(absl::string_view model_name);

  // Maps the graph building names to the checkpoint names.
  absl::flat_hash_map<std::string, std::string> name_mapping_;
  SafetensorLoader loader_;
  std::vector<std::unique_ptr<TensorMappingHooks>> hooks_;
  absl::flat_hash_map<std::string, TensorHandle> tensors_;
};

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_TENSOR_MAPPING_H_
