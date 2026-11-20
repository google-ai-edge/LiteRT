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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_COMPILE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_COMPILE_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {

namespace internal {

// Base template
template <typename T>
struct LambdaTraits
    : public LambdaTraits<decltype(&std::decay_t<T>::operator())> {};

// Specialization for const lambda / functor
template <typename ClassType, typename ReturnType, typename... Args>
struct LambdaTraits<ReturnType (ClassType::*)(Args...) const> {
  static constexpr size_t arg_count = sizeof...(Args);
};

// Specialization for mutable lambda / functor
template <typename ClassType, typename ReturnType, typename... Args>
struct LambdaTraits<ReturnType (ClassType::*)(Args...)> {
  static constexpr size_t arg_count = sizeof...(Args);
};

// Specialization for free functions
template <typename ReturnType, typename... Args>
struct LambdaTraits<ReturnType (*)(Args...)> {
  static constexpr size_t arg_count = sizeof...(Args);
};

// Specialization for function types
template <typename ReturnType, typename... Args>
struct LambdaTraits<ReturnType(Args...)> {
  static constexpr size_t arg_count = sizeof...(Args);
};

template <typename Traits, typename Lambda, typename Tuple>
auto TraceLambdaImpl(Lambda&& func, Tuple&& args_tuple) {
  // Call the user's lambda with the provided Tensors
  return std::apply(func, std::forward<Tuple>(args_tuple));
}

// Helper to extract and sort keys from a map of tensors.
template <typename MapType>
std::vector<std::string> GetSortedTensorMapKeys(const MapType& tensors_map) {
  std::vector<std::string> keys;
  keys.reserve(tensors_map.size());
  for (const auto& kv : tensors_map) {
    keys.push_back(kv.first);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

template <typename T>
auto GetSortedTensors(T& t) {
  auto tensors_map = t.tensors();
  std::vector<std::string> keys = GetSortedTensorMapKeys(tensors_map);
  std::vector<typename decltype(tensors_map)::mapped_type> result;
  result.reserve(keys.size());
  for (const auto& key : keys) {
    result.push_back(tensors_map.at(key));
  }
  return result;
}

// Helper to flatten arbitrary return types (like std::tuple or single Tensors)
// into a std::vector<TensorHandle>.
template <typename T>
auto FlattenTensors(T& t, std::vector<TensorHandle>& flat_outputs)
    -> decltype(t.tensors(), void()) {
  for (const auto& tensor : GetSortedTensors(t)) {
    TensorHandle h(*tensor);
    flat_outputs.push_back(h);
  }
}

template <typename T>
auto FlattenTensors(const T& t, std::vector<TensorHandle>& flat_outputs)
    -> decltype(TensorHandle(t), void()) {
  flat_outputs.push_back(TensorHandle(t));
}

template <typename... Ts>
void FlattenTensors(const std::tuple<Ts...>& t,
                    std::vector<TensorHandle>& flat_outputs) {
  std::apply(
      [&flat_outputs](const auto&... args) {
        (FlattenTensors(args, flat_outputs), ...);
      },
      t);
}

template <typename Lambda, typename... Args>
struct AutoLambdaTraits {
  using result_type = decltype(std::declval<Lambda>()(std::declval<Args>()...));
  using arg_tuple =
      std::tuple<std::remove_cv_t<std::remove_reference_t<Args>>...>;
  static constexpr size_t arg_count = sizeof...(Args);
};

template <typename T>
auto SetStructOutputs(T& t, std::vector<TensorHandle>& flat_outputs)
    -> decltype(t.tensors(), void()) {
  for (const auto& tensor : GetSortedTensors(t)) {
    flat_outputs.push_back(TensorHandle(*tensor));
  }
}

template <typename T>
auto SetStructOutputs(T& t, std::vector<TensorHandle>& flat_outputs)
    -> decltype(TensorHandle(t), void()) {}

}  // namespace internal

template <typename BackendRunner, typename ReturnType, typename... Args>
class CompiledRunner {
 public:
  CompiledRunner(std::unique_ptr<BackendRunner> runner,
                 std::vector<TensorHandle> placeholders, ReturnType outputs)
      : runner_(std::move(runner)),
        placeholders_(std::move(placeholders)),
        traced_outputs_(std::move(outputs)) {}

  ReturnType operator()(Args... inputs) {
    auto inputs_tuple = std::forward_as_tuple(inputs...);

    // Assign data from dynamic input tensors to placeholders in the runner.
    int i = 0;
    std::apply(
        [this, &i](auto&&... args) {
          ([&](auto& arg) { this->AssignInputBuffers(arg, i); }(args),
           ...);
        },
        inputs_tuple);

    // Execute the compiled graph on the backend.
    ABSL_CHECK_OK(runner_->Run());

    // Extract the backend buffers and attach them to our pre-traced output
    // tensors.
    AssignOutputBuffers(traced_outputs_);

    return traced_outputs_;
  }

 private:
  template <typename T>
  auto AssignInputBuffers(T& t, int& i) -> decltype(t.tensors(), void()) {
    for (const auto& tensor : internal::GetSortedTensors(t)) {
      const TensorHandle& placeholder = this->placeholders_[i++];
      auto p_name = std::string(placeholder.GetName());
      ABSL_CHECK_OK(this->runner_->SetInput(p_name, *tensor));
    }
  }

  template <typename T>
  auto AssignInputBuffers(T& t, int& i)
      -> decltype(TensorHandle(t), void()) {
    const TensorHandle& placeholder = this->placeholders_[i++];
    auto name = std::string(placeholder.GetName());
    ABSL_CHECK_OK(this->runner_->SetInput(name, t));
  }

  template <typename T>
  auto AssignOutputBuffers(T& t) -> decltype(t.tensors(), void()) {
    for (const auto& [name, ptr] : t.tensors()) {
      auto buf_or = runner_->GetOutputBuffer(name);
      if (buf_or.ok()) {
        ptr->SetBuffer(buf_or.value());
      }
    }
  }

  template <typename T>
  auto AssignOutputBuffers(T& t) -> decltype(TensorHandle(t), void()) {
    auto name = std::string(t.GetName());
    auto buf_or = runner_->GetOutputBuffer(name);
    if (buf_or.ok()) {
      t.SetBuffer(buf_or.value());
    }
  }

  template <typename... Ts>
  void AssignOutputBuffers(std::tuple<Ts...>& t) {
    std::apply(
        [this](auto&... args) { (this->AssignOutputBuffers(args), ...); }, t);
  }

 private:
  std::unique_ptr<BackendRunner> runner_;
  std::vector<TensorHandle> placeholders_;
  ReturnType traced_outputs_;
};

// DUMMY BACKEND RUNNER to simulate the new abstraction during Phase 3.
class DummyBackendRunner {
 public:
  DummyBackendRunner(const std::vector<TensorHandle>& placeholders,
                     const std::vector<TensorHandle>& flat_outputs) {}
  absl::Status BuildModel() {
    return absl::OkStatus();
  }
  absl::Status Run() { return absl::OkStatus(); }
  absl::Status SetInput(const std::string& name, const TensorHandle& tensor) {
    return absl::OkStatus();
  }
  absl::StatusOr<std::shared_ptr<Buffer>> GetOutputBuffer(
      const std::string& name) {
    return nullptr;
  }
};

template <typename BackendRunner, typename Lambda, typename... Tensors>
auto compile(Lambda&& func, Tensors&&... tensors) {
  using Traits = internal::AutoLambdaTraits<Lambda, Tensors...>;
  static_assert(sizeof...(Tensors) == Traits::arg_count,
                "Number of Tensors must match the number of lambda arguments.");

  auto args = std::make_tuple(std::forward<Tensors>(tensors)...);
  std::vector<TensorHandle> placeholders;
  std::apply(
      [&placeholders](auto&... a) {
        (internal::FlattenTensors(a, placeholders), ...);
      },
      args);

  auto traced_outputs = std::apply(std::forward<Lambda>(func), std::move(args));

  std::vector<TensorHandle> flat_outputs;
  internal::FlattenTensors(traced_outputs, flat_outputs);

  auto backend_runner =
      std::make_unique<BackendRunner>(placeholders, flat_outputs);
  backend_runner->BuildModel().IgnoreError();

  return CompiledRunner<BackendRunner, decltype(traced_outputs),
                        std::remove_reference_t<Tensors>...>(
      std::move(backend_runner), std::move(placeholders),
      std::move(traced_outputs));
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_COMPILE_H_
