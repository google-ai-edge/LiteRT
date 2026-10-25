/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LAMBDA_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LAMBDA_MODEL_RUNNER_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/backends/tflite/arithmetic_tflite.h"
#include "third_party/odml/litert/tensor/runners/litert/compiled_model_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert {
namespace tensor {

struct MapInputs {
  absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>> map;
  absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>*> tensors() const {
    absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>*> res;
    for (auto& [name, tensor] :
         const_cast<absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>&>(
             map)) {
      res[name] = &const_cast<Tensor<TfLiteMixinTag>&>(tensor);
    }
    return res;
  }
};

struct MapOutputs {
  absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>> map;
  absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>*> tensors() const {
    absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>*> res;
    for (auto& [name, tensor] :
         const_cast<absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>&>(
             map)) {
      res[name] = &const_cast<Tensor<TfLiteMixinTag>&>(tensor);
    }
    return res;
  }
};

using TensorsMap = absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>;

template <typename Lambda>
class LambdaModelRunner {
 public:

  explicit LambdaModelRunner(Environment& env, Options& options,
                             TensorsMap input_prototypes, Lambda f)
      : runner_(env, options, [input_prototypes, f](MapInputs& inputs) {
          inputs.map = input_prototypes;
          TensorsMap output_map = f(inputs.map);
          MapOutputs outputs;
          outputs.map = output_map;
          return outputs;
        }) {}

  explicit LambdaModelRunner(Environment& env, Options& options,
                             TensorsMap input_prototypes,
                             TensorsMap output_prototypes)
      : runner_(env, options,
                [input_prototypes, output_prototypes](MapInputs& inputs) {
                  inputs.map = input_prototypes;
                  MapOutputs outputs;
                  outputs.map = output_prototypes;
                  return outputs;
                }) {}

  absl::Status SetInput(const std::string& name, const TensorHandle& tensor) {
    return runner_.SetInput(name, tensor);
  }

  absl::Status SetInput(const std::string& name,
                        absl::Span<const uint8_t> data) {
    return runner_.SetInput(name, data);
  }

  absl::StatusOr<TensorHandle> GetInput(const std::string& name) {
    return runner_.GetInput(name);
  }

  absl::Status Run() { return runner_.Run(); }

  absl::StatusOr<TensorHandle> GetOutput(const std::string& name) {
    return runner_.GetOutput(name);
  }

  absl::Status SetOutput(const std::string& name, const TensorHandle& tensor) {
    return runner_.SetOutput(name, tensor);
  }

 private:
  CompiledModelRunner<std::function<MapOutputs(MapInputs&)>, MapInputs,
                      MapOutputs>
      runner_;
};

template <typename Lambda>
auto CreateLambdaRunner(Environment& env, Options& options,
                        TensorsMap input_prototypes, Lambda f) {
  return LambdaModelRunner<Lambda>(env, options, input_prototypes, f);
}

inline auto CreateStaticRunner(Environment& env, Options& options,
                               TensorsMap input_prototypes,
                               TensorsMap output_prototypes) {
  auto dummy_f = [](const TensorsMap&) { return TensorsMap(); };
  return LambdaModelRunner<decltype(dummy_f)>(env, options, input_prototypes,
                                              output_prototypes);
}

}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LAMBDA_MODEL_RUNNER_H_
