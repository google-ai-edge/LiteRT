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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_KV_CACHE_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_KV_CACHE_H_
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
namespace litert::tensor::examples::gemma4::cpu {
struct KvOwnerSpec {
  int owner;
  int head_dim;
  float key_scale;
  float value_scale;
};
inline std::shared_ptr<PerChannelAffineQuantization> KvQuantization(
    float scale) {
  return std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{scale}, std::vector<int64_t>{0});
}
inline std::shared_ptr<PerChannelAffineQuantization> CloneKvQuantization(
    const TensorHandle& cache) {
  if (!cache.GetQuantization()) return nullptr;
  auto q = cache.GetQuantization()->As<PerChannelAffineQuantization>();
  if (!q.ok()) return nullptr;
  return std::make_shared<PerChannelAffineQuantization>(*q);
}
template <class... M>
Tensor<M...> MakeInt8KeyCache(std::string name, int rows, int dim,
                              float scale) {
  return Tensor<M...>({.name = std::move(name),
                       .type = Type::kI8,
                       .shape = {1, 1, rows, dim},
                       .quantization = KvQuantization(scale)});
}
template <class... M>
Tensor<M...> QuantizeInt8Kv(Tensor<M...> input, const Tensor<M...>& cache) {
  auto result = Cast(input, Type::kI8);
  result.SetQuantization(CloneKvQuantization(cache));
  return result;
}
}  // namespace litert::tensor::examples::gemma4::cpu
#endif
