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
#include <memory>
#include <string>
#include <vector>

#include "tensor/arithmetic.h"
#include "tensor/tensor.h"
namespace litert::tensor::examples::gemma4::cpu {
struct KvOwnerSpec {
  int owner;
  int head_dim;
  float key_scale;
  float value_scale;
  int num_heads = 1;
};
inline std::shared_ptr<PerChannelAffineQuantization> KvQuantization(
    float scale) {
  return std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{scale}, std::vector<int64_t>{0});
}
inline std::shared_ptr<PerChannelAffineQuantization> CloneKvQuantization(
    const TensorHandle& cache) {
  return std::make_shared<PerChannelAffineQuantization>(
      *cache.GetQuantization()->As<PerChannelAffineQuantization>());
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
  const PerChannelAffineQuantization& quantization =
      *cache.GetQuantization()->template As<PerChannelAffineQuantization>();
  return Quantize(input, Type::kI8, quantization.scales,
                  quantization.zero_points);
}
}  // namespace litert::tensor::examples::gemma4::cpu
#endif
