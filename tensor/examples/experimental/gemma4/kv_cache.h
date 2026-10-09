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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_KV_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_KV_CACHE_H_
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
namespace litert::tensor::examples::experimental::gemma4::cpu {
// Specification for a layer that owns its own KV cache in Gemma4.
// Because Gemma4 uses KV cache sharing where later layers reuse the KV cache
// from earlier ("top") layers, each KvOwnerSpec describes the KV cache shape
// and quantization parameters for an owner layer that writes to and manages a
// distinct KV cache entry.
struct KvOwnerSpec {
  int owner;
  int head_dim;
  float key_scale;
  float value_scale;
  int num_heads = 1;
};

// Creates a symmetric per-tensor (single-scale, zero-point 0) affine
// quantization descriptor for an INT8 KV cache tensor.
inline std::shared_ptr<PerChannelAffineQuantization> KvQuantization(
    float scale) {
  return std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{scale}, std::vector<int64_t>{0});
}

// Clones the PerChannelAffineQuantization metadata from an existing KV cache
// tensor handle. Returns an InvalidArgumentError if `cache` lacks valid
// PerChannelAffineQuantization metadata.
inline absl::StatusOr<std::shared_ptr<PerChannelAffineQuantization>>
CloneKvQuantization(const TensorHandle& cache) {
  if (!cache.GetQuantization()) {
    return absl::InvalidArgumentError("KV cache tensor missing quantization");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const PerChannelAffineQuantization& quantization,
      cache.GetQuantization()->As<PerChannelAffineQuantization>());
  return std::make_shared<PerChannelAffineQuantization>(quantization);
}

// Constructs a 4D INT8 KV cache tensor with shape [1, 1, rows, dim] and the
// given symmetric quantization scale.
template <class... M>
Tensor<M...> MakeInt8KeyCache(std::string name, int rows, int dim,
                              float scale) {
  return Tensor<M...>({.name = std::move(name),
                       .type = Type::kI8,
                       .shape = {1, 1, rows, dim},
                       .quantization = KvQuantization(scale)});
}

// Quantizes a floating-point key or value tensor to INT8 using the quantization
// parameters attached to the target `cache` tensor. Returns an error tensor if
// `cache` lacks valid finite positive per-channel/per-tensor affine scales.
template <class... M>
Tensor<M...> QuantizeInt8Kv(Tensor<M...> input, const Tensor<M...>& cache) {
  if (!cache.GetQuantization()) {
    return absl::InvalidArgumentError("KV cache tensor missing quantization");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const PerChannelAffineQuantization& quantization,
      cache.GetQuantization()->template As<PerChannelAffineQuantization>());
  if (quantization.scales.empty()) {
    return absl::InvalidArgumentError("KV cache quantization scales empty");
  }
  for (float scale : quantization.scales) {
    if (!std::isfinite(scale) || scale <= 0.0f) {
      return absl::InvalidArgumentError(
          "KV cache quantization scale must be finite and positive");
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<int64_t> zero_points,
                              quantization.zero_points.template CopyAs<int64_t>(
                                  quantization.scales.size()));
  return Quantize(input, Type::kI8, quantization.scales,
                  std::move(zero_points));
}
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_KV_CACHE_H_
