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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_HELPERS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_HELPERS_H_

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
struct MlDriftBuildContext {
  // The GpuModelBuilder to append the operations.Provided by the conversion API
  // caller.
  ml_drift::GpuModelBuilder* gpu_model_builder = nullptr;
  ml_drift::DataType default_float_type = ml_drift::DataType::kUnknown;

  // Tensor to MlDrift tensor handle map.
  absl::flat_hash_map<graph::Tensor, ml_drift::GpuModelBuilder::TensorHandle>
      graph_tensor_to_handle;

  // Handles of constant tensors uploaded in their own layout ([B]HWC) because
  // their handle in `graph_tensor_to_handle` has the layout of another
  // consumer, e.g. the packed weights of a fully connected op. See
  // `GetConstantTensorHandleInTensorLayout`.
  absl::flat_hash_map<graph::Tensor, ml_drift::GpuModelBuilder::TensorHandle>
      constant_handles_in_tensor_layout;

  // Shared cache for weights to avoid duplicate compilation and enable sharing.
  absl::flat_hash_map<std::string, ml_drift::GpuModelBuilder::TensorHandle>
      weight_name_to_handle;

  // Set of weight names that are compiled as external placeholder tensors
  // (not constant).
  absl::flat_hash_set<std::string> external_weight_names;

  // Optional cache of packed constant weight descriptors keyed by weight name,
  // populated during `Prepare()` and shared across signatures (e.g. prefill and
  // decode) to skip redundant CPU weight unpacking and rearrangement.
  absl::flat_hash_map<std::string, ml_drift::TensorDescriptor>*
      shared_weight_descriptors = nullptr;

  // True when the GpuModelBuilder may run large quantized FullyConnected /
  // convolutions with dynamically INT8-quantized activations (i.e. it was
  // created without `ModelHints::kDisallow8bitConvs`).
  bool allow_8bit_convs = false;

  // The hints `gpu_model_builder` was created with (for ops that select ML
  // Drift kernels themselves, e.g. to produce their weights in the kernel's
  // layout).
  ml_drift::ModelHints hints;

  // Per-output-channel sums of the signed quantized values of per-channel
  // quantized constant FullyConnected weights (`sum_i`), keyed by the weights
  // tensor. Only populated when `allow_8bit_convs`: ML Drift needs them to run
  // large FullyConnected layers with INT8-quantized activations.
  absl::flat_hash_map<graph::Tensor, ml_drift::GpuModelBuilder::TensorHandle>
      weight_sum_i;
};

// `LRT_ALLOW_8BIT_CONVS=1`: let ML Drift run large quantized FullyConnected /
// convolutions with dynamically INT8-quantized activations and INT8 dot
// products.
inline bool Allow8BitConvsFromEnv() {
  const char* env = std::getenv("LRT_ALLOW_8BIT_CONVS");
  return env != nullptr && std::string_view(env) == "1";
}

// The additive float zero points `-scale * zero_point` of the `num_channels`
// output channels of per-channel quantized FullyConnected weights, as ML
// Drift's INT8 convolution dequantization takes them. As in the subtractive
// zero point tensor of the other FullyConnected kernels, all-zero
// `float_zero_points` don't override the integer `zero_points`. Fails unless
// `num_channels >= 0` and the scales and the zero points used each have one
// value or one per channel (no integer zero points means zero).
absl::StatusOr<std::vector<float>> AdditiveZeroPoints(
    const graph::PerChannelAffineQuantization& pcq, int num_channels);

inline ml_drift::TensorDescriptor CopyDescriptorWithoutData(
    const ml_drift::TensorDescriptor& src) {
  ml_drift::TensorDescriptor dst;
  src.CopyWithoutData(&dst);
  dst.SetBHWDCShape(src.GetBHWDCShape());
  return dst;
}

absl::StatusOr<ml_drift::BHWC> ShapeToBHWC(const std::vector<int>& shape,
                                           int fill_value = 1);

// Helper functions for converting LiteRT tensor graphs to ML Drift models.
absl::StatusOr<ml_drift::DataType> GetMlDriftDataType(
    Type ty,
    ml_drift::DataType ml_drift_float_type = ml_drift::DataType::kFloat32);

absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> GetTensorHandle(
    MlDriftBuildContext& ctx, const graph::Tensor& t,
    bool create_if_not_found = true, bool create_linear_tensor = false,
    bool force_buffer_storage = false,
    ml_drift::TensorDescriptor::PhysicalLayout1D physical_layout_1d =
        ml_drift::TensorDescriptor::PhysicalLayout1D::kDCHWBC4);

// Returns a handle of the constant tensor `t` in its own layout ([B]HWC), for
// consumers like elementwise ops: the handle of `t` in
// `ctx.graph_tensor_to_handle` may have the layout of another consumer, e.g.
// the packed weights of a fully connected op. Uploads `t` again if needed.
absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
GetConstantTensorHandleInTensorLayout(MlDriftBuildContext& ctx,
                                      const graph::Tensor& t);

template <typename T>
std::string ShapeStr(const T& shape) {
  return absl::StrCat("[", absl::StrJoin(shape, ", "), "]");
}

template <typename T>
int Product(const T& array) {
  int numel = 1;
  for (int i = 0; i < array.size(); ++i) {
    numel *= array[i];
  }
  return numel;
}

template <typename T>
absl::Status CheckOutputSameNumElements(MlDriftBuildContext& ctx, const T& op) {
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  for (const graph::Tensor& out : outputs) {
    absl::StatusOr<const graph::TensorInformation&> info = graph::GetInfo(out);
    ABSL_CHECK(info.ok()) << "Failed to get info for tensor: " << out.index;
    int expected_numel = Product(info->shape);
    auto actual_bhwc =
        ctx.graph_tensor_to_handle.at(out).tensor_desc.GetBHWCShape();
    int actual_numel = actual_bhwc.DimensionsProduct();
    if (expected_numel != actual_numel) {
      std::string expected_shape_str = ShapeStr(info->shape);
      auto actual_bhwc =
          ctx.graph_tensor_to_handle.at(out).tensor_desc.GetBHWCShape();
      std::string actual_shape_str = ShapeStr(std::vector<int>{
          actual_bhwc.b, actual_bhwc.h, actual_bhwc.w, actual_bhwc.c});
      return absl::InternalError(absl::StrCat(
          "Operation ", op.GetName(),
          ": failed to build MlDrift GpuOperation: number of "
          "output elements mismatch. Expected: ",
          expected_numel, " (", expected_shape_str, "), actual: ", actual_numel,
          " (", actual_shape_str, ")"));
    }
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status CheckGpuOpOk(MlDriftBuildContext& ctx, const T& op) {
  return CheckOutputSameNumElements(ctx, op);
}
}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_HELPERS_H_
