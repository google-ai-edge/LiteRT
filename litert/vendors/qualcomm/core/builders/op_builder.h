// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
enum class MirrorPadMode {
  REFLECT = 0,
  SYMMETRIC,
};

enum class PaddingType {
  Unknown = 0,
  Same,
  Valid,
};

typedef enum {
  FusedActivationNone = 0,
  FusedActivationRelu = 1,
  FusedActivationReluN1To1 = 2,
  FusedActivationRelu6 = 3,
  FusedActivationTanh = 4,
  FusedActivationSignBit = 5,
} FusedActivationType;

std::pair<std::uint32_t, std::uint32_t> ComputePaddingBeforeAfter(
    const std::uint32_t input_size, const std::uint32_t filter_size,
    const std::uint32_t stride, const std::uint32_t dilation_rate,
    const PaddingType padding_type);

// Returns true when a depthwise convolution has stride >= 4 and filter >= 4
// in both spatial dimensions, where QNN HTP prefers grouped Conv2d over
// DepthWiseConv2d (matching QAIRT's CheckUseGroupedConv2d heuristic).
bool ShouldUseGroupedConv2d(const TensorWrapper& filter_tensor,
                            std::uint32_t stride_h, std::uint32_t stride_w);

std::string GetUniqueOpName(const char* op_type);

OpWrapper& CreateOpWrapper(std::vector<OpWrapper>& ops, const char* op_type);

OpWrapper CreateOpWithSameParams(
    const OpWrapper& src, absl::Span<const ConstTensorWrapperRef> inputs,
    absl::Span<const ConstTensorWrapperRef> outputs);

/*
  This function creates a new tensor and replaces the original output tensor of
  the operation if the fused activation is not None.

  The replaced output tensor will be returned and can be used in fused
  activation node.
*/
TensorWrapper& CreateFusedActivationInputTensor(
    TensorPool& tensor_pool, const uint32_t fused_activation_function,
    std::vector<TensorWrapperRef>& output_tensors);

void AddFusedActivationNode(std::vector<OpWrapper>& res,
                            const uint32_t fused_activation_function,
                            const TensorWrapper& input_tensor,
                            const TensorWrapper& output_tensor);

template <typename... Args>
auto MakeVector(Args&&... args) {
  using T = std::decay_t<std::common_type_t<Args...>>;
  std::vector<T> v;
  v.reserve(sizeof...(args));
  (v.emplace_back(std::forward<Args>(args)), ...);
  return v;
}

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_
