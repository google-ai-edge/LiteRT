#include "litert/vendors/qualcomm/qnn_backend_test/utils.h"

#include "litert/cc/litert_model.h"
#include "litert/vendors/qualcomm/compiler/qnn_compose_graph.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
namespace litert::qnn {

bool ConvertLiteRtOp(litert::Op& op, ::qnn::TensorPool& tensor_pool,
                     std::vector<::qnn::TensorWrapperRef>& input_tensors,
                     std::vector<::qnn::TensorWrapperRef>& output_tensors,
                     std::vector<::qnn::OpWrapper>& op_wrappers,
                     bool use_htp_preference) {
  for (const auto& input : op.Inputs()) {
    ::qnn::TensorWrapper* tensor_wrapper{nullptr};
    auto status = ConvertTensor(input, tensor_pool, tensor_wrapper);
    if (status != kLiteRtStatusOk) {
      return false;
    }
    input_tensors.emplace_back(*tensor_wrapper);
  }
  for (const auto& output : op.Outputs()) {
    ::qnn::TensorWrapper* tensor_wrapper{nullptr};
    auto status = ConvertTensor(output, tensor_pool, tensor_wrapper);
    if (status != kLiteRtStatusOk) {
      return false;
    }
    output_tensors.emplace_back(*tensor_wrapper);
  }
  auto status = ConvertOp(use_htp_preference, op, tensor_pool, input_tensors,
                          output_tensors, op_wrappers);
  if (status != kLiteRtStatusOk) {
    return false;
  }
  return true;
}
}  // namespace litert::qnn
