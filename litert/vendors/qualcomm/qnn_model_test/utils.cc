#include "litert/vendors/qualcomm/qnn_model_test/utils.h"

#include "QnnGraph.h"  // from @qairt
#include "litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace {
typedef std::vector<qnn::OpWrapper> QnnModel;

#define QNN_RETURN_STATUS_IF_NOT_OK(expr) \
  if (QNN_SUCCESS != (expr)) {            \
    return false;                         \
  }
}  // namespace

namespace qnn {
bool ValidateModel(litert::qnn::QnnManager& qnn, QnnModel& ops) {
  return std::all_of(
      ops.begin(), ops.end(), [&qnn](OpWrapper& op_wrapper) -> bool {
        return kLiteRtStatusOk == qnn.ValidateOp(op_wrapper.GetOpConfig());
      });
}

bool CreateGraphAndCompile(litert::qnn::QnnManager& qnn,
                           QnnModel& ops) {
  // Create ops and their corresponding tensors.
  auto context_configs = litert::qnn::QnnManager::DefaultContextConfigs();
  auto context_handle = qnn.CreateContextHandle(context_configs);
  if (!context_handle) {
    return false;
  }
  Qnn_GraphHandle_t graph_handle = nullptr;
  QNN_RETURN_STATUS_IF_NOT_OK(qnn.Api()->graphCreate(
      (*context_handle).get(), "test",
      litert::qnn::GetDefaultGraphConfigs().data(), &graph_handle));
  for (auto& op_wrapper : ops) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      QNN_RETURN_STATUS_IF_NOT_OK(qnn.Api()->tensorCreateGraphTensor(
          graph_handle, &(tensor_wrapper_ref.get().GetQnnTensor())));
    }
    qnn.Api()->graphAddNode(graph_handle, op_wrapper.GetOpConfig());
  }

  QNN_RETURN_STATUS_IF_NOT_OK(
      qnn.Api()->graphFinalize(graph_handle, nullptr, nullptr));
  return true;
}

}  // namespace qnn