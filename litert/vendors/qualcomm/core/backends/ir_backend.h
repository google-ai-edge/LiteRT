// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_IR_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_IR_BACKEND_H_

#include "IR/QnnIrCommon.h"
#include "IR/QnnIrGraph.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {

class IrBackend : public QnnBackend {
 public:
  static const char *GetLibraryName() { return "libQnnIr.so"; }

  IrBackend(const QNN_INTERFACE_VER_TYPE *qnn_api);

  ~IrBackend();

  bool Init(Qnn_LogHandle_t log_handle, const Options &options) override;

 private:
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_IR_BACKEND_H_
