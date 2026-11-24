// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_IR_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_IR_BACKEND_H_

#include <map>
#include <optional>

#include "IR/QnnIrCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

class IrBackend : public QnnBackend {
 public:
  static const char* GetLibraryName() {
#if defined(_WIN32)
    return "QnnIr.dll";
#else
    return "libQnnIr.so";
#endif
  }

  static Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_IR_API_VERSION_MAJOR;
    backend_version.minor = QNN_IR_API_VERSION_MINOR;
    backend_version.patch = QNN_IR_API_VERSION_PATCH;
    return backend_version;
  }

  explicit IrBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  bool Init(const Options& options,
            std::optional<::qnn::SocInfo> soc_info) override;

  GraphConfigBuilder BuildGraphConfigs(
      const Options& options, absl::string_view qnn_graph_name) override;
};

// Post-pass for IR "strip_quant" export (qairt-converter DLC_STRIP_QUANT):
// rewrites quantized tensors to Float_32 and dequantizes static data after the
// regular graph composition, leaving that flow untouched.
//
// Op builders may overwrite quant params (conv bias is reset to scale 0 for
// HTP), so Record() op inputs before building each op, then Apply() once.
class IrQuantStripper {
 public:
  // Remembers the quant params of the not-yet-built op inputs.
  void Record(absl::Span<const TensorWrapperRef> tensors);

  // Returns false if a tensor uses an unsupported data type or encoding.
  bool Apply(TensorPool& tensor_pool);

 private:
  std::map<const TensorWrapper*, QuantizeParamsWrapperVariant> original_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_IR_BACKEND_H_
