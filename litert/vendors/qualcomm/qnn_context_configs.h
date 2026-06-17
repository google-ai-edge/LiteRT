// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_CONTEXT_CONFIGS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_CONTEXT_CONFIGS_H_

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "QnnContext.h"  // from @qairt

namespace litert::qnn {

// Empty (null-terminated) config list. QNN treats this as the default.
absl::Span<const QnnContext_Config_t*> DefaultContextConfigs();

// HTP: enables weight sharing across multiple contexts on the same backend.
absl::Span<const QnnContext_Config_t*> WeightSharingContextConfigs();

// GPU: sets the performance hint. Returns DefaultContextConfigs() for
// kDefault.
absl::Span<const QnnContext_Config_t*> GpuPerformanceContextConfigs(
    ::qnn::GpuPerformanceMode performance_mode);

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_CONTEXT_CONFIGS_H_
