// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_FACTORY_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_FACTORY_H_

#include <memory>
#include <optional>

#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {

std::unique_ptr<QnnBackend> CreateBackend(const QnnApi* api,
                                          const Options& options,
                                          std::optional<SocInfo> soc_info,
                                          bool is_compiler);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_FACTORY_H_
