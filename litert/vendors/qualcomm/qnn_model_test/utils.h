// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
namespace qnn {
bool ValidateModel(litert::qnn::QnnManager& qnn, std::vector<OpWrapper>& ops);
bool CreateGraphAndCompile(litert::qnn::QnnManager& qnn, std::vector<OpWrapper>& ops);
}
