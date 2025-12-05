// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/model_wrapper.h"

#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace qnn {

void ModelWrapper::AddOp(OpWrapper&& op, std::string_view prefix,
                         std::string_view suffix) {
  op.AddPrefixToName(prefix);
  op.AddSuffixToName(suffix);

  ops_.push_back(std::move(op));
}

void ModelWrapper::AddOps(std::vector<OpWrapper>&& ops, std::string_view prefix,
                          std::string_view suffix) {
  for (auto& op_wrapper : ops) {
    op_wrapper.AddPrefixToName(prefix);
    op_wrapper.AddSuffixToName(suffix);
  }

  std::move(ops.begin(), ops.end(), std::back_inserter(ops_));
}

}  // namespace qnn
