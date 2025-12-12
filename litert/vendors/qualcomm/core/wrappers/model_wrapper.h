// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_MODEL_WRAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_MODEL_WRAPPER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

class ModelWrapper {
 public:
  ModelWrapper() = default;

  ModelWrapper(const ModelWrapper&) = delete;
  ModelWrapper& operator=(const ModelWrapper&) = delete;

  ModelWrapper(ModelWrapper&&) = default;
  ModelWrapper& operator=(ModelWrapper&&) = default;

  void AddOp(OpWrapper&& op, std::string_view prefix, std::string_view suffix);

  void AddOps(std::vector<OpWrapper>&& ops, std::string_view prefix,
              std::string_view suffix);

  absl::Span<const OpWrapper> GetOps() const { return ops_; }

  std::vector<OpWrapper>& GetOps() { return ops_; }

  size_t GetOpsSize() { return ops_.size(); }

  TensorPool& GetTensorPool() { return tensor_pool_; }

  const TensorPool& GetTensorPool() const { return tensor_pool_; }

 private:
  TensorPool tensor_pool_;
  std::vector<OpWrapper> ops_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_MODEL_WRAPPER_H_
