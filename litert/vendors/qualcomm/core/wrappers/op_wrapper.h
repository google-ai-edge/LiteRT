// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_OP_WRAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_OP_WRAPPER_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/param_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

class OpWrapper final {
 public:
  explicit OpWrapper() = default;

  explicit OpWrapper(std::string name, const char* op_type, QnnOpCode op_code)
      : type_name_{op_type}, name_{std::move(name)}, op_code_{op_code} {}

  bool operator==(const OpWrapper& other) const;

  void SetName(std::string name);

  void SetType(const char* op_type, QnnOpCode op_code);

  void AddInputTensor(const TensorWrapper& tensor);

  void AddOutputTensor(const TensorWrapper& tensor);

  template <typename T>
  void AddScalarParam(const char* name, const T data,
                      const bool is_quant = false) {
    scalar_params_.emplace_back(name, data, is_quant);
  }

  void AddTensorParam(const char* name, const TensorWrapper& tensor);

  Qnn_OpConfig_t GetOpConfig();

  QnnOpCode GetOpCode() const;

  std::string_view GetName() const;

  bool IsOpCode(QnnOpCode op_code) const;

  const TensorWrapper& GetInputTensor(size_t i) const;

  const TensorWrapper& GetOutputTensor(size_t i) const;

  const TensorParamWrapper& GetTensorPararm(size_t i) const;

  std::optional<ScalarParamWrapper> GetScalarParam(size_t i) const;

  std::vector<std::reference_wrapper<TensorWrapper>> GetAllTensors();

  void SwapOutputs(OpWrapper& other);

  void UpdateTensors(
      const std::vector<std::optional<TensorWrapperRef>>& inputs,
      const std::vector<std::optional<TensorWrapperRef>>& outputs);

  void AddPrefixToName(absl::string_view prefix);

  void AddSuffixToName(absl::string_view suffix);

 private:
  const char* type_name_{nullptr};
  std::string name_{};  // human readable name
  std::vector<std::reference_wrapper<const TensorWrapper>> input_tensors_{};
  std::vector<std::reference_wrapper<const TensorWrapper>> output_tensors_{};
  std::vector<ScalarParamWrapper> scalar_params_{};
  std::vector<TensorParamWrapper> tensor_params_{};
  std::vector<Qnn_Tensor_t> qnn_input_tensors_{};
  std::vector<Qnn_Tensor_t> qnn_output_tensors_{};
  std::vector<Qnn_Param_t> qnn_params_{};
  QnnOpCode op_code_{QnnOpCode::kUnknown};
};

bool IsElementWiseMultiply(const OpWrapper& op);

bool IsElementWiseAdd(const OpWrapper& op);

bool IsElementWiseNot(const OpWrapper& op);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_OP_WRAPPER_H_
