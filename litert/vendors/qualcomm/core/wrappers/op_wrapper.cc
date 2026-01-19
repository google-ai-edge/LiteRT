// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/param_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

bool OpWrapper::operator==(const OpWrapper& other) const {
  if (op_code_ != other.op_code_) return false;
  if (!miscs::IsStrEq(type_name_, other.type_name_)) return false;

  if (!std::equal(
          input_tensors_.begin(), input_tensors_.end(),
          other.input_tensors_.begin(), other.input_tensors_.end(),
          [](const auto& a, const auto& b) { return a.get() == b.get(); })) {
    return false;
  }

  if (!std::equal(
          output_tensors_.begin(), output_tensors_.end(),
          other.output_tensors_.begin(), other.output_tensors_.end(),
          [](const auto& a, const auto& b) { return a.get() == b.get(); })) {
    return false;
  }

  return scalar_params_ == other.scalar_params_ &&
         tensor_params_ == other.tensor_params_;
}

void OpWrapper::SetName(std::string name) { name_ = std::move(name); }

void OpWrapper::SetType(const char* op_type, QnnOpCode op_code) {
  type_name_ = op_type;
  op_code_ = op_code;
}

void OpWrapper::AddInputTensor(const TensorWrapper& tensor) {
  input_tensors_.emplace_back(tensor);
}

void OpWrapper::AddOutputTensor(const TensorWrapper& tensor) {
  output_tensors_.emplace_back(tensor);
}

void OpWrapper::AddTensorParam(const char* name, const TensorWrapper& tensor) {
  tensor_params_.emplace_back(name, tensor);
}

Qnn_OpConfig_t OpWrapper::GetOpConfig() {
  Qnn_OpConfig_t qnn_op = QNN_OPCONFIG_INIT;
  qnn_op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  qnn_op.v1.typeName = type_name_;
  qnn_op.v1.name = name_.data();
  // input tensors
  qnn_input_tensors_.reserve(input_tensors_.size());
  qnn_input_tensors_.clear();
  for (const auto& input_tensor : input_tensors_) {
    auto& back = qnn_input_tensors_.emplace_back();
    input_tensor.get().CloneTo(back);
  }
  qnn_op.v1.numOfInputs = qnn_input_tensors_.size();
  qnn_op.v1.inputTensors = qnn_input_tensors_.data();
  // output tensors
  qnn_output_tensors_.reserve(output_tensors_.size());
  qnn_output_tensors_.clear();
  for (const auto& output_tensor : output_tensors_) {
    auto& back = qnn_output_tensors_.emplace_back();
    output_tensor.get().CloneTo(back);
  }
  qnn_op.v1.numOfOutputs = qnn_output_tensors_.size();
  qnn_op.v1.outputTensors = qnn_output_tensors_.data();
  // params
  qnn_params_.reserve(scalar_params_.size() + tensor_params_.size());
  qnn_params_.clear();
  for (const auto& scalar_param : scalar_params_) {
    auto& back = qnn_params_.emplace_back();
    scalar_param.CloneTo(back);
  }
  for (const auto& tensor_param : tensor_params_) {
    auto& back = qnn_params_.emplace_back();
    tensor_param.CloneTo(back);
  }
  qnn_op.v1.numOfParams = qnn_params_.size();
  qnn_op.v1.params = qnn_params_.data();
  return qnn_op;
}

QnnOpCode OpWrapper::GetOpCode() const { return op_code_; }

std::string_view OpWrapper::GetName() const {
  return std::string_view(name_.data(), name_.size());
}

bool OpWrapper::IsOpCode(QnnOpCode op_code) const {
  return op_code_ == op_code;
}

const TensorWrapper& OpWrapper::GetInputTensor(size_t i) const {
  return input_tensors_[i].get();
}

const TensorWrapper& OpWrapper::GetOutputTensor(size_t i) const {
  return output_tensors_[i].get();
}

const TensorParamWrapper& OpWrapper::GetTensorPararm(size_t i) const {
  return tensor_params_[i];
}

std::optional<ScalarParamWrapper> OpWrapper::GetScalarParam(size_t i) const {
  if (i >= scalar_params_.size()) {
    return std::nullopt;
  }
  return scalar_params_[i];
}

void OpWrapper::SwapOutputs(OpWrapper& other) {
  this->output_tensors_.swap(other.output_tensors_);
}

void OpWrapper::UpdateTensors(
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  if (inputs.size() != input_tensors_.size() ||
      outputs.size() != output_tensors_.size()) {
    QNN_LOG_WARNING("UpdateTensors skipped due to incorrect tensor count.");
    return;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].has_value()) {
      input_tensors_[i] = inputs[i].value().get();
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].has_value()) {
      output_tensors_[i] = outputs[i].value().get();
    }
  }
}

std::vector<std::reference_wrapper<TensorWrapper>> OpWrapper::GetAllTensors() {
  std::vector<std::reference_wrapper<TensorWrapper>> ret;
  ret.reserve(input_tensors_.size());
  for (auto& tensor_ref : input_tensors_) {
    ret.emplace_back(const_cast<TensorWrapper&>(tensor_ref.get()));
  }
  for (auto& tensor_ref : output_tensors_) {
    ret.emplace_back(const_cast<TensorWrapper&>(tensor_ref.get()));
  }
  for (const auto& param : tensor_params_) {
    ret.emplace_back(const_cast<TensorWrapper&>(param.GetTensor()));
  }
  return ret;
};

void OpWrapper::AddPrefixToName(absl::string_view prefix) {
  name_ = absl::StrCat(prefix, name_);
}

void OpWrapper::AddSuffixToName(absl::string_view suffix) {
  name_ = absl::StrCat(name_, suffix);
}

namespace {
bool IsElementWiseOpImpl(const OpWrapper& op, QnnOpCode op_code,
                         const char* op_param_name, std::uint32_t op_param) {
  if (op.GetOpCode() != op_code) {
    return false;
  }

  auto scalar_param = op.GetScalarParam(0);
  if (!scalar_param.has_value()) {
    return false;
  }

  Qnn_Param_t param;
  scalar_param->CloneTo(param);
  if (std::strcmp(param.name, op_param_name) != 0 ||
      param.paramType != QNN_PARAMTYPE_SCALAR ||
      param.scalarParam.dataType != QNN_DATATYPE_UINT_32 ||
      param.scalarParam.uint32Value != op_param) {
    return false;
  }

  return true;
}
}  // namespace

bool IsElementWiseMultiply(const OpWrapper& op) {
  return IsElementWiseOpImpl(op, QnnOpCode::kElementWiseBinary,
                             QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
                             QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY);
}

bool IsElementWiseAdd(const OpWrapper& op) {
  return IsElementWiseOpImpl(op, QnnOpCode::kElementWiseBinary,
                             QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
                             QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD);
}

bool IsElementWiseNot(const OpWrapper& op) {
  return IsElementWiseOpImpl(op, QnnOpCode::kElementWiseUnary,
                             QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
                             QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT);
}

}  // namespace qnn
