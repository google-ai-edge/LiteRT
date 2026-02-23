// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_OP_WRAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_OP_WRAPPER_H_

#include <memory>
#include <string>
#include <vector>

#include "common-types.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"

namespace litert::samsung {
template <class T> struct ScalarTypeCast {
  static constexpr ScalarType value = ScalarType::UNKNOWN;
};

template <> struct ScalarTypeCast<uint64_t> {
  static constexpr ScalarType value = ScalarType::UINT64;
};

template <> struct ScalarTypeCast<int64_t> {
  static constexpr ScalarType value = ScalarType::INT64;
};

template <> struct ScalarTypeCast<uint32_t> {
  static constexpr ScalarType value = ScalarType::UINT32;
};

template <> struct ScalarTypeCast<int32_t> {
  static constexpr ScalarType value = ScalarType::INT32;
};

template <> struct ScalarTypeCast<float> {
  static constexpr ScalarType value = ScalarType::FLOAT32;
};

template <> struct ScalarTypeCast<double> {
  static constexpr ScalarType value = ScalarType::FLOAT64;
};

template <> struct ScalarTypeCast<bool> {
  static constexpr ScalarType value = ScalarType::BOOL;
};

template <> struct ScalarTypeCast<char> {
  static constexpr ScalarType value = ScalarType::CHAR;
};

class OpParamWrapper {
public:
  OpParamWrapper(const OpParamWrapper &) = delete;
  OpParamWrapper(OpParamWrapper &&) = default;

  template <typename E>
  static OpParamWrapper Create(const std::string &key, const E *data,
                               size_t size, bool scalar = true) {
    OpParamWrapper op_param(key);
    auto bytes = sizeof(E) * size;
    op_param.storage_ = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
    memcpy(op_param.storage_.get(), data, bytes);
    op_param.op_params_.data = op_param.storage_.get();
    op_param.op_params_.size = size;
    op_param.op_params_.is_scalar = scalar;
    op_param.op_params_.type = ScalarTypeCast<E>::value;

    return op_param;
  }

  const std::string &GetKey() const { return key_; }

  const ParamWrapper &GetValue() const { return op_params_; }

private:
  OpParamWrapper(const std::string &key) : key_(key) {}
  std::string key_;
  ParamWrapper op_params_;
  std::unique_ptr<uint8_t[]> storage_;
};

class OpWrapper {
public:
  OpWrapper(const std::string &name, const std::string &type)
      : op_name_(name), op_type_(type) {}

  OpWrapper &AddInput(const Tensor &t) {
    LiteRtTensor input = t.Get();
    inputs_.emplace_back(Tensor(input));

    return *this;
  }

  OpWrapper &AddOutput(const Tensor &t) {
    LiteRtTensor output = t.Get();
    outputs_.emplace_back(Tensor(output));

    return *this;
  }

  template <typename T,
            typename = std::enable_if_t<std::is_arithmetic<T>::value, void>>
  OpWrapper &AddParam(const std::string &key, T value) {
    op_params_.emplace_back(OpParamWrapper::Create(key, &value, 1));
    return *this;
  }

  template <typename T,
            typename = std::enable_if_t<std::is_arithmetic<T>::value, void>>
  OpWrapper &AddParam(const std::string &key, const std::vector<T> &value) {
    op_params_.emplace_back(
        OpParamWrapper::Create(key, value.data(), value.size(), true));
    return *this;
  }

  OpWrapper &AddParam(const std::string &key, const std::string &value) {
    op_params_.emplace_back(
        OpParamWrapper::Create(key, value.data(), value.size(), true));
    return *this;
  }

  uint32_t GetNumOfParams() const { return op_params_.size(); }

  const OpParamWrapper &GetParamWithIndex(int index) const {
    return op_params_.at(index);
  }

  const std::vector<Tensor> &GetInputs() const { return inputs_; }

  const std::vector<Tensor> &GetOutputs() const { return outputs_; }

  const char *GetCName() const { return op_name_.c_str(); }

  const char *GetCType() const { return op_type_.c_str(); }

private:
  std::string op_name_;
  std::string op_type_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;

  std::vector<OpParamWrapper> op_params_;
};

} // namespace litert::samsung
#endif
