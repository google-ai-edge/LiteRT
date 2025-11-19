// Copyright (C) 2025 Intel Corporation
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_DECODER_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_DECODER_H_

#include <openvino/frontend/tensorflow_lite/decoder.hpp>
#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_macros.h"

namespace litert {
namespace openvino {

class DecoderOperation
    : public ov::frontend::tensorflow_lite::DecoderBaseOperation {
 public:
  // TODO: in/out _tensor_info copy has to be avoided
  explicit DecoderOperation(
      std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
          input_tensor_info,
      std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
          output_tensor_info,
      const litert::Op& litert_op, size_t node_index);
  virtual ~DecoderOperation() = default;

  // DecoderBase Interface implementations :
  /// \brief Get attribute value by name
  ov::Any get_attribute(const std::string& name) const override;
  /// \brief Get a number of inputs
  size_t get_input_size() const override { return input_tensor_info_.size(); }
  /// \brief Get a producer name and its output port index
  void get_input_node(size_t input_port_idx, std::string& producer_name,
                      std::string& producer_output_port_name,
                      size_t& producer_output_port_index) const override {
    // TODO: Needs implementation ? Benchmark/demo app worked fine even without
    // it.
    return;
  }
  /// \brief Get operation type
  const std::string& get_op_type() const override { return op_type_; }
  /// \brief Get node name
  const std::string& get_op_name() const override { return op_name_; }

  // DecoderBaseOperation Interface implementations :
  /// \brief Get input tensor name by index
  std::string get_input_tensor_name(size_t idx) const override {
    return input_tensor_info_[idx].m_tensor_name;
  }
  /// \brief Get input tensor type by index
  ov::element::Type get_input_tensor_type(size_t idx) const override {
    return input_tensor_info_[idx].m_element_type;
  }
  /// \brief Get output tensor name by index
  std::string get_output_tensor_name(size_t idx) const override {
    return output_tensor_info_[idx].m_tensor_name;
  }
  /// \brief Get output tensor type by index
  ov::element::Type get_output_tensor_type(size_t idx) const override {
    return output_tensor_info_[idx].m_element_type;
  }

  /// \brief Get input tensor info
  ov::frontend::tensorflow_lite::TensorMetaInfo get_input_tensor_info(
      size_t idx) const override {
    return input_tensor_info_[idx];
  }

  /// \brief Get output tensor info_
  ov::frontend::tensorflow_lite::TensorMetaInfo get_output_tensor_info(
      size_t idx) const override {
    return output_tensor_info_[idx];
  }

  /// \brief Get a number of outputs
  size_t get_output_size() const override { return output_tensor_info_.size(); }

  litert::Expected<ov::Any> fetch_attribute(const std::string& name) const;

 private:
  std::string op_type_;
  std::string op_name_;
  std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_tensor_info_;
  std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
      output_tensor_info_;
  const LiteRtOp litert_op_;
  const LiteRtOpCode litert_op_code_;
};

class DecoderTensor : public ov::frontend::tensorflow_lite::DecoderBaseTensor {
 public:
  explicit DecoderTensor(
      ov::frontend::tensorflow_lite::TensorMetaInfo tensor_meta_info,
      int64_t input_index, int64_t output_index)
      : m_tensor_meta_info(tensor_meta_info),
        input_index_(input_index),
        output_index_(output_index),
        op_name_("DecoderTensor") {};

  ov::frontend::tensorflow_lite::TensorMetaInfo get_tensor_info()
      const override {
    return m_tensor_meta_info;
  }

  /// \brief Get input index for tensor
  int64_t get_input_idx() const override { return input_index_; }

  /// \brief Get output index for tensor
  int64_t get_output_idx() const override { return output_index_; }

  /// \brief No attributes for tensor
  ov::Any get_attribute(const std::string& name) const override {
    LITERT_LOG(LITERT_ERROR, "get_attribute not implemented");
    return ov::Any{};
  }

  /// \brief No inputs for tensor
  size_t get_input_size() const override {
    LITERT_LOG(LITERT_ERROR, "get_input_size not implemented");
    return 0;
  }

  /// \brief No input nodes for tensor
  void get_input_node(size_t input_port_idx, std::string& producer_name,
                      std::string& producer_output_port_name,
                      size_t& producer_output_port_index) const override {
    LITERT_LOG(LITERT_ERROR, "get_input_node not implemented");
  }

  /// \brief No operation for tensor
  const std::string& get_op_type() const override { return op_type_; };

  /// \brief No operation name for tensor
  const std::string& get_op_name() const override {
    LITERT_LOG(LITERT_ERROR, "get_op_name not implemented");
    return op_name_;
  };

 private:
  ov::frontend::tensorflow_lite::TensorMetaInfo m_tensor_meta_info;
  int64_t input_index_;
  int64_t output_index_;
  std::string op_type_;
  const std::string op_name_;
};

}  // namespace openvino
}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_DECODER_H_
