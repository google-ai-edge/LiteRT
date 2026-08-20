// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/lstm_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::FloatNear;
using testing::Pointwise;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// Builds canonical FULL LSTM inputs in TFLite slot order. Per-slot data types
// follow the INT8 row of the HTP op-definition supplement.
std::vector<::qnn::TensorWrapperRef> CreateLstmInputs(
    ::qnn::TensorPool& tensor_pool, std::uint32_t n_batch,
    std::uint32_t n_input, std::uint32_t n_cell, bool with_layer_norm = false) {
  static constexpr Qnn_DataType_t kActType{QNN_DATATYPE_UFIXED_POINT_8};
  static constexpr Qnn_DataType_t kWeightType{QNN_DATATYPE_UFIXED_POINT_8};
  static constexpr Qnn_DataType_t kBiasType{QNN_DATATYPE_SFIXED_POINT_32};
  static constexpr Qnn_DataType_t kPeepholeType{QNN_DATATYPE_SFIXED_POINT_16};
  static constexpr Qnn_DataType_t kCellStateType{QNN_DATATYPE_SFIXED_POINT_16};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 256, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kWeightQuant{1.0f / 256, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{1.0f / 65536, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kPeepholeQuant{1.0f / 32768, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kCellStateQuant{1.0f / 32768,
                                                                0};
  const std::vector<std::uint32_t> kInputDims{n_batch, n_input};
  const std::vector<std::uint32_t> kInputWeightDims{n_cell, n_input};
  const std::vector<std::uint32_t> kRecurrentWeightDims{n_cell, n_cell};
  const std::vector<std::uint32_t> kPeepholeDims{n_cell};
  const std::vector<std::uint32_t> kBiasDims{n_cell};
  const std::vector<std::uint32_t> kProjectionDims{n_cell, n_cell};
  const std::vector<std::uint32_t> kStateDims{n_batch, n_cell};

  std::vector<::qnn::TensorWrapperRef> inputs{
      tensor_pool.CreateInputTensorWithName("input", kActType, kActQuant,
                                            kInputDims),
      tensor_pool.CreateInputTensorWithName("input_to_input", kWeightType,
                                            kWeightQuant, kInputWeightDims),
      tensor_pool.CreateInputTensorWithName("input_to_forget", kWeightType,
                                            kWeightQuant, kInputWeightDims),
      tensor_pool.CreateInputTensorWithName("input_to_cell", kWeightType,
                                            kWeightQuant, kInputWeightDims),
      tensor_pool.CreateInputTensorWithName("input_to_output", kWeightType,
                                            kWeightQuant, kInputWeightDims),
      tensor_pool.CreateInputTensorWithName("recurrent_to_input", kWeightType,
                                            kWeightQuant, kRecurrentWeightDims),
      tensor_pool.CreateInputTensorWithName("recurrent_to_forget", kWeightType,
                                            kWeightQuant, kRecurrentWeightDims),
      tensor_pool.CreateInputTensorWithName("recurrent_to_cell", kWeightType,
                                            kWeightQuant, kRecurrentWeightDims),
      tensor_pool.CreateInputTensorWithName("recurrent_to_output", kWeightType,
                                            kWeightQuant, kRecurrentWeightDims),
      tensor_pool.CreateInputTensorWithName("cell_to_input", kPeepholeType,
                                            kPeepholeQuant, kPeepholeDims),
      tensor_pool.CreateInputTensorWithName("cell_to_forget", kPeepholeType,
                                            kPeepholeQuant, kPeepholeDims),
      tensor_pool.CreateInputTensorWithName("cell_to_output", kPeepholeType,
                                            kPeepholeQuant, kPeepholeDims),
      tensor_pool.CreateInputTensorWithName("input_gate_bias", kBiasType,
                                            kBiasQuant, kBiasDims),
      tensor_pool.CreateInputTensorWithName("forget_gate_bias", kBiasType,
                                            kBiasQuant, kBiasDims),
      tensor_pool.CreateInputTensorWithName("cell_gate_bias", kBiasType,
                                            kBiasQuant, kBiasDims),
      tensor_pool.CreateInputTensorWithName("output_gate_bias", kBiasType,
                                            kBiasQuant, kBiasDims),
      tensor_pool.CreateInputTensorWithName("projection_weights", kWeightType,
                                            kWeightQuant, kProjectionDims),
      tensor_pool.CreateInputTensorWithName("projection_bias", kBiasType,
                                            kBiasQuant, kBiasDims),
      tensor_pool.CreateInputTensorWithName("output_state", kActType, kActQuant,
                                            kStateDims),
      tensor_pool.CreateInputTensorWithName("cell_state", kCellStateType,
                                            kCellStateQuant, kStateDims),
  };

  if (with_layer_norm) {
    inputs.emplace_back(tensor_pool.CreateInputTensorWithName(
        "input_layer_norm", kPeepholeType, kPeepholeQuant, kPeepholeDims));
    inputs.emplace_back(tensor_pool.CreateInputTensorWithName(
        "forget_layer_norm", kPeepholeType, kPeepholeQuant, kPeepholeDims));
    inputs.emplace_back(tensor_pool.CreateInputTensorWithName(
        "cell_layer_norm", kPeepholeType, kPeepholeQuant, kPeepholeDims));
    inputs.emplace_back(tensor_pool.CreateInputTensorWithName(
        "output_layer_norm", kPeepholeType, kPeepholeQuant, kPeepholeDims));
  }

  return inputs;
}

// Static FP32 weights for the numeric check. The recurrent weights and the
// peepholes are zero and the projection is the identity, so each step of a
// 2-input, 2-cell LSTM depends only on the input-to-* weights and the biases.
std::vector<::qnn::TensorWrapperRef> CreateExecuteLstmInputs(
    ::qnn::TensorPool& tensor_pool, ::qnn::TensorWrapper& input,
    ::qnn::TensorWrapper& output_state, ::qnn::TensorWrapper& cell_state) {
  static constexpr Qnn_DataType_t kType{QNN_DATATYPE_FLOAT_32};
  static constexpr std::array<float, 4> kInputToInput{0.1f, 0.2f, 0.3f, 0.4f};
  static constexpr std::array<float, 4> kInputToForget{0.05f, 0.10f, 0.15f,
                                                       0.20f};
  static constexpr std::array<float, 4> kInputToCell{0.20f, 0.10f, 0.05f,
                                                     0.30f};
  static constexpr std::array<float, 4> kInputToOutput{0.10f, 0.10f, 0.20f,
                                                       0.20f};
  static constexpr std::array<float, 4> kRecurrentZeros{0.0f, 0.0f, 0.0f, 0.0f};
  static constexpr std::array<float, 2> kPeepholes{0.0f, 0.0f};
  static constexpr std::array<float, 2> kInputGateBias{0.1f, 0.1f};
  static constexpr std::array<float, 2> kForgetGateBias{0.2f, 0.2f};
  static constexpr std::array<float, 2> kCellGateBias{0.0f, 0.0f};
  static constexpr std::array<float, 2> kOutputGateBias{0.1f, 0.1f};
  static constexpr std::array<float, 4> kProjectionIdentity{1.0f, 0.0f, 0.0f,
                                                            1.0f};
  static constexpr std::array<float, 2> kProjectionBias{0.0f, 0.0f};
  const std::vector<std::uint32_t> kMatrixDims{2, 2};
  const std::vector<std::uint32_t> kVectorDims{2};

  auto make_static = [&tensor_pool](const std::vector<std::uint32_t>& dims,
                                    const auto& data) -> ::qnn::TensorWrapper& {
    return tensor_pool.CreateStaticTensor(kType, {}, dims,
                                          sizeof(float) * data.size(),
                                          data.data());
  };

  return {
      input,
      make_static(kMatrixDims, kInputToInput),
      make_static(kMatrixDims, kInputToForget),
      make_static(kMatrixDims, kInputToCell),
      make_static(kMatrixDims, kInputToOutput),
      make_static(kMatrixDims, kRecurrentZeros),
      make_static(kMatrixDims, kRecurrentZeros),
      make_static(kMatrixDims, kRecurrentZeros),
      make_static(kMatrixDims, kRecurrentZeros),
      make_static(kVectorDims, kPeepholes),
      make_static(kVectorDims, kPeepholes),
      make_static(kVectorDims, kPeepholes),
      make_static(kVectorDims, kInputGateBias),
      make_static(kVectorDims, kForgetGateBias),
      make_static(kVectorDims, kCellGateBias),
      make_static(kVectorDims, kOutputGateBias),
      make_static(kMatrixDims, kProjectionIdentity),
      make_static(kVectorDims, kProjectionBias),
      output_state,
      cell_state,
  };
}

// Static quantized CIFG tensors for LayerNorm offline preparation. Peephole
// and projection slots remain canonical nulls, matching the HTP-supported
// LayerNorm pattern while still exercising the 24-slot FULL kernel form.
std::vector<::qnn::TensorWrapperRef> CreateLayerNormOfflinePrepareInputs(
    ::qnn::TensorPool& tensor_pool, ::qnn::TensorWrapper& input,
    ::qnn::TensorWrapper& output_state, ::qnn::TensorWrapper& cell_state) {
  static constexpr std::array<std::int8_t, 4> kWeights{64, -32, 16, 48};
  static constexpr std::array<std::int32_t, 2> kBiases{1024, -512};
  static constexpr std::array<std::int16_t, 2> kLayerNormCoefficients{
      16384, 8192};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kWeightQuant{1.0f / 128, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{1.0f / 16384, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kPeepholeQuant{1.0f / 32768,
                                                                0};
  const std::vector<std::uint32_t> kMatrixDims{2, 2};
  const std::vector<std::uint32_t> kVectorDims{2};

  auto make_static = [&tensor_pool](Qnn_DataType_t type, const auto& quant,
                                    const std::vector<std::uint32_t>& dims,
                                    const auto& data) -> ::qnn::TensorWrapper& {
    return tensor_pool.CreateStaticTensor(type, quant, dims,
                                          sizeof(data[0]) * data.size(),
                                          data.data());
  };
  auto make_weight = [&]() -> ::qnn::TensorWrapper& {
    return make_static(QNN_DATATYPE_SFIXED_POINT_8, kWeightQuant, kMatrixDims,
                       kWeights);
  };
  auto make_bias = [&]() -> ::qnn::TensorWrapper& {
    return make_static(QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, kVectorDims,
                       kBiases);
  };
  auto make_layer_norm = [&]() -> ::qnn::TensorWrapper& {
    return make_static(QNN_DATATYPE_SFIXED_POINT_16, kPeepholeQuant,
                       kVectorDims, kLayerNormCoefficients);
  };

  return {
      input,                          tensor_pool.CreateNullTensor(),
      make_weight(),                  make_weight(),
      make_weight(),                  tensor_pool.CreateNullTensor(),
      make_weight(),                  make_weight(),
      make_weight(),                  tensor_pool.CreateNullTensor(),
      tensor_pool.CreateNullTensor(), tensor_pool.CreateNullTensor(),
      tensor_pool.CreateNullTensor(), make_bias(),
      make_bias(),                    make_bias(),
      tensor_pool.CreateNullTensor(), tensor_pool.CreateNullTensor(),
      output_state,                   cell_state,
      tensor_pool.CreateNullTensor(), make_layer_norm(),
      make_layer_norm(),              make_layer_norm(),
  };
}

// The TFLite slot order differs from the QNN Lstm signature, so verify the
// remap puts each tensor in the QNN slot the master op definition expects.
TEST_P(QnnModelTest, LstmRemapsCanonicalFullSlots) {
  static constexpr std::uint32_t kNumBatch{2};
  static constexpr std::uint32_t kNumInput{3};
  static constexpr std::uint32_t kNumCell{4};
  static constexpr std::uint32_t kNumTime{5};
  static constexpr float kCellClip{8.0f};
  static constexpr float kProjClip{0.0f};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 256, 0};

  auto inputs = CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, {kNumBatch, kNumCell});

  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                kCellClip, kProjClip, /*time_major=*/false);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kLstm);
  ASSERT_EQ(ops[0].GetInputCount(), 25u);

  // QNN slots 0-9: input, the three input-to-* and three recurrent-to-* gates,
  // then the three gate biases -- all with the input-gate group excluded.
  EXPECT_EQ(ops[0].GetInputTensor(0).GetName(), "input");
  EXPECT_EQ(ops[0].GetInputTensor(1).GetName(), "input_to_forget");
  EXPECT_EQ(ops[0].GetInputTensor(2).GetName(), "input_to_cell");
  EXPECT_EQ(ops[0].GetInputTensor(3).GetName(), "input_to_output");
  EXPECT_EQ(ops[0].GetInputTensor(4).GetName(), "recurrent_to_forget");
  EXPECT_EQ(ops[0].GetInputTensor(5).GetName(), "recurrent_to_cell");
  EXPECT_EQ(ops[0].GetInputTensor(6).GetName(), "recurrent_to_output");
  EXPECT_EQ(ops[0].GetInputTensor(7).GetName(), "forget_gate_bias");
  EXPECT_EQ(ops[0].GetInputTensor(8).GetName(), "cell_gate_bias");
  EXPECT_EQ(ops[0].GetInputTensor(9).GetName(), "output_gate_bias");

  // QNN slots 10-11: the two state inputs.
  EXPECT_EQ(ops[0].GetInputTensor(10).GetName(), "output_state");
  EXPECT_EQ(ops[0].GetInputTensor(11).GetName(), "cell_state");

  // QNN slots 12-15 are the layer-norm weights, which HTP does not support, so
  // the builder leaves them unconnected.
  for (std::uint32_t i = 12; i < 16; ++i) {
    EXPECT_TRUE(ops[0].GetInputTensor(i).IsTensorNull());
  }

  // QNN slots 16-23: the input-gate group, peepholes and projection, which
  // TFLite places before the state tensors.
  EXPECT_EQ(ops[0].GetInputTensor(16).GetName(), "input_to_input");
  EXPECT_EQ(ops[0].GetInputTensor(17).GetName(), "recurrent_to_input");
  EXPECT_EQ(ops[0].GetInputTensor(18).GetName(), "cell_to_input");
  EXPECT_EQ(ops[0].GetInputTensor(19).GetName(), "cell_to_forget");
  EXPECT_EQ(ops[0].GetInputTensor(20).GetName(), "cell_to_output");
  EXPECT_EQ(ops[0].GetInputTensor(21).GetName(), "input_gate_bias");
  EXPECT_EQ(ops[0].GetInputTensor(22).GetName(), "projection_weights");
  EXPECT_EQ(ops[0].GetInputTensor(23).GetName(), "projection_bias");

  // QNN slot 24 is the reset signal, only meaningful for a 3D input.
  EXPECT_TRUE(ops[0].GetInputTensor(24).IsTensorNull());

  // TFLite exposes only the final output, so the builder must synthesize the
  // two state outputs QNN requires; the TFLite output lands in slot 2.
  EXPECT_EQ(ops[0].GetOutputTensor(2).GetName(), "output");

  // For a 3D output the TFLite tensor moves to out[0] and out[2] becomes the
  // synthesized final step, which is always [batch, output] however time_major
  // orders out[0]. Check both orderings, since taking the trailing two
  // dimensions happens to be right only for the time-major one.
  const std::vector<std::uint32_t> kStepDims{kNumBatch, kNumCell};
  for (const bool time_major : {false, true}) {
    const std::vector<std::uint32_t> seq_dims =
        time_major ? std::vector<std::uint32_t>{kNumTime, kNumBatch, kNumCell}
                   : std::vector<std::uint32_t>{kNumBatch, kNumTime, kNumCell};
    auto& seq_output = tensor_pool_.CreateOutputTensorWithName(
        time_major ? "seq_time_major" : "seq_batch_major",
        QNN_DATATYPE_UFIXED_POINT_8, kActQuant, seq_dims);
    auto seq_ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {seq_output},
                                      kCellClip, kProjClip, time_major);
    ASSERT_EQ(seq_ops.size(), 1u);
    EXPECT_EQ(seq_ops[0].GetOutputTensor(0).GetDimensions(), seq_dims);
    EXPECT_EQ(seq_ops[0].GetOutputTensor(2).GetDimensions(), kStepDims);
  }

}

TEST_P(QnnModelTest, LstmRemapsLayerNormFullSlots) {
  static constexpr std::uint32_t kNumBatch{1};
  static constexpr std::uint32_t kNumInput{3};
  static constexpr std::uint32_t kNumCell{4};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 256, 0};

  auto inputs =
      CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell, true);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, {kNumBatch, kNumCell});

  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                /*cell_clip=*/0.0f, /*proj_clip=*/0.0f,
                                /*time_major=*/false);
  ASSERT_EQ(ops.size(), 1u);
  ASSERT_EQ(ops[0].GetInputCount(), 25u);
  EXPECT_EQ(ops[0].GetInputTensor(12).GetName(), "input_layer_norm");
  EXPECT_EQ(ops[0].GetInputTensor(13).GetName(), "forget_layer_norm");
  EXPECT_EQ(ops[0].GetInputTensor(14).GetName(), "cell_layer_norm");
  EXPECT_EQ(ops[0].GetInputTensor(15).GetName(), "output_layer_norm");
}

TEST_P(QnnModelTest, LstmPreservesCanonicalOptionalSlots) {
  static constexpr std::uint32_t kNumBatch{1};
  static constexpr std::uint32_t kNumInput{3};
  static constexpr std::uint32_t kNumCell{4};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 256, 0};

  auto inputs = CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell);
  auto& null_tensor = tensor_pool_.CreateNullTensor();
  inputs[1] = null_tensor;   // input_to_input_weights
  inputs[5] = null_tensor;   // recurrent_to_input_weights
  inputs[9] = null_tensor;   // cell_to_input_weights
  inputs[10] = null_tensor;  // cell_to_forget_weights
  inputs[11] = null_tensor;  // cell_to_output_weights
  inputs[12] = null_tensor;  // input_gate_bias
  inputs[16] = null_tensor;  // projection_weights
  inputs[17] = null_tensor;  // projection_bias

  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, {kNumBatch, kNumCell});
  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                /*cell_clip=*/0.0f, /*proj_clip=*/0.0f,
                                /*time_major=*/false);

  ASSERT_EQ(ops.size(), 1u);
  EXPECT_TRUE(ops[0].GetInputTensor(16).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(17).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(18).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(19).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(20).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(21).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(22).IsTensorNull());
  EXPECT_TRUE(ops[0].GetInputTensor(23).IsTensorNull());
}

// The remap above is checked in isolation. This drives the op through the QNN
// backend so the HTP validator and the graph finalizer confirm the slot order
// is the one QNN expects. Uses the quantized form, which additionally needs
// the per-gate quantization scales a float Lstm does not.
TEST_P(QnnModelTest, LstmQuantizedFinalizesOnQnnBackend) {
  static constexpr std::uint32_t kNumBatch{1};
  static constexpr std::uint32_t kNumInput{4};
  static constexpr std::uint32_t kNumCell{4};
  static constexpr float kCellClip{0.0f};
  static constexpr float kProjClip{0.0f};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 256, 0};

  auto inputs = CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, {kNumBatch, kNumCell});

  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                kCellClip, kProjClip, /*time_major=*/false);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());
}

TEST_P(QnnModelTest, LstmQuantizedLayerNormPrepareAndExecute) {
  static constexpr std::uint32_t kNumBatch{1};
  static constexpr std::uint32_t kNumCell{2};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 128, -128};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kCellStateQuant{1.0f / 32768,
                                                                0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, {kNumBatch, kNumCell});
  auto& output_state_tensor = tensor_pool_.CreateInputTensorWithName(
      "output_state", QNN_DATATYPE_UFIXED_POINT_8, kActQuant,
      {kNumBatch, kNumCell});
  auto& cell_state_tensor = tensor_pool_.CreateInputTensorWithName(
      "cell_state", QNN_DATATYPE_SFIXED_POINT_16, kCellStateQuant,
      {kNumBatch, kNumCell});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, {kNumBatch, kNumCell});

#if defined(__ANDROID__)
  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_state_idx = qnn_model_.AddInputTensor(output_state_tensor);
  auto cell_state_idx = qnn_model_.AddInputTensor(cell_state_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
#endif

  auto inputs = CreateLayerNormOfflinePrepareInputs(
      tensor_pool_, input_tensor, output_state_tensor, cell_state_tensor);
  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                /*cell_clip=*/0.0f, /*proj_clip=*/0.0f,
                                /*time_major=*/false);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if defined(__ANDROID__)
  static constexpr std::array<std::uint8_t, 2> kZeroActivation{128, 128};
  static constexpr std::array<std::int16_t, 2> kZeroCellState{0, 0};
  ASSERT_TRUE(qnn_model_.SetInputData<std::uint8_t>(input_idx,
                                                    kZeroActivation));
  ASSERT_TRUE(qnn_model_.SetInputData<std::uint8_t>(output_state_idx,
                                                    kZeroActivation));
  ASSERT_TRUE(qnn_model_.SetInputData<std::int16_t>(cell_state_idx,
                                                    kZeroCellState));
                                                    
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::uint8_t>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(), testing::ElementsAre(0, 0));
#endif

  const char* context_binary_path =
      std::getenv("LITERT_QNN_CONTEXT_BINARY_OUTPUT");
  if (context_binary_path != nullptr) {
    std::vector<char> context_binary;
    ASSERT_EQ(qnn_manager_ptr_->GenerateContextBinary(context_handle_.Get(),
                                                       context_binary),
              kLiteRtStatusOk);
    std::ofstream output(context_binary_path,
                         std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    output.write(context_binary.data(), context_binary.size());
    ASSERT_TRUE(output.good());
  }
}

// A 3D input is a time-major sequence: TFLite's UnidirectionalSequenceLSTM
// emits the full [time, batch, output] sequence (the kernel resizes the output
// to the input shape), which QNN exposes on out[0], not on the final-step
// out[2]. So the builder must route the TFLite output to out[0] for a 3D
// input; had it kept it on out[2] the shapes would not match and Finalize
// would fail, which is what this covers when no device is present. On a real
// HTP it also checks the values, pinning the output order rather than just the
// shape. Two steps let the per-step values differ so a final-step-only bug
// shows in the goldens.
TEST_P(QnnModelTest, LstmExecutesTimeMajorSequence) {
  static constexpr std::uint32_t kNumTime{2};
  static constexpr std::uint32_t kNumBatch{1};
  static constexpr std::uint32_t kNumInput{2};
  static constexpr std::uint32_t kNumCell{2};
  static constexpr float kCellClip{0.0f};
  static constexpr float kProjClip{0.0f};
  // Time-major [time, batch, input]: step 0 = {1, 2}, step 1 = {2, 1}.
  static constexpr std::array<float, 4> kInputData{1.0f, 2.0f, 2.0f, 1.0f};
  // Full [time, batch, output] sequence from zeroed state, computed from the
  // reference LSTM equations.
  static constexpr std::array<float, 4> kExpectedSequence{
      0.1439910f, 0.2760279f, 0.2449168f, 0.3485698f};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, {kNumTime, kNumBatch, kNumInput});
  auto& output_state_tensor = tensor_pool_.CreateInputTensorWithName(
      "output_state", QNN_DATATYPE_FLOAT_32, {}, {kNumBatch, kNumCell});
  auto& cell_state_tensor = tensor_pool_.CreateInputTensorWithName(
      "cell_state", QNN_DATATYPE_FLOAT_32, {}, {kNumBatch, kNumCell});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, {kNumTime, kNumBatch, kNumCell});

  auto inputs = CreateExecuteLstmInputs(tensor_pool_, input_tensor,
                                        output_state_tensor, cell_state_tensor);
  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                kCellClip, kProjClip, /*time_major=*/true);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_state_idx = qnn_model_.AddInputTensor(output_state_tensor);
  auto cell_state_idx = qnn_model_.AddInputTensor(cell_state_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx, kInputData);
  qnn_model_.SetInputData<float>(output_state_idx, {0.0f, 0.0f});
  qnn_model_.SetInputData<float>(cell_state_idx, {0.0f, 0.0f});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), kExpectedSequence));
}

}  // namespace
}  // namespace litert::qnn
