// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
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

// Canonical FULL LSTM inputs in TFLite slot order. Per-slot data types follow
// the INT8 row of the HTP op-definition supplement.
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

// Static FP32 weights for the numeric check. Zero recurrent weights, zero
// peepholes and an identity projection make each step depend only on the
// input-to-* weights and the biases, so the goldens below follow from the
// reference LSTM equations in MasterOpDef by hand.
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

// Static quantized CIFG tensors with LayerNorm. Peephole and projection slots
// stay null, matching the LayerNorm pattern HTP supports.
std::vector<::qnn::TensorWrapperRef> CreateLayerNormExecuteInputs(
    ::qnn::TensorPool& tensor_pool, ::qnn::TensorWrapper& input,
    ::qnn::TensorWrapper& output_state, ::qnn::TensorWrapper& cell_state) {
  static constexpr std::array<std::int8_t, 4> kWeights{64, -32, 16, 48};
  static constexpr std::array<std::int32_t, 2> kBiases{1024, -512};
  static constexpr std::array<std::int16_t, 2> kLayerNormCoefficients{16384,
                                                                     8192};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kWeightQuant{1.0f / 128, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{1.0f / 16384, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kLayerNormQuant{1.0f / 32768,
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
    return make_static(QNN_DATATYPE_SFIXED_POINT_16, kLayerNormQuant,
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

// TFLite slot order differs from the QNN Lstm signature, so verify every slot
// lands where MasterOpDef says. Also covers the 24-slot LayerNorm form, which
// only differs at QNN in[12..15], and null optional slots, which must stay null
// rather than shift the signature.
TEST_P(QnnModelTest, LstmRemapsTfliteSlotsToQnnOrder) {
  static constexpr std::uint32_t kNumBatch{2};
  static constexpr std::uint32_t kNumInput{3};
  static constexpr std::uint32_t kNumCell{4};
  static constexpr std::uint32_t kNumTime{5};
  static constexpr float kCellClip{8.0f};
  static constexpr float kProjClip{0.0f};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{1.0f / 256, 0};
  const std::vector<std::uint32_t> kStateDims{kNumBatch, kNumCell};

  auto inputs = CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, kStateDims);

  auto ops = ::qnn::BuildLstmOp(tensor_pool_, inputs, {output_tensor},
                                kCellClip, kProjClip, /*time_major=*/false);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kLstm);
  ASSERT_EQ(ops[0].GetInputCount(), 25u);

  // QNN in[0..9]: input, the forget/cell/output weight and bias groups. The
  // input-gate group is excluded here and reappears at in[16..21].
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

  // QNN in[10..11]: the state inputs.
  EXPECT_EQ(ops[0].GetInputTensor(10).GetName(), "output_state");
  EXPECT_EQ(ops[0].GetInputTensor(11).GetName(), "cell_state");

  // QNN in[12..15]: LayerNorm, absent from the 20-slot form.
  for (std::uint32_t i = 12; i < 16; ++i) {
    EXPECT_TRUE(ops[0].GetInputTensor(i).IsTensorNull());
  }

  // QNN in[16..23]: the input-gate group, peepholes and projection, which
  // TFLite places before the state tensors.
  EXPECT_EQ(ops[0].GetInputTensor(16).GetName(), "input_to_input");
  EXPECT_EQ(ops[0].GetInputTensor(17).GetName(), "recurrent_to_input");
  EXPECT_EQ(ops[0].GetInputTensor(18).GetName(), "cell_to_input");
  EXPECT_EQ(ops[0].GetInputTensor(19).GetName(), "cell_to_forget");
  EXPECT_EQ(ops[0].GetInputTensor(20).GetName(), "cell_to_output");
  EXPECT_EQ(ops[0].GetInputTensor(21).GetName(), "input_gate_bias");
  EXPECT_EQ(ops[0].GetInputTensor(22).GetName(), "projection_weights");
  EXPECT_EQ(ops[0].GetInputTensor(23).GetName(), "projection_bias");

  // QNN in[24] is the reset signal, only meaningful for a 3D input.
  EXPECT_TRUE(ops[0].GetInputTensor(24).IsTensorNull());

  // TFLite exposes one output, so out[0] and out[1] are synthesized. For a 2D
  // output MasterOpDef makes out[2] identical to out[0], and the TFLite tensor
  // goes to out[2]. out[1] is the cell state, [batch, num_units].
  EXPECT_EQ(ops[0].GetOutputTensor(0).GetDimensions(), kStateDims);
  EXPECT_EQ(ops[0].GetOutputTensor(1).GetDimensions(), kStateDims);
  EXPECT_EQ(ops[0].GetOutputTensor(2).GetName(), "output");

  // For a 3D output the TFLite tensor moves to out[0] and out[2] becomes the
  // synthesized final step, always [batch, output] however time_major orders
  // out[0]. Taking the trailing two dimensions is right only for time-major.
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
    EXPECT_EQ(seq_ops[0].GetOutputTensor(0).GetName(),
              time_major ? "seq_time_major" : "seq_batch_major");
    EXPECT_EQ(seq_ops[0].GetOutputTensor(0).GetDimensions(), seq_dims);
    EXPECT_EQ(seq_ops[0].GetOutputTensor(1).GetDimensions(), kStateDims);
    EXPECT_EQ(seq_ops[0].GetOutputTensor(2).GetDimensions(), kStepDims);
  }

  // The 24-slot form only adds LayerNorm at QNN in[12..15].
  auto layer_norm_inputs =
      CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell,
                       /*with_layer_norm=*/true);
  auto layer_norm_ops =
      ::qnn::BuildLstmOp(tensor_pool_, layer_norm_inputs, {output_tensor},
                         kCellClip, kProjClip, /*time_major=*/false);
  ASSERT_EQ(layer_norm_ops.size(), 1u);
  ASSERT_EQ(layer_norm_ops[0].GetInputCount(), 25u);
  EXPECT_EQ(layer_norm_ops[0].GetInputTensor(12).GetName(), "input_layer_norm");
  EXPECT_EQ(layer_norm_ops[0].GetInputTensor(13).GetName(),
            "forget_layer_norm");
  EXPECT_EQ(layer_norm_ops[0].GetInputTensor(14).GetName(), "cell_layer_norm");
  EXPECT_EQ(layer_norm_ops[0].GetInputTensor(15).GetName(),
            "output_layer_norm");

  // Null optional inputs must stay null at their remapped slots, not shift the
  // signature. Nulling TFLite 1/5/9/10/11/12/16/17 blanks QNN in[16..23].
  auto optional_inputs =
      CreateLstmInputs(tensor_pool_, kNumBatch, kNumInput, kNumCell);
  auto& null_tensor = tensor_pool_.CreateNullTensor();
  for (const size_t slot : {1, 5, 9, 10, 11, 12, 16, 17}) {
    optional_inputs[slot] = null_tensor;
  }
  auto optional_ops =
      ::qnn::BuildLstmOp(tensor_pool_, optional_inputs, {output_tensor},
                         kCellClip, kProjClip, /*time_major=*/false);
  ASSERT_EQ(optional_ops.size(), 1u);
  for (std::uint32_t i = 16; i < 24; ++i) {
    EXPECT_TRUE(optional_ops[0].GetInputTensor(i).IsTensorNull());
  }

  // Only a 2D step or a 3D sequence output is legal; any other rank would
  // otherwise take the 2D path and be silently mislowered.
  for (const std::vector<std::uint32_t>& bad_dims :
       {std::vector<std::uint32_t>{kNumCell},
        std::vector<std::uint32_t>{kNumTime, kNumBatch, kNumTime, kNumCell}}) {
    auto& bad_output = tensor_pool_.CreateOutputTensorWithName(
        "bad_rank_output", QNN_DATATYPE_UFIXED_POINT_8, kActQuant, bad_dims);
    EXPECT_TRUE(::qnn::BuildLstmOp(tensor_pool_, inputs, {bad_output},
                                   kCellClip, kProjClip, /*time_major=*/false)
                    .empty());
  }
}

// The remap above is checked in isolation. This drives the quantized op through
// the QNN backend so the HTP validator and graph finalizer confirm the slot
// order, once without LayerNorm to cover the per-gate qscales a float Lstm
// omits, and once with it -- HTP fails graph prepare if the qscales are set
// alongside LayerNorm coefficients, so the builder must drop them there.
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

  // A LayerNorm Lstm in the same graph, so one Finalize covers both qscale
  // paths. CIFG with null peepholes and projection is the form HTP supports.
  static constexpr std::uint32_t kNumLayerNormCell{2};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kLayerNormActQuant{1.0f / 128,
                                                                   -128};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kCellStateQuant{1.0f / 32768,
                                                                0};
  const std::vector<std::uint32_t> kLayerNormDims{kNumBatch, kNumLayerNormCell};
  auto& layer_norm_input = tensor_pool_.CreateInputTensorWithName(
      "ln_input", QNN_DATATYPE_UFIXED_POINT_8, kLayerNormActQuant,
      kLayerNormDims);
  auto& layer_norm_output_state = tensor_pool_.CreateInputTensorWithName(
      "ln_output_state", QNN_DATATYPE_UFIXED_POINT_8, kLayerNormActQuant,
      kLayerNormDims);
  auto& layer_norm_cell_state = tensor_pool_.CreateInputTensorWithName(
      "ln_cell_state", QNN_DATATYPE_SFIXED_POINT_16, kCellStateQuant,
      kLayerNormDims);
  auto& layer_norm_output = tensor_pool_.CreateOutputTensorWithName(
      "ln_output", QNN_DATATYPE_UFIXED_POINT_8, kLayerNormActQuant,
      kLayerNormDims);

  auto layer_norm_inputs = CreateLayerNormExecuteInputs(
      tensor_pool_, layer_norm_input, layer_norm_output_state,
      layer_norm_cell_state);
  auto layer_norm_ops =
      ::qnn::BuildLstmOp(tensor_pool_, layer_norm_inputs, {layer_norm_output},
                         kCellClip, kProjClip, /*time_major=*/false);
  ASSERT_FALSE(layer_norm_ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  qnn_model_.MoveOpsToGraph(std::move(layer_norm_ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());
}

// A 3D input is a sequence, which QNN returns on out[0] rather than the
// final-step out[2]. Routing it to out[2] instead would not even match shapes,
// so Finalize covers this without a device; on HTP the goldens additionally pin
// the output order. Two steps make the per-step values differ so a
// final-step-only bug shows.
TEST_P(QnnModelTest, LstmExecutesTimeMajorSequence) {
  static constexpr std::uint32_t kNumTime{2};
  static constexpr std::uint32_t kNumBatch{1};
  static constexpr std::uint32_t kNumInput{2};
  static constexpr std::uint32_t kNumCell{2};
  static constexpr float kCellClip{0.0f};
  static constexpr float kProjClip{0.0f};
  // Time-major [time, batch, input]: step 0 = {1, 2}, step 1 = {2, 1}.
  static constexpr std::array<float, 4> kInputData{1.0f, 2.0f, 2.0f, 1.0f};
  // Full [time, batch, output] sequence from zeroed state.
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
  EXPECT_EQ(ops[0].GetOutputTensor(0).GetName(), "output");

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
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

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), kExpectedSequence));
}

}  // namespace
}  // namespace litert::qnn
