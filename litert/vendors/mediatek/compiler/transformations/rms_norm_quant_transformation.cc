// Copyright 2026 Google LLC.
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

#include "litert/vendors/mediatek/compiler/transformations/rms_norm_quant_transformation.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

using litert::ElementType;
using litert::Error;
using litert::Expected;
using litert::GetByteWidth;
using litert::RankedTensorType;
using litert::compiler::Builder;
using litert::compiler::CompositeOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CompositeOp;
using litert::compiler::m_Predicate;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

namespace {

Expected<std::vector<float>> DequantizeWeights(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto ranked_type, tensor.RankedTensorType());
  size_t num_elements = 1;
  for (auto dim : ranked_type.Layout().Dimensions()) {
    if (dim <= 0) {
      return Error(
          kLiteRtStatusErrorUnsupported,
          "Dynamic or negative dimensions are not supported for weights "
          "dequantization");
    }
    num_elements *= dim;
  }

  auto byte_width = GetByteWidth(ranked_type.ElementType());
  if (!byte_width) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported element type for byte width calculation");
  }
  size_t expected_bytes = byte_width->NumBytes(num_elements);
  auto bytes = tensor.Weights().Bytes();
  if (bytes.size() < expected_bytes) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Weight buffer size mismatch");
  }

  std::vector<float> f32_data(num_elements);

  if (tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
    auto q = tensor.PerTensorQuantization();
    float scale = q.scale;
    int64_t zero_point = q.zero_point;

    switch (ranked_type.ElementType()) {
      case ElementType::Int8: {
        const int8_t* src = reinterpret_cast<const int8_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          f32_data[i] = (static_cast<float>(src[i]) - zero_point) * scale;
        }
        break;
      }
      case ElementType::Int16: {
        const int16_t* src = reinterpret_cast<const int16_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          f32_data[i] = (static_cast<float>(src[i]) - zero_point) * scale;
        }
        break;
      }
      case ElementType::UInt8: {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          f32_data[i] = (static_cast<float>(src[i]) - zero_point) * scale;
        }
        break;
      }
      case ElementType::UInt16: {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          f32_data[i] = (static_cast<float>(src[i]) - zero_point) * scale;
        }
        break;
      }
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported element type for weights dequantization");
    }
  } else if (tensor.QTypeId() == kLiteRtQuantizationPerChannel) {
    auto q = tensor.PerChannelQuantization();
    if (q.scales == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Per-channel quantization scales missing");
    }
    if (num_elements != q.num_channels) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported multi-dimensional per-channel quantization");
    }
    switch (ranked_type.ElementType()) {
      case ElementType::Int8: {
        const int8_t* src = reinterpret_cast<const int8_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          int64_t zp = (q.zero_points != nullptr) ? q.zero_points[i] : 0;
          f32_data[i] = (static_cast<float>(src[i]) - zp) * q.scales[i];
        }
        break;
      }
      case ElementType::Int16: {
        const int16_t* src = reinterpret_cast<const int16_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          int64_t zp = (q.zero_points != nullptr) ? q.zero_points[i] : 0;
          f32_data[i] = (static_cast<float>(src[i]) - zp) * q.scales[i];
        }
        break;
      }
      case ElementType::UInt8: {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          int64_t zp = (q.zero_points != nullptr) ? q.zero_points[i] : 0;
          f32_data[i] = (static_cast<float>(src[i]) - zp) * q.scales[i];
        }
        break;
      }
      case ElementType::UInt16: {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
        for (size_t i = 0; i < num_elements; ++i) {
          int64_t zp = (q.zero_points != nullptr) ? q.zero_points[i] : 0;
          f32_data[i] = (static_cast<float>(src[i]) - zp) * q.scales[i];
        }
        break;
      }
      default:
        return Error(
            kLiteRtStatusErrorUnsupported,
            "Unsupported element type for per-channel weights dequantization");
    }
  } else if (tensor.QTypeId() == kLiteRtQuantizationBlockWise) {
    LITERT_LOG(
        LITERT_WARNING,
        "Block-wise quantization is unsupported for weights dequantization");
    return Error(
        kLiteRtStatusErrorUnsupported,
        "Block-wise quantization is unsupported for weights dequantization");
  } else {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported quantization type for weights dequantization");
  }

  return f32_data;
}

}  // namespace

extern "C" {

LiteRtStatus RmsNormQuantTransformation(const LiteRtCompilerContext* context,
                                        LiteRtBuilder builder_ptr,
                                        LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  Tensor input_tensor(context, nullptr);
  Tensor gamma_tensor(context, nullptr);

  auto not_blockwise = m_Predicate<Tensor>([](const Tensor& t) {
    if (t.QTypeId() == kLiteRtQuantizationBlockWise) {
      LITERT_LOG(LITERT_WARNING,
                 "Block-wise quantization is unsupported in "
                 "RmsNormQuantTransformation; "
                 "skipping transformation.");
      return false;
    }
    return true;
  });

  auto pattern = m_AllOf(
      m_CompositeOp(
          "odml.rms_norm",
          m_AllOf(m_CaptureOrSameAs(&input_tensor, m_Any()), not_blockwise),
          m_AllOf(m_CaptureOrSameAs(&gamma_tensor, m_Any()), not_blockwise)),
      m_Predicate<Op>([&not_blockwise](const Op& op) {
        auto outputs = op.Outputs();
        if (outputs.size() != 1) return false;
        if (!not_blockwise.Match(outputs[0], nullptr)) return false;
        return op.Inputs()[0].HasQuantization() ||
               op.Inputs()[1].HasQuantization() || outputs[0].HasQuantization();
      }));

  if (!Match(root_op, pattern)) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor output_tensor = root_op.Outputs()[0];
  bool is_input_quant = input_tensor.HasQuantization();
  bool is_gamma_quant = gamma_tensor.HasQuantization();
  bool is_output_quant = output_tensor.HasQuantization();

  Tensor f32_input = input_tensor;
  if (is_input_quant) {
    auto in_type = input_tensor.RankedTensorType();
    if (!in_type) {
      return in_type.Error().Status();
    }
    RankedTensorType f32_type = *in_type;
    f32_type.SetElementType(ElementType::Float32);
    auto new_tensor = builder.BuildTensor(
        RankedTensorSpecBuilder(std::move(f32_type)).Build());
    if (!new_tensor) {
      return new_tensor.Error().Status();
    }
    f32_input = *new_tensor;
    auto dequant_op = builder.BuildOp(kLiteRtOpCodeTflDequantize,
                                      {input_tensor}, {f32_input});
    if (!dequant_op) {
      return dequant_op.Error().Status();
    }
  }

  Tensor f32_gamma = gamma_tensor;
  if (is_gamma_quant) {
    auto gamma_type = gamma_tensor.RankedTensorType();
    if (!gamma_type) {
      return gamma_type.Error().Status();
    }
    RankedTensorType f32_type = *gamma_type;
    f32_type.SetElementType(ElementType::Float32);

    if (gamma_tensor.HasWeights()) {
      // Constant folding: dequantize weights offline at compile-time into a
      // float buffer and attach directly to f32_gamma without inserting a
      // dequantize op.
      auto f32_weights_data = DequantizeWeights(gamma_tensor);
      if (!f32_weights_data) {
        return f32_weights_data.Error().Status();
      }

      auto new_tensor = builder.BuildTensor(
          RankedTensorSpecBuilder(std::move(f32_type)).Build());
      if (!new_tensor) {
        return new_tensor.Error().Status();
      }
      f32_gamma = *new_tensor;

      auto build_w_res = builder.BuildWeights<float>(
          absl::MakeConstSpan(*f32_weights_data), f32_gamma);
      if (!build_w_res) {
        return build_w_res.Error().Status();
      }
    } else {
      // Dynamic activation fallback: insert runtime dequantize op.
      auto new_tensor = builder.BuildTensor(
          RankedTensorSpecBuilder(std::move(f32_type)).Build());
      if (!new_tensor) {
        return new_tensor.Error().Status();
      }
      f32_gamma = *new_tensor;
      auto dequant_op = builder.BuildOp(kLiteRtOpCodeTflDequantize,
                                        {gamma_tensor}, {f32_gamma});
      if (!dequant_op) {
        return dequant_op.Error().Status();
      }
    }
  }

  Tensor f32_output = output_tensor;
  if (is_output_quant) {
    auto out_type = output_tensor.RankedTensorType();
    if (!out_type) {
      return out_type.Error().Status();
    }
    RankedTensorType f32_type = *out_type;
    f32_type.SetElementType(ElementType::Float32);
    auto new_tensor = builder.BuildTensor(
        RankedTensorSpecBuilder(std::move(f32_type)).Build());
    if (!new_tensor) {
      return new_tensor.Error().Status();
    }
    f32_output = *new_tensor;
  }

  // Build a new composite op with float32 inputs and output
  auto new_rms_norm = builder.BuildOp(kLiteRtOpCodeShloComposite,
                                      {f32_input, f32_gamma}, {f32_output});
  if (!new_rms_norm) {
    return new_rms_norm.Error().Status();
  }

  // Copy composite options from the original op using SetOpOptions
  CompositeOptions options;
  LITERT_RETURN_IF_ERROR(options.InitFromOp(context, root_op.Get()));
  auto set_opts_res = builder.SetOpOptions(*new_rms_norm, std::move(options));
  if (!set_opts_res) {
    return set_opts_res.Error().Status();
  }

  if (is_output_quant) {
    auto quant_op = builder.BuildOp(kLiteRtOpCodeTflQuantize, {f32_output},
                                    {output_tensor});
    if (!quant_op) {
      return quant_op.Error().Status();
    }
  }

  builder.EraseOp(root_op);

  return kLiteRtStatusOk;
}

}  // extern "C"
