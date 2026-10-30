// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"

#include <algorithm>
#include <initializer_list>

namespace litert::arm_vulkan_ml {

const std::vector<OperationRegistration>& GetOperationRegistrations() {
  static const std::vector<OperationRegistration> registrations = [] {
    std::vector<OperationRegistration> result;
    using R = LoweringRoute;
    constexpr auto i8 = kLiteRtElementTypeInt8;
    constexpr auto f32 = kLiteRtElementTypeFloat32;
    constexpr auto i32 = kLiteRtElementTypeInt32;
    constexpr auto u32 = kLiteRtElementTypeUInt32;
    constexpr auto boolean = kLiteRtElementTypeBool;
    constexpr auto c64 = kLiteRtElementTypeComplex64;
    auto add = [&](std::initializer_list<LiteRtOpCode> codes,
                   std::initializer_list<OperationVariant> variants,
                   int data_input = 0) {
      for (auto code : codes) result.push_back({code, data_input, variants});
    };
#define OP(name) kLiteRtOpCodeTfl##name
    add({OP(Add), OP(Mul)}, {
        {i8, i8, R::kGeneral,
            "Static broadcasting; per-tensor affine activations; proven INT32 "
            "ranges; NONE/RELU/RELU6"},
        {f32, f32, R::kGeneral, "Static broadcasting; NONE/RELU/RELU6"}});
    add({OP(Sub)}, {
        {i8, i8, R::kGeneral,
            "Static broadcasting; per-tensor affine activations; proven INT32 "
            "ranges; NONE/RELU/RELU6"},
        {f32, f32, R::kPortable, "Static broadcasting; NONE/RELU/RELU6"}});
    add({OP(Relu), OP(Relu6)}, {
        {i8, i8, R::kGeneral,
            "Matching shapes; affine INT8; activation/requantization bounds checked"},
        {f32, f32, R::kPortable, "Matching shapes"}});
    add({OP(Abs)}, {
        {i8, i8, R::kGeneral,
            "Matching shapes; affine INT8; requantization range proven; TABLE"},
        {f32, f32, R::kPortable, "Matching shapes"}});
    add({OP(Tanh)}, {
        {i8, i8, R::kGeneral,
            "Matching shapes; output scale=1/128, zero_point=0; TABLE"},
        {f32, f32, R::kPortable, "Matching shapes"}});
    add({OP(Logistic)}, {
        {i8, i8, R::kGeneral,
            "Matching shapes; output scale=1/256, zero_point=-128; TABLE"},
        {f32, f32, R::kGeneral, "Matching shapes"}});
    add({OP(HardSwish)}, {
        {i8, i8, R::kGeneral,
            "Matching shapes; affine INT8; fixed-point TABLE scales checked"},
        {f32, f32, R::kGeneral, "Matching shapes"}});
    add({OP(Conv2d)}, {
        {i8, i8, R::kGeneral,
            "NHWC; constant symmetric INT8 weights and matching INT32 bias; "
            "per-channel weights; grouped convolution; proven accumulator/rescale "
            "ranges"},
        {f32, f32, R::kGeneral,
            "NHWC; constant weights/bias; no grouped convolution; supported "
            "padding/stride/dilation and fused activation"}});
    add({OP(DepthwiseConv2d)}, {
        {i8, i8, R::kGeneral,
            "NHWC; positive depth multiplier; constant symmetric INT8 weights and "
            "matching INT32 bias; proven ranges"},
        {f32, f32, R::kGeneral,
            "NHWC; positive depth multiplier; constant weights/bias; supported geometry "
            "and activation"}});
    add({OP(FullyConnected)}, {
        {i8, i8, R::kGeneral,
            "Constant dense INT8 weights and matching INT32 bias; default weight "
            "format; keep_num_dims=false; proven convolution-derived ranges"},
        {f32, f32, R::kExtended,
            "Constant dense weights and bias; default format; "
            "keep_num_dims=false; supported fixed dimensions and activation"}});
    add({OP(TransposeConv)}, {
        {i8, i8, R::kExtended,
            "NHWC; constant output shape and symmetric weights; optional matching "
            "INT32 bias; canonical SAME/VALID; stride<=kernel; proven ranges"},
        {f32, f32, R::kExtended,
            "NHWC; constant output shape/weights; optional bias; canonical "
            "SAME/VALID geometry"}}, 2);
    add({OP(Conv3d)}, {{f32, f32, R::kPortable,
        "NDHWC; constant DHWIO weights/bias; SAME/VALID; supported stride/dilation; NONE/RELU/RELU6"}});
    add({OP(AveragePool2d), OP(MaxPool2d)}, {
        {i8, i8, R::kGeneral,
            "NHWC; SAME/VALID; matching input/output quantization; bounded windows; "
            "NONE/RELU/RELU6"},
        {f32, f32, R::kPortable, "NHWC; SAME/VALID; bounded geometry; NONE/RELU/RELU6"}});
    add({OP(Reshape)}, {
        {i8, i8, R::kGeneral,
            "Constant shape; preserved element count and quantization; at most one inferred dimension"},
        {f32, f32, R::kGeneral,
            "Constant shape; preserved element count; at most one inferred dimension"}});
    add({OP(Transpose)}, {
        {i8, i8, R::kGeneral,
            "Constant complete permutation; matching quantization and output shape"},
        {f32, f32, R::kGeneral, "Constant complete permutation; matching output shape"}});
    add({OP(Pad)}, {
        {i8, i8, R::kGeneral,
            "Constant nonnegative padding; matching quantization; real-zero fill"},
        {f32, f32, R::kGeneral, "Constant nonnegative padding; real-zero fill"}});
    add({OP(Padv2)}, {
        {i8, i8, R::kGeneral,
            "Constant nonnegative padding and scalar fill; matching quantization"},
        {f32, f32, R::kPortable,
            "Constant nonnegative padding and finite scalar fill"}});
    add({OP(Slice)}, {
        {i8, i8, R::kGeneral,
            "Constant rank-sized starts/sizes; matching quantization; nonempty bounds"},
        {f32, f32, R::kPortable, "Constant rank-sized starts/sizes; nonempty bounds"}});
    add({OP(Concatenation)}, {
        {i8, i8, R::kGeneral,
            "Compatible non-axis dimensions; matching quantization; no fused activation"},
        {f32, f32, R::kPortable,
            "Compatible non-axis dimensions; no fused activation"}});
    add({OP(Squeeze), OP(ExpandDims), OP(Pack), OP(Unpack), OP(SplitV),
         OP(Tile), OP(ReverseV2)}, {
        {i8, i8, R::kPortable,
            "Constant controls; preserved quantization; nonempty results"},
        {f32, f32, R::kPortable,
            "Constant controls; nonempty results"}});
    add({OP(StridedSlice)}, {
        {i8, i8, R::kPortable,
            "Constant bounds; positive strides; supported masks; preserved quantization; nonempty results"},
        {f32, f32, R::kPortable,
            "Constant bounds; positive strides; supported masks; nonempty results"}});
    add({OP(Split)}, {
        {i8, i8, R::kPortable, "Constant axis; preserved quantization; nonempty split results"},
        {f32, f32, R::kPortable, "Constant axis; nonempty split results"}}, 1);
    add({OP(Maximum), OP(Minimum)}, {
        {i8, i8, R::kPortable, "Static broadcasting; identical input/output quantization"},
        {f32, f32, R::kPortable, "Static broadcasting"}});
    add({OP(Div)}, {{f32, f32, R::kPortable,
        "Static broadcasting; NONE/RELU/RELU6"}});
    add({OP(Pow)}, {{f32, f32, R::kPortable, "Static broadcasting"}});
    add({OP(Sin), OP(Cos), OP(Neg), OP(Square), OP(Sqrt), OP(Rsqrt),
         OP(Exp), OP(Log), OP(Floor), OP(Ceil)}, {
        {f32, f32, R::kPortable, "Matching shapes; FP32 only"}});
    add({OP(Gelu)}, {{f32, f32, R::kPortable,
        "Exact erf or approximate tanh mode; matching shapes"}});
    add({OP(Mean), OP(ReduceProd)}, {{f32, f32, R::kPortable,
        "Constant nonempty axes; keep_dims true/false; non-scalar result"}});
    add({OP(Sum)}, {
        {i8, i8, R::kGeneral,
            "Constant axes; keep_dims=true; proven INT32 accumulation/rescaling"},
        {f32, f32, R::kExtended, "Constant axes; keep_dims true/false; non-scalar result"}});
    add({OP(ReduceMax), OP(ReduceMin)}, {
        {i8, i8, R::kExtended, "Constant axes; preserved quantization; non-scalar result"},
        {f32, f32, R::kExtended, "Constant axes; non-scalar result"}});
    add({OP(Gather)}, {
        {i8, i8, R::kExtended,
            "Constant in-range INT32 indices; batch_dims=0; preserved "
            "quantization"},
        {f32, f32, R::kExtended, "Constant in-range INT32 indices; batch_dims=0"}});
    add({OP(ScatterNd)}, {{f32, f32, R::kExtended,
        "Constant unique in-range indices and output shape; bounded materialization; no duplicate-index accumulation"}}, 1);
    add({OP(ResizeBilinear), OP(ResizeNearestNeighbor)}, {
        {i8, i8, R::kExtended,
            "Static NHWC sizes; preserved quantization; supported "
            "coordinate/border limits; no aligned singleton dimensions"},
        {f32, f32, R::kExtended,
            "Static NHWC sizes; supported coordinate/border limits; no aligned "
            "singleton dimensions"}});
    add({OP(BatchMatmul)}, {{f32, f32, R::kPortable,
        "Static batch broadcasting; transpose flags; bounded batch materialization"}});
    add({OP(Softmax)}, {
        {i8, i8, R::kGeneral,
            "Last axis 1..4095; finite positive scale/beta; output scale=1/256, "
            "zero_point=-128; auto/int8 LUTs or EXT-INT16-gated int16 LUTs"},
        {f32, f32, R::kPortable, "Static last-axis Softmax; finite positive beta"}});
    add({OP(Quantize)}, {
        {i8, i8, R::kGeneral,
            "Identical shapes; per-tensor affine quantization; representable "
            "TABLE/rescale range"},
        {f32, i8, R::kGeneral, "Identical shapes; finite positive per-tensor output scale"}});
    add({OP(Dequantize)}, {{i8, f32, R::kGeneral,
        "Identical shapes; per-tensor affine INT8 input"}});
    add({OP(LogicalAnd), OP(LogicalOr)}, {
        {boolean, boolean, R::kTyped,
            "BOOL data; binary static broadcasting; byte storage at Vulkan "
            "interfaces"}});
    add({OP(LogicalNot)}, {{boolean, boolean, R::kTyped, "Matching BOOL shapes"}});
    add({OP(Equal), OP(NotEqual)}, {
        {boolean, boolean, R::kTyped, "Matching BOOL operands; static broadcasting"},
        {i32, boolean, R::kTyped, "Matching INT32 operands; static broadcasting"},
        {f32, boolean, R::kTyped, "Matching FP32 operands; static broadcasting"}});
    add({OP(ReduceAll), OP(ReduceAny)}, {{boolean, boolean, R::kTyped,
        "Constant axes; keep_dims true/false; non-scalar result"}});
    add({OP(ArgMax)}, {{f32, i32, R::kTyped,
        "One constant INT32 axis; non-scalar result; INT32 indices"}});
    add({OP(BitwiseXor)}, {{i32, i32, R::kTyped, "Static broadcasting"}});
    add({OP(FloorDiv)}, {{i32, i32, R::kTyped,
        "Static broadcasting; positive constant divisors; signed floor correction"}});
    add({OP(RightShift)}, {{u32, u32, R::kTyped,
        "Static broadcasting; constant UINT32 shifts in 0..31"}});
    add({OP(Select), OP(SelectV2)}, {
        {i8, i8, R::kDecomposed, "BOOL condition; static broadcasting; data operands/output share quantization"},
        {f32, f32, R::kDecomposed, "BOOL condition; static broadcasting"},
        {i32, i32, R::kDecomposed, "BOOL condition; static broadcasting"},
        {boolean, boolean, R::kDecomposed, "BOOL condition/data; static broadcasting"}}, 1);
    add({OP(Greater), OP(GreaterEqual), OP(Less), OP(LessEqual)}, {
        {i8, boolean, R::kDecomposed, "Static broadcasting; matching operand quantization"},
        {f32, boolean, R::kDecomposed, "Static broadcasting; matching operand types"},
        {i32, boolean, R::kDecomposed, "Static broadcasting; matching operand types"}});
    add({OP(Prelu), OP(LeakyRelu), OP(ReluN1To1), OP(Relu0To1)}, {
        {i8, i8, R::kDecomposed, "Matching data/output shapes; valid affine scales; PRELU broadcasts quantized alpha; proven rescale ranges"},
        {f32, f32, R::kDecomposed, "Matching data/output shapes; PRELU broadcasts alpha; finite LEAKY_RELU alpha"}});
    auto extend = [&](LiteRtOpCode code, OperationVariant variant) {
      for (auto& entry : result) if (entry.code == code) entry.variants.push_back(variant);
    };
    for (auto code : {OP(Exp), OP(Log), OP(Rsqrt), OP(Gelu), OP(Square)})
      extend(code, {i8, i8, R::kDecomposed,
          "Compile-time mathematical TABLE; LOG/RSQRT input zero_point=-128; GELU exact/approximate"});
    extend(OP(Mean), {i8, i8, R::kDecomposed,
        "Constant nonempty axes; keep_dims true/false; scalar outputs; proven INT32 sum/rescale ranges"});
    extend(OP(BatchMatmul), {i8, i8, R::kDecomposed,
        "Static batch broadcasting; transpose flags; proven INT32 accumulation and native RESCALE"});
    extend(OP(Conv3d), {i8, i8, R::kDecomposed,
        "NDHWC; constant symmetric DHWIO weights/matching INT32 bias; per-output-channel scales; SAME/VALID; proven ranges"});
    add({OP(SpaceToDepth), OP(DepthToSpace), OP(SpaceToBatchNd), OP(BatchToSpaceNd)}, {
        {i8, i8, R::kDecomposed, "Static NHWC; constant block/padding/crop; preserved quantization; bounded geometry"},
        {f32, f32, R::kDecomposed, "Static NHWC; constant block/padding/crop; bounded geometry"},
        {i32, i32, R::kDecomposed, "Static NHWC; constant block/padding/crop; bounded geometry"}});
    add({OP(MirrorPad), OP(BroadcastTo)}, {
        {i8, i8, R::kDecomposed, "Constant controls; preserved quantization; legal reflection/broadcast bounds"},
        {f32, f32, R::kDecomposed, "Constant controls; legal reflection/broadcast bounds"},
        {i32, i32, R::kDecomposed, "Constant controls; legal reflection/broadcast bounds"}});
    for (auto input : {i8, f32, i32, boolean}) {
      for (auto output : {i8, f32, i32, boolean}) {
        bool present = false;
        for (auto& entry : result) if (entry.code == OP(Cast)) {
          entry.variants.push_back({input, output, R::kDecomposed,
              "Raw numeric conversion; float-to-int inputs within INT32 range; integer narrowing preserves low bits"});
          present = true;
        }
        if (!present) add({OP(Cast)}, {{input, output, R::kDecomposed,
            "Raw numeric conversion; float-to-int inputs within INT32 range; integer narrowing preserves low bits"}});
      }
    }
    add({OP(AddN), OP(Sign)}, {
        {f32, f32, R::kDecomposed, "Matching shapes/types; ADD_N has at least two inputs"},
        {i32, i32, R::kDecomposed, "Matching shapes/types; ADD_N has at least two inputs"}});
    add({OP(Round), OP(Elu), OP(LogSoftmax), OP(L2Normalization)}, {
        {f32, f32, R::kDecomposed, "Matching shapes; last-axis normalization; L2 normalization has no fused activation"}});
    add({OP(SquaredDifference)}, {{f32, f32, R::kDecomposed, "Static broadcasting"}});
    add({OP(FloorMod)}, {
        {f32, f32, R::kDecomposed, "Static broadcasting; constant normal power-of-two divisors and reciprocals; bounded constant controls"},
        {i32, i32, R::kDecomposed, "Static broadcasting; strictly positive constant divisors"}});
    extend(OP(FloorDiv), {f32, f32, R::kDecomposed, "Static broadcasting; constant normal power-of-two divisors and reciprocals; bounded constant controls"});
    add({OP(ArgMin)}, {
        {i8, i32, R::kDecomposed, "Constant axis; INT32 indices; preserved first-index tie order"},
        {f32, i32, R::kDecomposed, "Constant axis; INT32 indices; preserved first-index tie order"},
        {i32, i32, R::kDecomposed, "Constant axis; INT32 indices; safe INT32_MIN handling"}});
    for (auto code : {OP(Rank), OP(Shape)}) {
      add({code}, {{i8, i32, R::kDecomposed, "Static shape; INT32 shape metadata"},
          {f32, i32, R::kDecomposed, "Static shape; INT32 shape metadata"},
          {i32, i32, R::kDecomposed, "Static shape; INT32 shape metadata"},
          {boolean, i32, R::kDecomposed, "Static shape; INT32 shape metadata"}});
    }
    add({OP(ZerosLike)}, {{f32, f32, R::kDecomposed, "Static shape"},
        {i32, i32, R::kDecomposed, "Static shape"}});
    add({OP(Fill)}, {{f32, f32, R::kDecomposed, "Scalar value and constant positive output dimensions"},
        {i32, i32, R::kDecomposed, "Scalar value and constant positive output dimensions"}}, 1);
    add({OP(GatherNd)}, {
        {i8, i8, R::kDecomposed, "Constant in-range INT32 indices; preserved quantization; bounded materialization"},
        {f32, f32, R::kDecomposed, "Constant in-range INT32 indices; bounded materialization"},
        {i32, i32, R::kDecomposed, "Constant in-range INT32 indices; bounded materialization"}});
    add({OP(OneHot)}, {
        {i32, i8, R::kDecomposed, "Constant bounded depth and on/off scalars; INT32 indices; out-of-range indices produce off values"},
        {i32, f32, R::kDecomposed, "Constant bounded depth and on/off scalars; INT32 indices"},
        {i32, i32, R::kDecomposed, "Constant bounded depth and on/off scalars; INT32 indices"},
        {i32, boolean, R::kDecomposed, "Constant bounded depth and on/off BOOL scalars; INT32 indices"}});
    add({OP(SparseToDense)}, {
        {f32, f32, R::kDecomposed, "Constant unique in-range INT32 indices, output shape and default; scalar or per-index values; bounded output"},
        {i32, i32, R::kDecomposed, "Constant unique in-range INT32 indices, output shape and default; scalar or per-index values; bounded output"}}, 2);
    add({OP(Bucketize)}, {
        {f32, i32, R::kDecomposed, "At most 256 finite sorted boundaries; static shape"},
        {i32, i32, R::kDecomposed, "At most 256 finite sorted boundaries; internal FP32 comparisons require PRO-FP"}});
    add({OP(FakeQuant)}, {{f32, f32, R::kDecomposed, "Finite ordered range; 2..16 bits; narrow range supported"}});
    add({OP(Atan2)}, {{f32, f32, R::kDecomposed, "Matching operand shapes; PRO-FP range reduction and polynomial"}});
    add({OP(Rfft2d)}, {{f32, c64, R::kDecomposed,
        "Constant power-of-two FFT lengths; input cropped/zero-padded; interleaved COMPLEX64 storage; EXT-FFT required"}});
    add({OP(Real), OP(Imag)}, {{c64, f32, R::kDecomposed,
        "Static COMPLEX64 input; interleaved FP32 real/imaginary storage; scalar outputs supported"}});
#undef OP
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
      return a.code < b.code;
    });
    return result;
  }();
  return registrations;
}

const OperationRegistration* FindOperationRegistration(LiteRtOpCode code) {
  const auto& registrations = GetOperationRegistrations();
  auto found = std::lower_bound(registrations.begin(), registrations.end(), code,
      [](const auto& registration, LiteRtOpCode value) {
        return registration.code < value;
      });
  return found != registrations.end() && found->code == code ? &*found : nullptr;
}

const OperationVariant* FindOperationVariant(
    const OperationRegistration& registration, LiteRtElementType input_type,
    LiteRtElementType output_type) {
  for (const auto& variant : registration.variants) {
    if (variant.input_type == input_type && variant.output_type == output_type)
      return &variant;
  }
  return nullptr;
}

const char* LoweringRouteName(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kGeneral: return "general";
    case LoweringRoute::kPortable: return "portable";
    case LoweringRoute::kTyped: return "typed";
    case LoweringRoute::kExtended: return "extended";
    case LoweringRoute::kDecomposed: return "decomposed";
  }
  return "invalid";
}

const char* RegisteredElementTypeName(LiteRtElementType type) {
  switch (type) {
    case kLiteRtElementTypeInt8: return "INT8";
    case kLiteRtElementTypeFloat32: return "FLOAT32";
    case kLiteRtElementTypeInt32: return "INT32";
    case kLiteRtElementTypeUInt32: return "UINT32";
    case kLiteRtElementTypeBool: return "BOOL";
    case kLiteRtElementTypeComplex64: return "COMPLEX64";
    default: return "UNREGISTERED";
  }
}

}  // namespace litert::arm_vulkan_ml
