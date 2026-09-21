/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "node_binary_info.h"

namespace {

LiteRtStatus SetI(AmlNode node, const char* key, int value) {
  if (AmlNodeSetParamI(node, key, value) != kAmlCompilerStatusOk) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus SetF(AmlNode node, const char* key, float value) {
  if (AmlNodeSetParamF(node, key, value) != kAmlCompilerStatusOk) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus ParseOpParams(AmlNode node, int op_code, LiteRtOp c_op) {
  switch (op_code) {
    case kLiteRtOpCodeTflAdd: {
      litert::AddOptions opts;
      opts.InitFromOp(c_op);
      LiteRtStatus st =
          SetI(node, "activation",
               static_cast<int>(opts.fused_activation_function));
      if (st != kLiteRtStatusOk) {
        return st;
      }
      bool pot_scale_int16 = false;
      if (LiteRtGetAddPotScaleInt16Option(c_op, &pot_scale_int16) !=
          kLiteRtStatusOk) {
        pot_scale_int16 = false;
      }
      return SetI(node, "pot_scale_int16", static_cast<int>(pot_scale_int16));
    }

    case kLiteRtOpCodeTflConcatenation: {
      litert::ConcatenationOptions opts;
      opts.InitFromOp(c_op);
      LiteRtStatus st =
          SetI(node, "activation",
               static_cast<int>(opts.fused_activation_function));
      if (st != kLiteRtStatusOk) {
        return st;
      }
      return SetI(node, "axis", static_cast<int>(opts.axis));
    }

    case kLiteRtOpCodeTflDiv: {
      litert::DivOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflFullyConnected: {
      litert::FullyConnectedOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "activation",
               static_cast<int>(opts.fused_activation_function)) !=
              kLiteRtStatusOk ||
          SetI(node, "keep_num_dims", static_cast<int>(opts.keep_num_dims)) !=
              kLiteRtStatusOk ||
          SetI(node, "asymmetric_quantize_input",
               static_cast<int>(opts.asymmetric_quantize_input)) !=
              kLiteRtStatusOk ||
          SetI(node, "quantized_bias_type",
               static_cast<int>(opts.quantized_bias_type)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "weights_format",
                  static_cast<int>(opts.weights_format));
    }

    case kLiteRtOpCodeTflMul: {
      litert::MulOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflSoftmax: {
      litert::SoftmaxOptions opts;
      opts.InitFromOp(c_op);
      return SetF(node, "beta", static_cast<float>(opts.beta));
    }

    case kLiteRtOpCodeTflStridedSlice: {
      litert::StridedSliceOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "begin_mask", static_cast<int>(opts.begin_mask)) !=
              kLiteRtStatusOk ||
          SetI(node, "end_mask", static_cast<int>(opts.end_mask)) !=
              kLiteRtStatusOk ||
          SetI(node, "ellipsis_mask", static_cast<int>(opts.ellipsis_mask)) !=
              kLiteRtStatusOk ||
          SetI(node, "new_axis_mask", static_cast<int>(opts.new_axis_mask)) !=
              kLiteRtStatusOk ||
          SetI(node, "shrink_axis_mask",
               static_cast<int>(opts.shrink_axis_mask)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "offset", static_cast<int>(opts.offset));
    }

    case kLiteRtOpCodeTflSub: {
      litert::SubOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "activation",
               static_cast<int>(opts.fused_activation_function)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      bool pot_scale_int16 = false;
      if (LiteRtGetSubPotScaleInt16Option(c_op, &pot_scale_int16) !=
          kLiteRtStatusOk) {
        pot_scale_int16 = false;
      }
      return SetI(node, "pot_scale_int16", static_cast<int>(pot_scale_int16));
    }

    case kLiteRtOpCodeTflSum: {
      litert::SumOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceMax: {
      litert::ReduceMaxOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceMin: {
      litert::ReduceMinOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceAny: {
      litert::ReduceAnyOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceAll: {
      litert::ReduceAllOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflPack: {
      litert::PackOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "axis", static_cast<int>(opts.axis)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      int32_t values_count = 0;
      if (LiteRtGetPackValuesCountOption(c_op, &values_count) ==
          kLiteRtStatusOk) {
        return SetI(node, "values_count", static_cast<int>(values_count));
      }
      return kLiteRtStatusOk;
    }

    case kLiteRtOpCodeTflUnpack: {
      litert::UnpackOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "axis", static_cast<int>(opts.axis)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "num", static_cast<int>(opts.num));
    }

    case kLiteRtOpCodeTflGather: {
      litert::GatherOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "axis", static_cast<int>(opts.axis)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "batch_dims", static_cast<int>(opts.batch_dims));
    }

    case kLiteRtOpCodeTflMean: {
      litert::MeanOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflConv2d: {
      litert::Conv2dOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "padding", static_cast<int>(opts.padding)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_w", static_cast<int>(opts.stride_w)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_h", static_cast<int>(opts.stride_h)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_w", static_cast<int>(opts.dilation_w_factor)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_h", static_cast<int>(opts.dilation_h_factor)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflConv3d: {
      litert::Conv3dOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "padding", static_cast<int>(opts.padding)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_w", static_cast<int>(opts.stride_w)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_h", static_cast<int>(opts.stride_h)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_d", static_cast<int>(opts.stride_d)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_w", static_cast<int>(opts.dilation_w_factor)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_h", static_cast<int>(opts.dilation_h_factor)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_d", static_cast<int>(opts.dilation_d_factor)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflAveragePool2d: {
      litert::AveragePool2dOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "padding", static_cast<int>(opts.padding)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_w", static_cast<int>(opts.stride_w)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_h", static_cast<int>(opts.stride_h)) !=
              kLiteRtStatusOk ||
          SetI(node, "filter_w", static_cast<int>(opts.filter_width)) !=
              kLiteRtStatusOk ||
          SetI(node, "filter_h", static_cast<int>(opts.filter_height)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflDepthwiseConv2d: {
      litert::DepthwiseConv2dOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "padding", static_cast<int>(opts.padding)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_w", static_cast<int>(opts.stride_w)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_h", static_cast<int>(opts.stride_h)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_w", static_cast<int>(opts.dilation_w_factor)) !=
              kLiteRtStatusOk ||
          SetI(node, "dilation_h", static_cast<int>(opts.dilation_h_factor)) !=
              kLiteRtStatusOk ||
          SetI(node, "multiplier", static_cast<int>(opts.depth_multiplier)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflMaxPool2d: {
      litert::MaxPool2dOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "padding", static_cast<int>(opts.padding)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_w", static_cast<int>(opts.stride_w)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_h", static_cast<int>(opts.stride_h)) !=
              kLiteRtStatusOk ||
          SetI(node, "filter_w", static_cast<int>(opts.filter_width)) !=
              kLiteRtStatusOk ||
          SetI(node, "filter_h", static_cast<int>(opts.filter_height)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflResizeBilinear: {
      litert::ResizeBilinearOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "align_corners", static_cast<int>(opts.align_corners)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "half_pixel_centers",
                  static_cast<int>(opts.half_pixel_centers));
    }

    case kLiteRtOpCodeTflLeakyRelu: {
      litert::LeakyReluOptions opts;
      opts.InitFromOp(c_op);
      return SetF(node, "alpha", static_cast<float>(opts.alpha));
    }

    case kLiteRtOpCodeTflSpaceToDepth: {
      litert::SpaceToDepthOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "block_size", static_cast<int>(opts.block_size));
    }

    case kLiteRtOpCodeTflDepthToSpace: {
      litert::DepthToSpaceOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "block_size", static_cast<int>(opts.block_size));
    }

    case kLiteRtOpCodeTflResizeNearestNeighbor: {
      litert::ResizeNearestNeighborOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "align_corners", static_cast<int>(opts.align_corners)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "half_pixel_centers",
                  static_cast<int>(opts.half_pixel_centers));
    }

    case kLiteRtOpCodeTflCumsum: {
      litert::CumSumOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "exclusive", static_cast<int>(opts.exclusive)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "reverse", static_cast<int>(opts.reverse));
    }

    case kLiteRtOpCodeTflGelu: {
      litert::GeluOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "approximate", static_cast<int>(opts.approximate));
    }

    case kLiteRtOpCodeTflMirrorPad: {
      litert::MirrorPadOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "mode", static_cast<int>(opts.mode));
    }

    case kLiteRtOpCodeTflSplit: {
      litert::SplitOptions opts;
      opts.InitFromOp(c_op);
      return SetI(node, "num_split", static_cast<int>(opts.num_splits));
    }

    case kLiteRtOpCodeTflTransposeConv: {
      litert::TransposeConvOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "padding", static_cast<int>(opts.padding)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_w", static_cast<int>(opts.stride_w)) !=
              kLiteRtStatusOk ||
          SetI(node, "stride_h", static_cast<int>(opts.stride_h)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflBatchMatmul: {
      litert::BatchMatmulOptions opts;
      opts.InitFromOp(c_op);
      if (SetI(node, "adj_x", static_cast<int>(opts.adj_x)) !=
              kLiteRtStatusOk ||
          SetI(node, "adj_y", static_cast<int>(opts.adj_y)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs",
                  static_cast<int>(opts.asymmetric_quantize_input));
    }

    case kLiteRtOpCodeTflDynamicUpdateSlice:
    case kLiteRtOpCodeShloDynamicUpdateSlice:
    case kLiteRtOpCodeTflReshape:
    default:
      break;
  }

  return kLiteRtStatusOk;
}
