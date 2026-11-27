// Copyright 2026 Amlogic, Inc.
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

#include "node_binary_info.h"

#include <cstdio>

#include "litert/core/model/model.h"
#include "litert/vendors/amlogic/core/aml_compiler_core_loader.h"

namespace {

LiteRtStatus InitOpts(litert::OpOptions& opts, LiteRtOp op) {
  return opts.InitFromOp(op);
}

LiteRtStatus SetI(AmlNode node, const char* key, int value) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr ||
      api->AmlNodeSetParamI(node, key, value) != kAmlCompilerStatusOk) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus SetF(AmlNode node, const char* key, float value) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr ||
      api->AmlNodeSetParamF(node, key, value) != kAmlCompilerStatusOk) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus ParseOpParams(AmlNode node, int op_code, LiteRtOp c_op) {
  switch (op_code) {
    case kLiteRtOpCodeTflAdd: {
      litert::AddOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      LiteRtStatus st =
          SetI(node, "activation",
               static_cast<int>(opts.fused_activation_function));
      if (st != kLiteRtStatusOk) {
        return st;
      }
      // Public op-options API no longer exposes PotScaleInt16.
      return SetI(node, "pot_scale_int16", 0);
    }

    case kLiteRtOpCodeTflConcatenation: {
      litert::ConcatenationOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflFullyConnected: {
      litert::FullyConnectedOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "activation",
                  static_cast<int>(opts.fused_activation_function));
    }

    case kLiteRtOpCodeTflSoftmax: {
      litert::SoftmaxOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetF(node, "beta", static_cast<float>(opts.beta));
    }

    case kLiteRtOpCodeTflStridedSlice: {
      litert::StridedSliceOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "activation",
               static_cast<int>(opts.fused_activation_function)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      // Public op-options API no longer exposes PotScaleInt16.
      return SetI(node, "pot_scale_int16", 0);
    }

    case kLiteRtOpCodeTflSum: {
      litert::SumOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceMax: {
      litert::ReduceMaxOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceMin: {
      litert::ReduceMinOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceAny: {
      litert::ReduceAnyOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflReduceAll: {
      litert::ReduceAllOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflPack: {
      litert::PackOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "axis", static_cast<int>(opts.axis)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "num", static_cast<int>(opts.num));
    }

    case kLiteRtOpCodeTflGather: {
      litert::GatherOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "axis", static_cast<int>(opts.axis)) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "batch_dims", static_cast<int>(opts.batch_dims));
    }

    case kLiteRtOpCodeTflMean: {
      litert::MeanOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "keep_dims", static_cast<int>(opts.keep_dims));
    }

    case kLiteRtOpCodeTflConv2d: {
      litert::Conv2dOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "align_corners", static_cast<int>(opts.align_corners)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "half_pixel_centers",
                  static_cast<int>(opts.half_pixel_centers));
    }

    case kLiteRtOpCodeTflLeakyRelu: {
      litert::LeakyReluOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetF(node, "alpha", static_cast<float>(opts.alpha));
    }

    case kLiteRtOpCodeTflSpaceToDepth: {
      litert::SpaceToDepthOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "block_size", static_cast<int>(opts.block_size));
    }

    case kLiteRtOpCodeTflDepthToSpace: {
      litert::DepthToSpaceOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "block_size", static_cast<int>(opts.block_size));
    }

    case kLiteRtOpCodeTflResizeNearestNeighbor: {
      litert::ResizeNearestNeighborOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "align_corners", static_cast<int>(opts.align_corners)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "half_pixel_centers",
                  static_cast<int>(opts.half_pixel_centers));
    }

    case kLiteRtOpCodeTflCumsum: {
      litert::CumSumOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "exclusive", static_cast<int>(opts.exclusive)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "reverse", static_cast<int>(opts.reverse));
    }

    case kLiteRtOpCodeTflGelu: {
      litert::GeluOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "approximate", static_cast<int>(opts.approximate));
    }

    case kLiteRtOpCodeTflMirrorPad: {
      litert::MirrorPadOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "mode", static_cast<int>(opts.mode));
    }

    case kLiteRtOpCodeTflSplit: {
      litert::SplitOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "num_split", static_cast<int>(opts.num_splits));
    }

    case kLiteRtOpCodeTflTransposeConv: {
      litert::TransposeConvOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "adj_x", static_cast<int>(opts.adj_x)) !=
              kLiteRtStatusOk ||
          SetI(node, "adj_y", static_cast<int>(opts.adj_y)) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs",
                  static_cast<int>(opts.asymmetric_quantize_input));
    }

    case kLiteRtOpCodeTflL2Pool2d: {
      litert::L2Pool2dOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
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

    case kLiteRtOpCodeTflSqueeze: {
      litert::SqueezeOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      if (SetI(node, "num", static_cast<int>(opts.squeeze_dims.size())) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      for (size_t i = 0; i < opts.squeeze_dims.size(); ++i) {
        char key[32];
        std::snprintf(key, sizeof(key), "squeeze_dim_%zu", i);
        if (SetI(node, key, static_cast<int>(opts.squeeze_dims[i])) !=
            kLiteRtStatusOk) {
          return kLiteRtStatusErrorRuntimeFailure;
        }
      }
      return kLiteRtStatusOk;
    }

    case kLiteRtOpCodeTflReduceProd: {
      const auto* reducer =
          litert::internal::GetTflOptions(*c_op).AsReducerOptions();
      const bool keep_dims = reducer != nullptr && reducer->keep_dims;
      return SetI(node, "keep_dims", keep_dims ? 1 : 0);
    }

    case kLiteRtOpCodeTflArgMax: {
      const auto* arg_max =
          litert::internal::GetTflOptions(*c_op).AsArgMaxOptions();
      const int output_type =
          arg_max != nullptr ? static_cast<int>(arg_max->output_type) : 0;
      return SetI(node, "output_type", output_type);
    }

    case kLiteRtOpCodeTflArgMin: {
      const auto* arg_min =
          litert::internal::GetTflOptions(*c_op).AsArgMinOptions();
      const int output_type =
          arg_min != nullptr ? static_cast<int>(arg_min->output_type) : 0;
      return SetI(node, "output_type", output_type);
    }

    case kLiteRtOpCodeTflLstm: {
      const auto* lstm = litert::internal::GetTflOptions(*c_op).AsLSTMOptions();
      const int activation =
          lstm != nullptr ? static_cast<int>(lstm->fused_activation_function)
                          : 0;
      const float cell_clip = lstm != nullptr ? lstm->cell_clip : 0.0f;
      const float proj_clip = lstm != nullptr ? lstm->proj_clip : 0.0f;
      const int asymmetric =
          lstm != nullptr && lstm->asymmetric_quantize_inputs ? 1 : 0;
      if (SetI(node, "activation", activation) != kLiteRtStatusOk ||
          SetF(node, "cell_clip", cell_clip) != kLiteRtStatusOk ||
          SetF(node, "proj_clip", proj_clip) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs", asymmetric);
    }

    case kLiteRtOpCodeTflRnn: {
      const auto* rnn = litert::internal::GetTflOptions(*c_op).AsRNNOptions();
      const int activation =
          rnn != nullptr ? static_cast<int>(rnn->fused_activation_function) : 0;
      const int asymmetric =
          rnn != nullptr && rnn->asymmetric_quantize_inputs ? 1 : 0;
      if (SetI(node, "activation", activation) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs", asymmetric);
    }

    case kLiteRtOpCodeTflUnidirectionalSequenceRnn: {
      const auto* rnn =
          litert::internal::GetTflOptions(*c_op).AsSequenceRNNOptions();
      const int activation =
          rnn != nullptr ? static_cast<int>(rnn->fused_activation_function) : 0;
      const int time_major = rnn != nullptr && rnn->time_major ? 1 : 0;
      const int asymmetric =
          rnn != nullptr && rnn->asymmetric_quantize_inputs ? 1 : 0;
      if (SetI(node, "activation", activation) != kLiteRtStatusOk ||
          SetI(node, "time_major", time_major) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs", asymmetric);
    }

    case kLiteRtOpCodeTflBidirectionalSequenceRnn: {
      const auto* rnn = litert::internal::GetTflOptions(*c_op)
                            .AsBidirectionalSequenceRNNOptions();
      const int activation =
          rnn != nullptr ? static_cast<int>(rnn->fused_activation_function) : 0;
      const int time_major = rnn != nullptr && rnn->time_major ? 1 : 0;
      const int merge_outputs = rnn != nullptr && rnn->merge_outputs ? 1 : 0;
      const int asymmetric =
          rnn != nullptr && rnn->asymmetric_quantize_inputs ? 1 : 0;
      if (SetI(node, "activation", activation) != kLiteRtStatusOk ||
          SetI(node, "time_major", time_major) != kLiteRtStatusOk ||
          SetI(node, "merge_outputs", merge_outputs) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs", asymmetric);
    }

    case kLiteRtOpCodeTflUnidirectionalSequenceLstm: {
      const auto* lstm = litert::internal::GetTflOptions(*c_op)
                             .AsUnidirectionalSequenceLSTMOptions();
      const int activation =
          lstm != nullptr ? static_cast<int>(lstm->fused_activation_function)
                          : 0;
      const float cell_clip = lstm != nullptr ? lstm->cell_clip : 0.0f;
      const float proj_clip = lstm != nullptr ? lstm->proj_clip : 0.0f;
      const int time_major = lstm != nullptr && lstm->time_major ? 1 : 0;
      const int asymmetric =
          lstm != nullptr && lstm->asymmetric_quantize_inputs ? 1 : 0;
      if (SetI(node, "activation", activation) != kLiteRtStatusOk ||
          SetF(node, "cell_clip", cell_clip) != kLiteRtStatusOk ||
          SetF(node, "proj_clip", proj_clip) != kLiteRtStatusOk ||
          SetI(node, "time_major", time_major) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs", asymmetric);
    }

    case kLiteRtOpCodeTflBidirectionalSequenceLstm: {
      const auto* lstm = litert::internal::GetTflOptions(*c_op)
                             .AsBidirectionalSequenceLSTMOptions();
      const int activation =
          lstm != nullptr ? static_cast<int>(lstm->fused_activation_function)
                          : 0;
      const float cell_clip = lstm != nullptr ? lstm->cell_clip : 0.0f;
      const float proj_clip = lstm != nullptr ? lstm->proj_clip : 0.0f;
      const int time_major = lstm != nullptr && lstm->time_major ? 1 : 0;
      const int merge_outputs = lstm != nullptr && lstm->merge_outputs ? 1 : 0;
      const int asymmetric =
          lstm != nullptr && lstm->asymmetric_quantize_inputs ? 1 : 0;
      if (SetI(node, "activation", activation) != kLiteRtStatusOk ||
          SetF(node, "cell_clip", cell_clip) != kLiteRtStatusOk ||
          SetF(node, "proj_clip", proj_clip) != kLiteRtStatusOk ||
          SetI(node, "time_major", time_major) != kLiteRtStatusOk ||
          SetI(node, "merge_outputs", merge_outputs) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "asymmetric_quantize_inputs", asymmetric);
    }

    case kLiteRtOpCodeTflLocalResponseNormalization: {
      int32_t radius = 0;
      float bias = 0.f;
      float alpha = 0.f;
      float beta = 0.f;
      if (LiteRtGetLocalResponseNormalizationRadiusOption(c_op, &radius) !=
              kLiteRtStatusOk ||
          LiteRtGetLocalResponseNormalizationBiasOption(c_op, &bias) !=
              kLiteRtStatusOk ||
          LiteRtGetLocalResponseNormalizationAlphaOption(c_op, &alpha) !=
              kLiteRtStatusOk ||
          LiteRtGetLocalResponseNormalizationBetaOption(c_op, &beta) !=
              kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      if (SetI(node, "radius", radius) != kLiteRtStatusOk ||
          SetF(node, "bias", bias) != kLiteRtStatusOk ||
          SetF(node, "alpha", alpha) != kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetF(node, "beta", beta);
    }

    case kLiteRtOpCodeTflL2Normalization: {
      const auto* opts =
          litert::internal::GetTflOptions(*c_op).AsL2NormOptions();
      const int activation =
          opts != nullptr ? static_cast<int>(opts->fused_activation_function)
                          : 0;
      return SetI(node, "activation", activation);
    }

    case kLiteRtOpCodeTflShape: {
      const auto* opts =
          litert::internal::GetTflOptions(*c_op).AsShapeOptions();
      const int out_type =
          opts != nullptr ? static_cast<int>(opts->out_type)
                          : static_cast<int>(tflite::TensorType_INT32);
      return SetI(node, "out_type", out_type);
    }

    case kLiteRtOpCodeTflUnique: {
      const auto* opts =
          litert::internal::GetTflOptions(*c_op).AsUniqueOptions();
      const int idx_out_type =
          opts != nullptr ? static_cast<int>(opts->idx_out_type)
                          : static_cast<int>(tflite::TensorType_INT32);
      return SetI(node, "idx_out_type", idx_out_type);
    }

    case kLiteRtOpCodeTflOneHot: {
      litert::OneHotOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "axis", static_cast<int>(opts.axis));
    }

    case kLiteRtOpCodeTflCast: {
      const auto* opts =
          litert::internal::GetTflOptions(*c_op).AsCastOptions();
      if (opts == nullptr) {
        return kLiteRtStatusOk;
      }
      if (SetI(node, "in_data_type", static_cast<int>(opts->in_data_type)) !=
          kLiteRtStatusOk) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return SetI(node, "out_data_type", static_cast<int>(opts->out_data_type));
    }

    case kLiteRtOpCodeTflSplitV: {
      litert::SplitVOptions opts;
      if (const LiteRtStatus init_status = InitOpts(opts, c_op);
          init_status != kLiteRtStatusOk) {
        return init_status;
      }
      return SetI(node, "num_splits", static_cast<int>(opts.num_splits));
    }

    // Builtin options carry no scalar fields. TOPK k, GATHER_ND indices, and
    // NMS boxes/scores/iou are tensors, so there is nothing to SetParam.
    case kLiteRtOpCodeTflTopkV2:
    case kLiteRtOpCodeTflGatherNd:
    case kLiteRtOpCodeTflHardSwish:
    case kLiteRtOpCodeTflLogSoftmax:
    case kLiteRtOpCodeTflNonMaxSuppressionV4:
    case kLiteRtOpCodeTflNonMaxSuppressionV5:
      return kLiteRtStatusOk;

    case kLiteRtOpCodeTflDynamicUpdateSlice:
    case kLiteRtOpCodeShloDynamicUpdateSlice:
    case kLiteRtOpCodeTflReshape:
    default:
      break;
  }

  return kLiteRtStatusOk;
}
