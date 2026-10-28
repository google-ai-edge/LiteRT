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

#include "ml_drift_delegate/delegate/composite/tiled_quantized_fully_connected_kernel.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected_util.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/compiler_options.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/util.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

class TiledQuantizedFullyConnected : public ::ml_drift::GPUOperation {
 public:
  TiledQuantizedFullyConnected() = default;
  TiledQuantizedFullyConnected(TiledQuantizedFullyConnected&&) = default;
  TiledQuantizedFullyConnected& operator=(TiledQuantizedFullyConnected&&) =
      default;
  TiledQuantizedFullyConnected(const TiledQuantizedFullyConnected&) = delete;
  TiledQuantizedFullyConnected& operator=(const TiledQuantizedFullyConnected&) =
      delete;

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  ::ml_drift::int3 GetGridSize() const override {
    const int total_spatial =
        dst_[0]->Batch() * dst_[0]->Height() * dst_[0]->Width();
    const int spatial_tiles =
        ::ml_drift::DivideRoundUp(total_spatial, tile_w_);
    return ::ml_drift::int3(dst_[0]->Slices(),
                            work_group_size_.y * spatial_tiles, 1);
  }

  static std::unique_ptr<TiledQuantizedFullyConnected> Create(
      const ::ml_drift::GpuInfo& gpu_info,
      ::ml_drift::CalculationsPrecision precision,
      const ::ml_drift::TensorDescriptor& src,
      const ::ml_drift::TensorDescriptor& dst,
      const ::ml_drift::ExternalWeights& weights,
      const ::ml_drift::TensorDescriptor* bias,
      const ::ml_drift::BHWC& dst_shape) {
    auto op = std::make_unique<TiledQuantizedFullyConnected>();
    const auto& weights_desc = weights.desc;
    const auto& weights_shape = weights.shape;
    const ::ml_drift::DataType weights_type =
        ::ml_drift::fc::GetDataTypeForWeights(weights_desc.type);
    const ::ml_drift::OHWI scale_zp_shape =
        weights.scale ? weights.scale_zp_shape : ::ml_drift::OHWI(1, 1, 1, 1);
    const bool has_bias = bias != nullptr;
    const bool has_zero_point =
        weights.zero_point != nullptr || weights.scalar_zero_point.has_value();

    const int total_spatial = dst_shape.b * dst_shape.h * dst_shape.w;
    int tile_w = 8;
    const char* tile_env = std::getenv("LRT_FC_TILE_W");
    int parsed = 0;
    if (tile_env != nullptr && absl::SimpleAtoi(tile_env, &parsed) &&
        parsed >= 1 && parsed <= 16) {
      tile_w = parsed;
    }
    tile_w = std::min(tile_w, total_spatial);
    op->tile_w_ = tile_w;

    op->AddSrcTensor("src_tensor", src);
    op->AddDstTensor("dst_tensor", dst);
    op->args_.AddInt("o_group_size", weights_desc.GetOutputGroupSize());
    const int dst_slices = ::ml_drift::DivideRoundUp(weights_shape.o, 4);
    const int src_slices = ::ml_drift::DivideRoundUp(weights_shape.i, 4);
    op->args_.AddInt(
        "o_groups",
        ::ml_drift::DivideRoundUp(dst_slices,
                                  weights_desc.GetOutputGroupSize()));
    if (scale_zp_shape.i != 1) {
      const int src_groups = scale_zp_shape.i;
      op->args_.AddInt("src_groups", src_groups);
      op->args_.AddInt("src_group_size", src_slices / src_groups);
    }

    const int spatial_tiles = ::ml_drift::DivideRoundUp(total_spatial, tile_w);
    const int max_y_size =
        scale_zp_shape.i != 1 ? scale_zp_shape.i : src_slices;
    op->work_group_size_ =
        SelectTiledWorkGroupSize(gpu_info, dst_slices, max_y_size,
                                 spatial_tiles, GetEnvInt("LRT_FC_WG_Y"));
    const bool wg_reduction = op->work_group_size_.y != 1;

    // Launch the spatial tiles of one block of output slices back to back, so
    // the INT4 weights of that block are fetched from DRAM once and then hit in
    // the GPU L2 for the remaining tiles (the same weight reuse that ML Drift's
    // conv_wave_memory gets from processing a whole spatial block per weight
    // load). `LRT_FC_LAUNCH_ORDER=dst` restores the output-slice-major order.
    const char* launch_order_env = std::getenv("LRT_FC_LAUNCH_ORDER");
    const bool spatial_major =
        launch_order_env == nullptr ||
        std::string_view(launch_order_env) != "dst";
    op->work_group_launch_order_ =
        spatial_major ? ::ml_drift::int3(1, 0, 2) : ::ml_drift::int3(0, 1, 2);

    op->code_ = GenerateKernelCode(
        src, dst, dst_shape, precision, gpu_info, weights_desc, weights_type,
        scale_zp_shape, has_bias, has_zero_point, tile_w, wg_reduction,
        spatial_major, op->work_group_size_);

    if (gpu_info.IsMali() || gpu_info.IsAdreno() || gpu_info.IsMaleoon() ||
        gpu_info.IsPowerVR() ||
        (precision == ::ml_drift::CalculationsPrecision::kF16 &&
         gpu_info.IsIntel())) {
      op->compiler_options_.push_back(
          ::ml_drift::CompilerOptions::kClFastRelaxedMath);
    }

    ::ml_drift::fc::AddWeightsArguments(weights, /*vec_size=*/16, op.get());
    if (bias) {
      op->AddSrcTensor("biases", *bias);
    }
    return op;
  }

 private:
  // Returns the integer value of environment variable `name`, or 0 if unset.
  static int GetEnvInt(const char* name) {
    const char* value = std::getenv(name);
    int parsed = 0;
    if (value == nullptr || !absl::SimpleAtoi(value, &parsed)) return 0;
    return parsed;
  }

  // Selects the work group of the spatially tiled kernel.
  //
  // The decode FullyConnected kernel (`fc1x1_int*_weights`) processes a single
  // token, so all of its parallelism has to come from splitting the input
  // channel reduction across WG_SIZE_Y threads (64x4 or 16x16 on Adreno), which
  // costs a `tile_w x WG_SIZE_X x WG_SIZE_Y` __local reduction buffer and
  // log2(WG_SIZE_Y) barriers. Here every spatial tile already contributes
  // `dst_slices` independent output slices, so the reduction is only split
  // when `dst_slices * spatial_tiles` alone cannot fill the GPU (e.g. the
  // 1024-wide down/out projections of a short prefill). Otherwise
  // WG_SIZE_Y = 1: no __local memory, no barriers, one output slice and
  // `tile_w` tokens per thread over the full input channel range.
  // `wg_y_override` (LRT_FC_WG_Y) forces WG_SIZE_Y for experiments.
  static ::ml_drift::int3 SelectTiledWorkGroupSize(
      const ::ml_drift::GpuInfo& gpu_info, int dst_slices, int max_y_size,
      int spatial_tiles, int wg_y_override) {
    int max_wg_size = 256;
    if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsLowEnd()) {
      max_wg_size = 32;
    }
    constexpr int kMaxYSize = 16;
    const int y_limit =
        std::max(1, std::min({max_y_size, kMaxYSize, max_wg_size}));
    // Enough threads for several resident waves on every compute unit.
    const int target_threads =
        std::max(1, gpu_info.GetComputeUnitsCount()) * 1024;
    const int independent_tasks = dst_slices * spatial_tiles;
    int y_size = 1;
    if (wg_y_override >= 1) {
      while (y_size * 2 <= std::min(wg_y_override, y_limit)) {
        y_size *= 2;
      }
    } else {
      while (independent_tasks * y_size < target_threads &&
             y_size * 2 <= y_limit) {
        y_size *= 2;
      }
    }
    int x_size = std::max(1, std::min(64, max_wg_size / y_size));
    while (x_size > 8 && x_size / 2 >= dst_slices) {
      x_size /= 2;
    }
    return ::ml_drift::int3(x_size, y_size, 1);
  }

  static std::string GenerateReadWeightsCode(
      ::ml_drift::DataType weights_type,
      const ::ml_drift::WeightsDescription& weights_desc) {
    std::string c;
    if (weights_desc.IsLinearLayout()) {
      c += "    int linear_i4o4 = src_s * args.dst_tensor.Slices() + dst_s;\n";
      if (weights_type == ::ml_drift::DataType::kInt2) {
        c += "    uint w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      } else if (weights_type == ::ml_drift::DataType::kInt4) {
        c += "    uint2 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      } else {
        c += "    uint4 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    } else {
      if (weights_type == ::ml_drift::DataType::kInt8) {
        c += "    uint4 w = args.weights.Read(dst_s, src_s);\n";
        c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      } else if (weights_type == ::ml_drift::DataType::kInt4) {
        c += "    ushort4 w = args.weights.Read(dst_s, src_s);\n";
        c += "    ucl::U16x4ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      } else if (weights_type == ::ml_drift::DataType::kInt2) {
        c += "    uchar4 w = args.weights.Read(dst_s, src_s);\n";
        c += "    ucl::U8x4ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    }
    return c;
  }

  static std::string GenerateKernelCode(
      const ::ml_drift::TensorDescriptor& src,
      const ::ml_drift::TensorDescriptor& dst,
      const ::ml_drift::BHWC& dst_shape,
      ::ml_drift::CalculationsPrecision precision,
      const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::WeightsDescription& weights_desc,
      ::ml_drift::DataType weights_type, const ::ml_drift::OHWI& scale_zp_shape,
      bool has_bias, bool has_zero_point, int tile_w, bool wg_reduction,
      bool spatial_major, const ::ml_drift::int3& work_group_size) {
    const ::ml_drift::DataType acc_type =
        precision == ::ml_drift::CalculationsPrecision::kF16
            ? ::ml_drift::DataType::kFloat16
            : ::ml_drift::DataType::kFloat32;
    const std::string zero_value = ::ml_drift::GetZeroValue(acc_type);
    const bool w_only = (dst_shape.b == 1 && dst_shape.h == 1);

    std::string c;
    c += "MAIN_FUNCTION($0) {\n";
    // work_group_launch_order_ permutes the dispatched group ids: with the
    // spatial-major order, group id 0 walks the spatial tiles of one block of
    // output slices and group id 1 walks the output slice blocks.
    if (spatial_major) {
      c += "  int sp_tile = ucl::GetGroupId<0>();\n";
      c += "  int dst_group = ucl::GetGroupId<1>();\n";
    } else {
      c += "  int dst_group = ucl::GetGroupId<0>();\n";
      c += "  int sp_tile = ucl::GetGroupId<1>();\n";
    }
    c += "  int dst_s = dst_group * ucl::GetGroupSize<0>() + "
         "ucl::GetLocalId<0>();\n";
    c += "  int dst_end_slice = args.dst_tensor.Slices();\n";
    if (wg_reduction) {
      c += "  int dst_s_wg_offset = dst_group * ucl::GetGroupSize<0>();\n";
      c += "  if (dst_s_wg_offset >= dst_end_slice) return;\n";
    } else {
      c += "  if (dst_s >= dst_end_slice) return;\n";
    }
    c += "  int sp_base = sp_tile * " + std::to_string(tile_w) + ";\n";
    if (w_only) {
      c += "  int total_sp = args.dst_tensor.Width();\n";
    } else {
      c += "  int total_sp = " +
           std::to_string(dst_shape.b * dst_shape.h * dst_shape.w) + ";\n";
    }
    c += "  if (sp_base >= total_sp) return;\n";

    for (int sp_id = 0; sp_id < tile_w; ++sp_id) {
      c += "  AccType r_sp" + std::to_string(sp_id) +
           " = ucl::Init<AccType>(" + zero_value + ");\n";
    }

    if (wg_reduction) {
      c += "  int2 tid;\n";
      c += "  tid.x = ucl::GetLocalId<0>();\n";
      c += "  tid.y = ucl::GetLocalId<1>();\n";
      c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
    }

    const bool is_block_quantized =
        ::ml_drift::fc::IsBlockQuantized(weights_type, scale_zp_shape);
    c += "  Type w_scale;\n";
    c += "  Type w_bias;\n";
    if (!is_block_quantized) {
      if (::ml_drift::fc::IsLinearQuantized(weights_type, scale_zp_shape)) {
        c += ::ml_drift::fc::ReadScaleZeroPointLinear(
            "dst_s", scale_zp_shape, has_zero_point, weights_type);
      } else if (::ml_drift::fc::IsScalarQuantized(weights_type,
                                                   scale_zp_shape)) {
        c += ::ml_drift::fc::ReadScaleZeroPointScalar(has_zero_point,
                                                      weights_type);
      }
    }

    const std::string start_slice = wg_reduction ? "tid.y" : "0";
    const std::string slice_stride = wg_reduction ? "WG_SIZE_Y" : "1";
    if (is_block_quantized) {
      c += "  for (int src_group_id = " + start_slice +
           "; src_group_id < args.src_groups; src_group_id += " +
           slice_stride + ") {\n";
      c += ::ml_drift::fc::ReadScaleZeroPointBlock(
          "dst_s", scale_zp_shape, has_zero_point, weights_type);
      c += "  for (int src_sub_id = 0; src_sub_id < args.src_group_size; "
           "src_sub_id += 1) {\n";
      c += "    int src_s = src_group_id * args.src_group_size + src_sub_id;\n";
    } else {
      c += "  for (int src_s = " + start_slice +
           "; src_s < args.src_tensor.Slices(); src_s += " + slice_stride +
           ") {\n";
    }

    for (int i = 0; i < tile_w; ++i) {
      const std::string val_name = "v" + std::to_string(i);
      if (w_only) {
        std::string x_coord = "sp_base + " + std::to_string(i);
        if (!src.CanReadOutOfBorder(::ml_drift::Axis::kWidth, gpu_info)) {
          x_coord = "min(" + x_coord + ", total_sp - 1)";
        }
        c += "    Type " + val_name + " = args.src_tensor.Read(" + x_coord +
             ", 0, src_s, 0);\n";
      } else {
        c += "    int sp_idx" + std::to_string(i) + " = min(sp_base + " +
             std::to_string(i) + ", total_sp - 1);\n";
        c += "    int b_c" + std::to_string(i) + " = sp_idx" +
             std::to_string(i) + " % " + std::to_string(dst_shape.b) + ";\n";
        c += "    int x_c" + std::to_string(i) + " = (sp_idx" +
             std::to_string(i) + " / " + std::to_string(dst_shape.b) + ") % " +
             std::to_string(dst_shape.w) + ";\n";
        c += "    int y_c" + std::to_string(i) + " = sp_idx" +
             std::to_string(i) + " / " +
             std::to_string(dst_shape.b * dst_shape.w) + ";\n";
        c += "    Type " + val_name + " = args.src_tensor.Read(x_c" +
             std::to_string(i) + ", y_c" + std::to_string(i) +
             ", src_s, b_c" + std::to_string(i) + ");\n";
      }
    }

    c += "    Type w0, w1, w2, w3;\n";
    c += GenerateReadWeightsCode(weights_type, weights_desc);
    const bool isI4O4 = weights_desc.IsI4O4();
    const bool use_fma = ::ml_drift::fc::UseFMA(gpu_info);
    c += ::ml_drift::fc::WeightsScaleAddBias("w_scale", "w_bias", isI4O4,
                                             use_fma);
    for (int i = 0; i < tile_w; ++i) {
      c += ::ml_drift::fc::AccumulateFloat("r_sp" + std::to_string(i),
                                           "v" + std::to_string(i), precision,
                                           isI4O4, use_fma);
    }

    if (is_block_quantized) {
      c += "  } // end for block loop\n";
    }
    c += "  } // end for loop\n";

    if (wg_reduction) {
      c += "  } // end if condition\n";
      const int local_batch_size = ::ml_drift::fc::GetLocalBatchSize(
          gpu_info, precision, tile_w, work_group_size);
      c += "  __local AccType temp";
      if (local_batch_size != 1) {
        c += "[" + std::to_string(local_batch_size) + "]";
      }
      c += "[" + std::to_string(work_group_size.x * work_group_size.y) + "];\n";

      const int upload_groups =
          ::ml_drift::DivideRoundUp(tile_w, local_batch_size);
      for (int group = 0; group < upload_groups; ++group) {
        const int first = group * local_batch_size;
        const int last = std::min(tile_w - 1, first + local_batch_size - 1);
        c += ::ml_drift::fc::GetReductionCode(first, last, local_batch_size,
                                              "tid.x", "tid.y", "WG_SIZE_Y",
                                              "temp");
        if (group != upload_groups - 1) {
          c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
        }
      }
      c += "  if (dst_s >= dst_end_slice) return;\n";
      c += "  if (tid.y != 0) return;\n";
    }

    if (has_bias) {
      c += "  Type bias_value = args.biases.Read(dst_s);\n";
    }
    for (int sp_id = 0; sp_id < tile_w; ++sp_id) {
      const std::string sp_expr = "sp_base + " + std::to_string(sp_id);
      c += "  if (" + sp_expr + " < total_sp) {\n";
      c += "    args.dst_tensor::type res_value = "
           "ucl::Convert<args.dst_tensor::type>(r_sp" +
           std::to_string(sp_id) + ");\n";
      if (has_bias) {
        c += "    res_value += bias_value;\n";
      }
      if (w_only) {
        c += "    args.dst_tensor.Write(res_value, " + sp_expr +
             ", 0, dst_s, 0);\n";
      } else {
        c += "    int sp_idx = " + sp_expr + ";\n";
        c += "    int b_c = sp_idx % " + std::to_string(dst_shape.b) + ";\n";
        c += "    int x_c = (sp_idx / " + std::to_string(dst_shape.b) +
             ") % " + std::to_string(dst_shape.w) + ";\n";
        c += "    int y_c = sp_idx / " +
             std::to_string(dst_shape.b * dst_shape.w) + ";\n";
        c += "    args.dst_tensor.Write(res_value, x_c, y_c, dst_s, b_c);\n";
      }
      c += "  }\n";
    }
    c += "}\n";

    absl::StrReplaceAll(
        {{"SType", ::ml_drift::ToUclDataType(dst.GetDataType(), 1)},
         {"Type", ::ml_drift::ToUclDataType(dst.GetDataType(), 4)},
         {"AccSType", ::ml_drift::ToUclDataType(acc_type, 1)},
         {"AccType", ::ml_drift::ToUclDataType(acc_type, 4)},
         {"WG_SIZE_X", std::to_string(work_group_size.x)},
         {"WG_SIZE_Y", std::to_string(work_group_size.y)}},
        &c);
    return c;
  }

  int tile_w_ = 8;
};

}  // namespace

std::unique_ptr<::ml_drift::GPUOperation> CreateTiledQuantizedFullyConnected(
    const ::ml_drift::GpuInfo& gpu_info,
    ::ml_drift::CalculationsPrecision precision,
    const ::ml_drift::TensorDescriptor& src,
    const ::ml_drift::TensorDescriptor& dst,
    const ::ml_drift::ExternalWeights& weights,
    const ::ml_drift::TensorDescriptor* bias,
    const ::ml_drift::BHWC& dst_shape) {
  return TiledQuantizedFullyConnected::Create(gpu_info, precision, src, dst,
                                              weights, bias, dst_shape);
}

::ml_drift::GpuModelBuilder::TensorHandle AddQuantizedFullyConnected(
    ::ml_drift::GpuModelBuilder* builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::Weights& weights,
    const ::ml_drift::GpuModelBuilder::TensorHandle* biases,
    absl::string_view op_name) {
  const ::ml_drift::BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const int total_spatial = src_shape.b * src_shape.h * src_shape.w;
  // Same precision as the builder's own FullyConnected kernels: FP16
  // activations accumulate in FP32 when the builder runs in `kF32F16`.
  const ::ml_drift::CalculationsPrecision precision =
      builder->GetConvPrecision(src.tensor_desc.GetDataType());
  const auto& gpu_info = builder->gpu_info();
  const bool prefer_conv_fallback =
      std::getenv("LRT_FC_PREFER_CONV") != nullptr &&
      std::string_view(std::getenv("LRT_FC_PREFER_CONV")) == "1";
  // Crossover to ML Drift's own large-batch path (runtime
  // `weights_convert_*_to_float*` + FP16 `conv_wave_memory`), which is also
  // what LiteRT-LM runs for the prefill of `generic_model` LLMs. That path
  // pays a full weight conversion per call, amortized over the tokens of the
  // chunk, but its convolution is ~2.3x faster than the tiled kernel once the
  // chunk fills whole 128-token waves (Qwen 3.5 0.8B, Adreno 840: 1K prompt in
  // 256-token chunks 896.6 vs 733.5 tok/s; 64-token prompt 829 vs 1000 tok/s).
  // The tiled kernel is kept while the conversion would cost at least
  // kTiledMinConversionCostPercent of the convolution, using the cost model of
  // GpuModelBuilder::FullyConnectedSrcFloatExternalWeightsWithConversion
  // (flops_per_byte = 32), i.e. up to ~100 tokens for INT4 -> FP16 weights.
  // `LRT_FC_TILED_MAX_SPATIAL=N` overrides the crossover (0: always tiled).
  constexpr double kTiledMinConversionCostPercent = 40.0;
  bool use_tiled = true;
  const char* tiled_max_env = std::getenv("LRT_FC_TILED_MAX_SPATIAL");
  if (tiled_max_env != nullptr) {
    int tiled_max_spatial = 0;
    if (!absl::SimpleAtoi(tiled_max_env, &tiled_max_spatial)) {
      tiled_max_spatial = 0;
    }
    use_tiled = tiled_max_spatial <= 0 || total_spatial <= tiled_max_spatial;
  } else {
    const double weight_bytes_per_param =
        (::ml_drift::SizeInBitsOf(weights.desc.type) +
         ::ml_drift::SizeInBitsOf(src.tensor_desc.GetDataType())) /
        8.0;
    const double flops_per_param = 2.0 * total_spatial;
    const double conversion_cost_percent =
        100.0 * weight_bytes_per_param / flops_per_param * 32.0;
    use_tiled = conversion_cost_percent >= kTiledMinConversionCostPercent;
  }

  if (!prefer_conv_fallback && use_tiled && weights.shape.h == 1 &&
      total_spatial >
          ::ml_drift::GetRecommendedMaxTotalSpatialSize(gpu_info, precision)) {
    ::ml_drift::BHWC dst_shape = src_shape;
    dst_shape.c = weights.shape.o;
    auto dst = builder->AddTensor(dst_shape, src.tensor_desc.GetDataType());

    ::ml_drift::ExternalWeights ext_weights;
    ext_weights.desc = weights.desc;
    ext_weights.shape = weights.shape;
    ext_weights.scale_zp_shape = weights.scale_zp_shape;
    if (weights.scale.has_value()) {
      ext_weights.scale = &weights.scale->tensor_desc;
    }
    if (weights.zero_point.has_value()) {
      ext_weights.zero_point = &weights.zero_point->tensor_desc;
    }
    const ::ml_drift::TensorDescriptor* bias_desc =
        biases ? &biases->tensor_desc : nullptr;

    auto fc_op = CreateTiledQuantizedFullyConnected(
        gpu_info, precision, src.tensor_desc, dst.tensor_desc, ext_weights,
        bias_desc, dst_shape);
    // Same order as the kernel's source tensors: src, weights, then the
    // optional scale, zero point and bias.
    std::vector<::ml_drift::GpuModelBuilder::ValueId> input_ids = {
        src.id, weights.weights.id};
    if (weights.scale.has_value()) {
      input_ids.push_back(weights.scale->id);
    }
    if (weights.zero_point.has_value()) {
      input_ids.push_back(weights.zero_point->id);
    }
    if (biases) {
      input_ids.push_back(biases->id);
    }
    builder->AddGpuOperation(input_ids, {dst.id}, std::move(fc_op),
                             std::string(op_name));
    return dst;
  }

  if (weights.desc.type == ::ml_drift::DataType::kUint8) {
    return builder->FullyConnectedInt8ExternalWeights(src, weights, biases);
  } else if (weights.desc.type == ::ml_drift::DataType::kUint4) {
    return builder->FullyConnectedInt4ExternalWeights(src, weights, biases);
  } else {
    return builder->FullyConnectedInt2ExternalWeights(src, weights, biases);
  }
}

}  // namespace litert::ml_drift
