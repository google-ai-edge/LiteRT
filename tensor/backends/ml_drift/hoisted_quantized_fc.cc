/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/ml_drift/hoisted_quantized_fc.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected_util.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift

namespace litert::tensor {
namespace ml_drift = ::ml_drift;
namespace {

bool OpenClHoistedFcEnabled(const ml_drift::GpuInfo& gpu_info,
                            ml_drift::DataType src_dtype, int total_spatial) {
  if (!gpu_info.IsApiOpenCl()) return false;
  if (total_spatial != 1) return false;
  if (src_dtype != ml_drift::DataType::kFloat16) return false;
  if (const char* env = std::getenv("LRT_OPENCL_HOISTED_FC");
      env != nullptr && env[0] != '\0') {
    const absl::string_view sv(env);
    return sv == "1" || sv == "true";
  }
  return false;
}

int RangeShiftForStorageType(ml_drift::DataType storage_type) {
  if (storage_type == ml_drift::DataType::kUint8) return 128;
  if (storage_type == ml_drift::DataType::kUint4) return 8;
  if (storage_type == ml_drift::DataType::kUint2) return 2;
  return 0;
}

ml_drift::int3 SelectWorkGroupSize(const ml_drift::GpuInfo& gpu_info,
                                   ml_drift::DataType weights_type,
                                   int dst_slices, int max_y_size,
                                   int total_spatial, int spatial_tiles) {
  int wg_total_size = 128;
  if (gpu_info.IsAdreno()) {
    wg_total_size =
        gpu_info.adreno_info.generation >=
                ml_drift::AdrenoInfo::Generation::kGen7
            ? 256
            : 128;
    if (gpu_info.adreno_info.IsLowEnd()) {
      wg_total_size = 32;
    }
  } else if (gpu_info.IsDesktopGpu() &&
             gpu_info.GetComputeUnitsCount() >= 32) {
    wg_total_size = 256;
  }

  if (total_spatial == 1) {
    int y_size = weights_type == ml_drift::DataType::kInt2 ? 32 : 16;
    if (dst_slices >= 512) {
      y_size = weights_type == ml_drift::DataType::kInt8 ? 8 : 16;
    }
    if (dst_slices >= 768) y_size = 8;
    if (dst_slices >= 1024) y_size = 4;
    if (dst_slices >= 1024 * 2 && weights_type == ml_drift::DataType::kInt8) {
      y_size = 2;
    }
    if (dst_slices >= 1024 * 4 && weights_type != ml_drift::DataType::kInt2) {
      y_size = std::min(2, y_size);
    }
    if (dst_slices >= 1024 * 8) y_size = 1;
    y_size = std::min(y_size, wg_total_size);
    int x_size = std::max(1, wg_total_size / y_size);
    if (dst_slices <= 64 && x_size >= 8) {
      x_size /= 2;
      y_size *= 2;
    }
    const int cu_count = std::max(1, gpu_info.GetComputeUnitsCount());
    while (y_size > max_y_size && y_size > 1 && x_size < dst_slices) {
      y_size /= 2;
      x_size *= 2;
    }
    while (y_size > max_y_size && y_size > 1) {
      y_size /= 2;
    }
    while ((x_size > dst_slices ||
            (dst_slices / x_size < cu_count && y_size * 2 <= max_y_size)) &&
           x_size > 1 && y_size * 2 <= wg_total_size) {
      x_size /= 2;
      y_size *= 2;
    }
    return ml_drift::int3(x_size, y_size, 1);
  }

  constexpr int kMaxTiledYSize = 16;
  const int y_limit =
      std::max(1, std::min({max_y_size, kMaxTiledYSize, wg_total_size}));
  const int target_threads =
      std::max(1, gpu_info.GetComputeUnitsCount()) * 1024;
  const int independent_tasks = dst_slices * spatial_tiles;
  int y_size = 1;
  while (independent_tasks * y_size < target_threads &&
         y_size * 2 <= y_limit) {
    y_size *= 2;
  }
  int x_size = std::max(1, std::min(64, wg_total_size / y_size));
  while (x_size > 8 && x_size / 2 >= dst_slices) {
    x_size /= 2;
  }
  return ml_drift::int3(x_size, y_size, 1);
}

void AppendUnpackWeights(const ml_drift::GpuInfo& gpu_info,
                         const ml_drift::WeightsDescription& weights_desc,
                         bool signed_int8_xor, std::string* code) {
  if (weights_desc.type == ml_drift::DataType::kUint8) {
    if (weights_desc.IsLinearLayout()) {
      absl::StrAppend(code,
                      "    int linear_i4o4 = src_s * args.dst_tensor.Slices() "
                      "+ dst_s;\n",
                      "    uint4 w = args.weights.Read(linear_i4o4);\n");
    } else {
      absl::StrAppend(code,
                      "    uint4 w = args.weights.Read(dst_s, src_s);\n");
    }
    if (signed_int8_xor) {
      // XOR with 0x80808080 converts stored unsigned `q + 128` back to signed
      // `char4` in [-128, 127], making symmetric INT8 zero-mean with no bias
      // correction term.
      absl::StrAppend(
          code,
          "    w0 = ucl::Convert<float4>("
          "ucl::Reinterpret<uint, char4>(w.x ^ 0x80808080u));\n",
          "    w1 = ucl::Convert<float4>("
          "ucl::Reinterpret<uint, char4>(w.y ^ 0x80808080u));\n",
          "    w2 = ucl::Convert<float4>("
          "ucl::Reinterpret<uint, char4>(w.z ^ 0x80808080u));\n",
          "    w3 = ucl::Convert<float4>("
          "ucl::Reinterpret<uint, char4>(w.w ^ 0x80808080u));\n");
    } else {
      absl::StrAppend(code,
                      "    ucl::U32x4ToU8x16AsVec4x4<float>(w, w0, w1, w2, "
                      "w3);\n");
    }
    return;
  }
  if (weights_desc.type == ml_drift::DataType::kUint4) {
    if (weights_desc.IsLinearLayout()) {
      absl::StrAppend(code,
                      "    int linear_i4o4 = src_s * args.dst_tensor.Slices() "
                      "+ dst_s;\n",
                      "    uint2 w = args.weights.Read(linear_i4o4);\n",
                      "    ucl::U32x2ToU4x16AsVec4x4<float>(w, w0, w1, w2, "
                      "w3);\n");
    } else {
      absl::StrAppend(code,
                      "    ushort4 w = args.weights.Read(dst_s, src_s);\n",
                      "    ucl::U16x4ToU4x16AsVec4x4<float>(w, w0, w1, w2, "
                      "w3);\n");
    }
    return;
  }
  // `kUint2`
  if (weights_desc.IsLinearLayout()) {
    absl::StrAppend(code,
                    "    int linear_i4o4 = src_s * args.dst_tensor.Slices() "
                    "+ dst_s;\n",
                    "    uint w = args.weights.Read(linear_i4o4);\n",
                    "    ucl::U32x1ToU2x16AsVec4x4<float>(w, w0, w1, w2, "
                    "w3);\n");
  } else {
    absl::StrAppend(code,
                    "    uchar4 w = args.weights.Read(dst_s, src_s);\n",
                    "    ucl::U8x4ToU2x16AsVec4x4<float>(w, w0, w1, w2, "
                    "w3);\n");
  }
  (void)gpu_info;
}

void AppendApplyHoistedScaleBias(const ml_drift::OHWI& scale_zp_shape,
                                 ml_drift::DataType weights_type,
                                 int effective_range_shift,
                                 bool is_block_quantized, bool has_zero_point,
                                 int tile_w, const std::string& acc_prefix,
                                 const std::string& xsum_prefix,
                                 const std::string& dst_prefix,
                                 const std::string& assign_op, bool use_fma,
                                 std::string* code) {
  const bool needs_bias = effective_range_shift != 0 || has_zero_point;
  absl::StrAppend(code, "    {\n");
  if (is_block_quantized) {
    absl::StrAppend(code,
                    "      float4 w_scale = args.weights_scale.Read<float>("
                    "dst_s, 0, src_group_id);\n");
  } else if (ml_drift::fc::IsLinearQuantized(weights_type, scale_zp_shape)) {
    absl::StrAppend(code,
                    "      float4 w_scale = args.weights_scale.Read<float>("
                    "dst_s);\n");
  } else {
    absl::StrAppend(code,
                    "      float4 w_scale = ucl::Init<float4>(args.scale);\n");
  }
  if (needs_bias) {
    if (has_zero_point) {
      if (is_block_quantized) {
        absl::StrAppend(
            code,
            "      float4 w_zp = args.weights_zero_point.Read<float>("
            "dst_s, 0, src_group_id);\n");
      } else if (ml_drift::fc::IsLinearQuantized(weights_type,
                                                 scale_zp_shape)) {
        absl::StrAppend(
            code,
            "      float4 w_zp = args.weights_zero_point.Read<float>("
            "dst_s);\n");
      } else {
        absl::StrAppend(
            code,
            "      float4 w_zp = ucl::Init<float4>(args.zero_point);\n");
      }
      if (effective_range_shift != 0) {
        absl::StrAppend(code, "      float4 w_bias = -w_scale * ("
                              "ucl::Init<float4>(",
                        effective_range_shift, ".0f) + w_zp);\n");
      } else {
        absl::StrAppend(code, "      float4 w_bias = -w_scale * w_zp;\n");
      }
    } else {
      absl::StrAppend(code,
                      "      float4 w_bias = -w_scale * ucl::Init<float4>(",
                      effective_range_shift, ".0f);\n");
    }
    for (int i = 0; i < tile_w; ++i) {
      if (use_fma) {
        absl::StrAppend(code, "      ", dst_prefix, i, " ", assign_op, " fma(",
                        acc_prefix, i, ", w_scale, ucl::Init<float4>(",
                        xsum_prefix, i, ") * w_bias);\n");
      } else {
        absl::StrAppend(code, "      ", dst_prefix, i, " ", assign_op, " ",
                        acc_prefix, i, " * w_scale + ", xsum_prefix, i,
                        " * w_bias;\n");
      }
    }
  } else {
    for (int i = 0; i < tile_w; ++i) {
      absl::StrAppend(code, "      ", dst_prefix, i, " ", assign_op, " ",
                      acc_prefix, i, " * w_scale;\n");
    }
  }
  absl::StrAppend(code, "    }\n");
}

class HoistedQuantizedFullyConnected : public ml_drift::GPUOperation {
 public:
  HoistedQuantizedFullyConnected() = default;
  HoistedQuantizedFullyConnected(HoistedQuantizedFullyConnected&&) = default;
  HoistedQuantizedFullyConnected& operator=(HoistedQuantizedFullyConnected&&) =
      default;
  HoistedQuantizedFullyConnected(const HoistedQuantizedFullyConnected&) =
      delete;
  HoistedQuantizedFullyConnected& operator=(
      const HoistedQuantizedFullyConnected&) = delete;

  std::vector<ml_drift::int3> GetPossibleKernelWorkGroups(
      ml_drift::TuningType tuning_type, const ml_drift::GpuInfo& gpu_info,
      const ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  ml_drift::int3 GetGridSize() const override {
    const int total_spatial =
        dst_[0]->Batch() * dst_[0]->Height() * dst_[0]->Width();
    const int spatial_tiles = ml_drift::DivideRoundUp(total_spatial, tile_w_);
    return ml_drift::int3(dst_[0]->Slices(),
                          work_group_size_.y * spatial_tiles, 1);
  }

  static std::unique_ptr<HoistedQuantizedFullyConnected> Create(
      const ml_drift::GpuInfo& gpu_info, const ml_drift::TensorDescriptor& src,
      const ml_drift::TensorDescriptor& dst,
      const ml_drift::ExternalWeights& weights,
      const ml_drift::TensorDescriptor* bias, const ml_drift::BHWC& dst_shape) {
    auto op = std::make_unique<HoistedQuantizedFullyConnected>();
    const ml_drift::WeightsDescription& weights_desc = weights.desc;
    const ml_drift::OHWI& weights_shape = weights.shape;
    const ml_drift::DataType weights_type =
        ml_drift::fc::GetDataTypeForWeights(weights_desc.type);
    const ml_drift::OHWI scale_zp_shape =
        weights.scale ? weights.scale_zp_shape : ml_drift::OHWI(1, 1, 1, 1);
    const bool has_bias = bias != nullptr;
    const bool has_zero_point =
        weights.zero_point != nullptr || weights.scalar_zero_point.has_value();

    const int total_spatial = dst_shape.b * dst_shape.h * dst_shape.w;
    const int tile_w = total_spatial == 1 ? 1 : std::min(8, total_spatial);
    op->tile_w_ = tile_w;

    op->AddSrcTensor("src_tensor", src);
    op->AddDstTensor("dst_tensor", dst);
    const int dst_slices = ml_drift::DivideRoundUp(weights_shape.o, 4);
    const int src_slices = ml_drift::DivideRoundUp(weights_shape.i, 4);
    if (scale_zp_shape.i != 1) {
      const int src_groups = scale_zp_shape.i;
      op->args_.AddInt("src_groups", src_groups);
      op->args_.AddInt("src_group_size", src_slices / src_groups);
    }

    const int spatial_tiles = ml_drift::DivideRoundUp(total_spatial, tile_w);
    const int max_y_size =
        scale_zp_shape.i != 1 ? scale_zp_shape.i : src_slices;
    op->work_group_size_ = SelectWorkGroupSize(
        gpu_info, weights_type, dst_slices, max_y_size, total_spatial,
        spatial_tiles);
    const bool wg_reduction = op->work_group_size_.y != 1;
    const bool spatial_major = spatial_tiles > 1;
    op->work_group_launch_order_ =
        spatial_major ? ml_drift::int3(1, 0, 2) : ml_drift::int3(0, 1, 2);

    op->code_ = GenerateKernelCode(
        src, dst_shape, gpu_info, weights_desc, weights_type, scale_zp_shape,
        has_bias, has_zero_point, tile_w, wg_reduction, spatial_major,
        op->work_group_size_);
    // All arithmetic is in FP32 (`float4`), where FP16-subnormal scales
    // (`~1e-5`) are normal IEEE-754 floats; avoid `-cl-fast-relaxed-math`
    // while using explicit `fma(...)` so Adreno/Mali emit single-cycle FMAs
    // without risking half-precision FTZ or unsafe compiler reassociation.
    ml_drift::fc::AddWeightsArguments(weights, /*vec_size=*/16, op.get());
    if (bias != nullptr) {
      op->AddSrcTensor("biases", *bias);
    }
    op->flops_ = static_cast<uint64_t>(dst_shape.b) * dst_shape.h *
                 dst_shape.w * dst_shape.c * weights_shape.i * 2ULL;
    return op;
  }

 private:
  static std::string GenerateKernelCode(
      const ml_drift::TensorDescriptor& src, const ml_drift::BHWC& dst_shape,
      const ml_drift::GpuInfo& gpu_info,
      const ml_drift::WeightsDescription& weights_desc,
      ml_drift::DataType weights_type, const ml_drift::OHWI& scale_zp_shape,
      bool has_bias, bool has_zero_point, int tile_w, bool wg_reduction,
      bool spatial_major, const ml_drift::int3& work_group_size) {
    const int total_spatial = dst_shape.b * dst_shape.h * dst_shape.w;
    const bool w_only = (dst_shape.b == 1 && dst_shape.h == 1);
    const bool single_token = (total_spatial == 1);
    const bool is_block_quantized =
        ml_drift::fc::IsBlockQuantized(weights_type, scale_zp_shape);
    const bool signed_int8_xor =
        weights_desc.type == ml_drift::DataType::kUint8 &&
        (gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal());
    const int effective_range_shift =
        signed_int8_xor ? 0 : RangeShiftForStorageType(weights_desc.type);
    const bool track_x_sum = effective_range_shift != 0 || has_zero_point;
    const bool use_fma = ml_drift::fc::UseFMA(gpu_info);

    std::string c;
    c += "MAIN_FUNCTION($0) {\n";
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
    if (!single_token) {
      absl::StrAppend(&c, "  int sp_base = sp_tile * ", tile_w, ";\n");
      if (w_only) {
        c += "  int total_sp = args.dst_tensor.Width();\n";
      } else {
        absl::StrAppend(&c, "  int total_sp = ", total_spatial, ";\n");
      }
      c += "  if (sp_base >= total_sp) return;\n";
    }

    for (int sp_id = 0; sp_id < tile_w; ++sp_id) {
      absl::StrAppend(&c, "  float4 r_sp", sp_id,
                      " = ucl::Init<float4>(0.0f);\n");
    }
    if (wg_reduction) {
      c += "  int2 tid;\n";
      c += "  tid.x = ucl::GetLocalId<0>();\n";
      c += "  tid.y = ucl::GetLocalId<1>();\n";
      c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
    }

    auto emit_step_body = [&](const std::string& acc_prefix,
                              const std::string& xsum_prefix) {
      for (int i = 0; i < tile_w; ++i) {
        const std::string val_name = absl::StrCat("v", i);
        if (single_token) {
          absl::StrAppend(&c, "    float4 ", val_name,
                          " = args.src_tensor.Read<float>(0, 0, src_s, 0);\n");
        } else if (w_only) {
          std::string x_coord = absl::StrCat("sp_base + ", i);
          if (!src.CanReadOutOfBorder(ml_drift::Axis::kWidth, gpu_info)) {
            x_coord = absl::StrCat("min(", x_coord, ", total_sp - 1)");
          }
          absl::StrAppend(&c, "    float4 ", val_name,
                          " = args.src_tensor.Read<float>(", x_coord,
                          ", 0, src_s, 0);\n");
        } else {
          absl::StrAppend(&c, "    int sp_idx", i, " = min(sp_base + ", i,
                          ", total_sp - 1);\n");
          absl::StrAppend(&c, "    int b_c", i, " = sp_idx", i, " % ",
                          dst_shape.b, ";\n");
          absl::StrAppend(&c, "    int x_c", i, " = (sp_idx", i, " / ",
                          dst_shape.b, ") % ", dst_shape.w, ";\n");
          absl::StrAppend(&c, "    int y_c", i, " = sp_idx", i, " / ",
                          dst_shape.b * dst_shape.w, ";\n");
          absl::StrAppend(&c, "    float4 ", val_name,
                          " = args.src_tensor.Read<float>(x_c", i, ", y_c", i,
                          ", src_s, b_c", i, ");\n");
        }
        if (track_x_sum) {
          absl::StrAppend(&c, "    ", xsum_prefix, i, " += (", val_name,
                          ".x + ", val_name, ".y) + (", val_name, ".z + ",
                          val_name, ".w);\n");
        }
      }

      c += "    float4 w0, w1, w2, w3;\n";
      AppendUnpackWeights(gpu_info, weights_desc, signed_int8_xor, &c);
      for (int i = 0; i < tile_w; ++i) {
        const std::string acc = absl::StrCat(acc_prefix, i);
        const std::string v = absl::StrCat("v", i);
        if (use_fma) {
          absl::StrAppend(&c, "    ", acc, " = fma(ucl::Init<float4>(", v,
                          ".x), w0, ", acc, ");\n");
          absl::StrAppend(&c, "    ", acc, " = fma(ucl::Init<float4>(", v,
                          ".y), w1, ", acc, ");\n");
          absl::StrAppend(&c, "    ", acc, " = fma(ucl::Init<float4>(", v,
                          ".z), w2, ", acc, ");\n");
          absl::StrAppend(&c, "    ", acc, " = fma(ucl::Init<float4>(", v,
                          ".w), w3, ", acc, ");\n");
        } else {
          absl::StrAppend(&c, "    ", acc, " += ", v, ".x * w0 + ", v,
                          ".y * w1 + ", v, ".z * w2 + ", v, ".w * w3;\n");
        }
      }
    };

    const std::string start_slice = wg_reduction ? "tid.y" : "0";
    const std::string slice_stride = wg_reduction ? "WG_SIZE_Y" : "1";
    if (is_block_quantized) {
      absl::StrAppend(&c, "  for (int src_group_id = ", start_slice,
                      "; src_group_id < args.src_groups; src_group_id += ",
                      slice_stride, ") {\n");
      for (int i = 0; i < tile_w; ++i) {
        absl::StrAppend(&c, "    float4 b_acc", i,
                        " = ucl::Init<float4>(0.0f);\n");
        if (track_x_sum) {
          absl::StrAppend(&c, "    float b_xsum", i, " = 0.0f;\n");
        }
      }
      c += "    for (int src_sub_id = 0; src_sub_id < args.src_group_size; "
           "src_sub_id += 1) {\n";
      c += "      int src_s = src_group_id * args.src_group_size + "
           "src_sub_id;\n";
      emit_step_body("b_acc", "b_xsum");
      c += "    }\n";
      AppendApplyHoistedScaleBias(scale_zp_shape, weights_type,
                                  effective_range_shift,
                                  /*is_block_quantized=*/true, has_zero_point,
                                  tile_w, "b_acc", "b_xsum", "r_sp", "+=",
                                  use_fma, &c);
      c += "  }\n";
    } else {
      for (int i = 0; i < tile_w; ++i) {
        absl::StrAppend(&c, "  float4 acc", i,
                        " = ucl::Init<float4>(0.0f);\n");
        if (track_x_sum) {
          absl::StrAppend(&c, "  float xsum", i, " = 0.0f;\n");
        }
      }
      absl::StrAppend(&c, "  for (int src_s = ", start_slice,
                      "; src_s < args.src_tensor.Slices(); src_s += ",
                      slice_stride, ") {\n");
      emit_step_body("acc", "xsum");
      c += "  }\n";
      AppendApplyHoistedScaleBias(scale_zp_shape, weights_type,
                                  effective_range_shift,
                                  /*is_block_quantized=*/false, has_zero_point,
                                  tile_w, "acc", "xsum", "r_sp", "=", use_fma,
                                  &c);
    }

    if (wg_reduction) {
      c += "  }\n";
      const int local_batch_size = ml_drift::fc::GetLocalBatchSize(
          gpu_info, ml_drift::CalculationsPrecision::kF32, tile_w,
          work_group_size);
      c += "  __local float4 temp";
      if (local_batch_size != 1) {
        absl::StrAppend(&c, "[", local_batch_size, "]");
      }
      absl::StrAppend(&c, "[", work_group_size.x * work_group_size.y, "];\n");

      const int upload_groups =
          ml_drift::DivideRoundUp(tile_w, local_batch_size);
      for (int group = 0; group < upload_groups; ++group) {
        const int first = group * local_batch_size;
        const int last = std::min(tile_w - 1, first + local_batch_size - 1);
        c += ml_drift::fc::GetReductionCode(first, last, local_batch_size,
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
      c += "  float4 bias_value = args.biases.Read<float>(dst_s);\n";
    }
    if (single_token) {
      if (has_bias) {
        c += "  r_sp0 += bias_value;\n";
      }
      c += "  args.dst_tensor::type res_value = "
           "ucl::Convert<args.dst_tensor::type>(r_sp0);\n";
      c += "  args.dst_tensor.Write(res_value, 0, 0, dst_s, 0);\n";
    } else {
      for (int sp_id = 0; sp_id < tile_w; ++sp_id) {
        const std::string sp_expr = absl::StrCat("sp_base + ", sp_id);
        const std::string r_name = absl::StrCat("r_sp", sp_id);
        absl::StrAppend(&c, "  if (", sp_expr, " < total_sp) {\n");
        if (has_bias) {
          absl::StrAppend(&c, "    ", r_name, " += bias_value;\n");
        }
        absl::StrAppend(&c,
                        "    args.dst_tensor::type res_value = "
                        "ucl::Convert<args.dst_tensor::type>(",
                        r_name, ");\n");
        if (w_only) {
          absl::StrAppend(&c, "    args.dst_tensor.Write(res_value, ", sp_expr,
                          ", 0, dst_s, 0);\n");
        } else {
          absl::StrAppend(&c, "    int sp_idx = ", sp_expr, ";\n");
          absl::StrAppend(&c, "    int b_c = sp_idx % ", dst_shape.b, ";\n");
          absl::StrAppend(&c, "    int x_c = (sp_idx / ", dst_shape.b, ") % ",
                          dst_shape.w, ";\n");
          absl::StrAppend(&c, "    int y_c = sp_idx / ",
                          dst_shape.b * dst_shape.w, ";\n");
          absl::StrAppend(
              &c,
              "    args.dst_tensor.Write(res_value, x_c, y_c, dst_s, b_c);\n");
        }
        c += "  }\n";
      }
    }
    c += "}\n";

    absl::StrReplaceAll({{"WG_SIZE_X", std::to_string(work_group_size.x)},
                         {"WG_SIZE_Y", std::to_string(work_group_size.y)}},
                        &c);
    return c;
  }

  int tile_w_ = 1;
};

}  // namespace

std::optional<ml_drift::GpuModelBuilder::TensorHandle>
TryAddHoistedQuantizedFullyConnected(
    ml_drift::GpuModelBuilder* builder,
    const ml_drift::GpuModelBuilder::TensorHandle& src,
    const ml_drift::GpuModelBuilder::Weights& weights,
    const ml_drift::GpuModelBuilder::TensorHandle* biases,
    absl::string_view op_name, bool force_for_f32_scale) {
  if (builder == nullptr) return std::nullopt;
  if (weights.shape.h != 1 || weights.shape.w != 1 || weights.shape.o <= 0 ||
      weights.shape.i <= 0) {
    return std::nullopt;
  }
  if (!weights.desc.IsI4O4()) return std::nullopt;
  if (weights.desc.type != ml_drift::DataType::kUint2 &&
      weights.desc.type != ml_drift::DataType::kUint4 &&
      weights.desc.type != ml_drift::DataType::kUint8) {
    return std::nullopt;
  }
  if (!weights.scale.has_value() || weights.scale_zp_shape.h != 1 ||
      weights.scale_zp_shape.w != 1) {
    return std::nullopt;
  }
  const ml_drift::DataType src_dtype = src.tensor_desc.GetDataType();
  if (src_dtype != ml_drift::DataType::kFloat16 &&
      src_dtype != ml_drift::DataType::kFloat32) {
    return std::nullopt;
  }
  const ml_drift::BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const int total_spatial = src_shape.b * src_shape.h * src_shape.w;
  const int src_slices = ml_drift::DivideRoundUp(weights.shape.i, 4);
  if (weights.scale_zp_shape.i > 1 &&
      src_slices % weights.scale_zp_shape.i != 0) {
    return std::nullopt;
  }
  if (!force_for_f32_scale &&
      !OpenClHoistedFcEnabled(builder->gpu_info(), src_dtype, total_spatial)) {
    return std::nullopt;
  }

  ml_drift::BHWC dst_shape = src_shape;
  dst_shape.c = weights.shape.o;
  ml_drift::GpuModelBuilder::TensorHandle dst =
      builder->AddTensor(dst_shape, src_dtype);

  ml_drift::ExternalWeights ext_weights;
  ext_weights.desc = weights.desc;
  ext_weights.shape = weights.shape;
  ext_weights.scale_zp_shape = weights.scale_zp_shape;
  ext_weights.scale = &weights.scale->tensor_desc;
  if (weights.zero_point.has_value()) {
    ext_weights.zero_point = &weights.zero_point->tensor_desc;
  }
  const ml_drift::TensorDescriptor* bias_desc =
      biases != nullptr ? &biases->tensor_desc : nullptr;

  auto fc_op = HoistedQuantizedFullyConnected::Create(
      builder->gpu_info(), src.tensor_desc, dst.tensor_desc, ext_weights,
      bias_desc, dst_shape);
  std::vector<ml_drift::GpuModelBuilder::ValueId> input_ids = {
      src.id, weights.weights.id, weights.scale->id};
  if (weights.zero_point.has_value()) {
    input_ids.push_back(weights.zero_point->id);
  }
  if (biases != nullptr) {
    input_ids.push_back(biases->id);
  }
  builder->AddGpuOperation(input_ids, {dst.id}, std::move(fc_op),
                           absl::StrCat(op_name, "_hoisted"));
  return dst;
}

}  // namespace litert::tensor
