/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/delegates/gpu/common/tasks/conv_generic_coop_mat.h"

#include <string>

#include "tflite/delegates/gpu/common/gpu_info.h"
#include "tflite/delegates/gpu/common/tasks/conv_generic.h"
#include "tflite/delegates/gpu/common/task/gpu_operation.h"
#include "tflite/delegates/gpu/common/task/buffer_desc.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/task/weights_layout.h"

namespace tflite {
namespace gpu {

// ============================================================================
// Cooperative Matrix Convolution Kernel (OpenCL C)
// ============================================================================
// Architecture:
// - 64 threads per workgroup (8x8) forced by fixed_work_group_size
// - 2 subgroups of 32 threads each per workgroup
// - 2 local groups per subgroup (16 threads each, 2 threads per pixel position)
// - Cooperative matrix size: 16x16x16 (M=16, N=16, K=16)
// - num_threads_per_pixel_position = subgroup_size / coopMatWidth = 32/16 = 2
// - num_loads_per_subgroup = coopMatDepth / num_threads_per_pixel_position / bundle_size = 16/2/4 = 2
// - num_stores_per_subgroup = coopMatHeight / num_threads_per_pixel_position / bundle_size = 16/2/4 = 2
//
// Weight layout:
//   Weights are rearranged from the standard OHWI layout to a tiled format
//   compatible with coop_mat_load (column-major B matrix, 16x16 tiles):
//
//     [H][W][O/16][I/16][16 x o][16 x i]
//
//   Each [16o][16i] inner block is a 16x16 tile consumed by coop_mat_load
//   with CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR. The outer [H][W]
//   dimension corresponds to kernel spatial positions (oy*w + ox), so the
//   kernel iterates over filter positions linearly when loading weight tiles.
//
//   The standard TFLite GPU delegate uses OHWI + runtime im2col; this
//   direct-convolution kernel avoids im2col by staging input pixels through
//   LDS and loading weights in this pre-tiled layout instead.
// ============================================================================

std::string GenerateConvolutionKernelCooperativeMatrix(
    const OperationDef& definition,
    const GpuInfo& gpu_info,
    GPUOperation* op,
    int out_g_batch,
    int num_tiles) {

  op->AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op->AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  if (definition.src_tensors.size() == 2) {
    const DataType weights_type = definition.GetDataType();
    BufferDescriptor weights_desc;
    weights_desc.element_type = weights_type;
    weights_desc.element_size = 4;
    weights_desc.memory_type = MemoryType::GLOBAL;
    op->AddSrcBuffer("weights", weights_desc);
  }

  op->args_.AddInt("stride_x");
  op->args_.AddInt("padding_x");
  op->args_.AddInt("kernel_size_x");
  op->args_.AddInt("dilation_x");
  op->args_.AddInt("stride_y");
  op->args_.AddInt("padding_y");
  op->args_.AddInt("kernel_size_y");
  op->args_.AddInt("dilation_y");
  op->args_.AddInt("stride_z");
  op->args_.AddInt("padding_z");
  op->args_.AddInt("kernel_size_z");
  op->args_.AddInt("dilation_z");
  op->args_.AddInt("task_size_b");
  op->args_.AddInt("task_size_x");
  op->args_.AddInt("task_size_y");
  op->args_.AddInt("task_size_z");
  op->args_.AddInt("num_output_channel_groups");
  op->args_.AddInt("num_input_channel_groups");
  op->args_.AddInt("output_channels");
  op->args_.AddInt("input_channels");
  op->args_.AddInt("input_channels_padded");
  op->args_.AddInt("output_channels_padded");

  // NB = output channel groups (16 ch each) accumulated per SG per pass (A reuse).
  // NT = pixel tiles (16 px each, 8x2) per SG (B reuse). Accumulators = NT*NB.
  const int NB = out_g_batch < 1 ? 1 : (out_g_batch > 4 ? 4 : out_g_batch);
  const int NT = num_tiles < 1 ? 1 : (num_tiles > 2 ? 2 : num_tiles);
  const std::string NBs = std::to_string(NB);
  auto S = [](int v) { return std::to_string(v); };

  std::string c;
  const bool is_f32 = (definition.precision == CalculationsPrecision::F32);
  const std::string data_type_str = is_f32 ? "float" : "half";
  const std::string acc_type_str = is_f32 ? "float" : "half";
  const std::string zero_lit = is_f32 ? "0.0f" : "0.0h";

  if (!is_f32) {
    c += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }
  c += "#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable\n";
  c += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n\n";
  c += "typedef " + data_type_str + " matA_t __attribute__((coop_mat(\n";
  c += "    \CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP\,\n";
  c += "    16, 16,\n";
  c += "    \CLK_COOPERATIVE_MATRIX_A\n";
  c += ")));\n\n";
  c += "typedef " + data_type_str + " matB_t __attribute__((coop_mat(\n";
  c += "    \CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP\,\n";
  c += "    16, 16,\n";
  c += "    \CLK_COOPERATIVE_MATRIX_B\n";
  c += ")));\n\n";
  c += "typedef " + acc_type_str + " matC_t __attribute__((coop_mat(\n";
  c += "    \CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP\,\n";
  c += "    16, 16,\n";
  c += "    \CLK_COOPERATIVE_MATRIX_ACCUMULATOR\n";
  c += ")));\n\n";

  c += "__attribute__((reqd_work_group_size(8, 8, 1)))\n";
  c += "MAIN_FUNCTION($0) {\n";
// Set batch reference — required for BHWC tensors where HasAxis(BATCH)=true
// but the kernel passes only 3 args to Read/Write (x, y, channel).
// Without this, ParseCoordsFromArgs returns false → "Unrecognized Read selector".
c += "  int B = 0;\n";
c += "  args.src_tensor.SetBatchRef(B);\n";
c += "  args.dst_tensor.SetBatchRef(B);\n";
c += "  const uint subgroupID = get_sub_group_id();\n";
  c += "  const uint subgroupLocalID = get_sub_group_local_id();\n";
  c += "\n";

  // ---- Matrices: NT A, 1 B, NT*NB accumulators ----
  for (int t = 0; t < NT; ++t) c += "  matA_t matA" + S(t) + ";\n";
  c += "  matB_t matB;\n";
  for (int t = 0; t < NT; ++t)
    for (int i = 0; i < NB; ++i)
      c += "  matC_t acc" + S(t) + "_" + S(i) + ";\n";
  c += "\n";

  // ---- LDS: NT tiles x 256 half per SG. Output staging aliases the input tiles. ----
  const int sg_lds = NT * 256;
  c += "  __local " + data_type_str + " tileShared[" + S(2 * sg_lds) + "];\n";
  c += "\n";

  c += "  const int stride_x = args.stride_x;\n";
  c += "  const int stride_y = args.stride_y;\n";
  c += "  const int padding_x = args.padding_x;\n";
  c += "  const int padding_y = args.padding_y;\n";
  c += "  const int dilation_x = args.dilation_x;\n";
  c += "  const int dilation_y = args.dilation_y;\n";
  c += "  const int out_channels = args.output_channels;\n";
  c += "  const int num_input_channel_groups = args.num_input_channel_groups;\n";
  c += "  const int num_output_channel_groups = args.num_output_channel_groups;\n";
  c += "  __global " + data_type_str + "* weights_buffer_half = (__global " + data_type_str + "*)weights_buffer;\n";
  c += "\n";

  // ---- Per-lane staging indices: 2 lanes per pixel, 8 channels (2 x half4) per lane ----
  c += "  const int lanePixel = (int)(subgroupLocalID >> 1);\n";
  c += "  const int laneHalf = (int)(subgroupLocalID & 1u);\n";
  c += "  const int iSubBundledInputDim = laneHalf * 2;\n";
  c += "  const int sharedElem = lanePixel * 16 + laneHalf * 8;\n";
  c += "  __local " + data_type_str + "* mySh = tileShared + (int)subgroupID * " + S(sg_lds) + ";\n";
  c += "\n";

  // ---- Pixel mapping: WG covers 8 x (2*NT) pixels; both SGs cover the SAME pixels ----
  c += "  const int2 iStartPxPos = (int2)(get_group_id(0) * 8, get_group_id(1) * " + S(2 * NT) + ");\n";
  c += "  const int2 dstSize = (int2)(args.dst_tensor.Width(), args.dst_tensor.Height());\n";
  c += "  const int2 srcMax = (int2)(args.src_tensor.Width() - 1, args.src_tensor.Height() - 1);\n";
  for (int t = 0; t < NT; ++t) {
    c += "  const int2 outPxPos" + S(t) + " = iStartPxPos + (int2)(lanePixel % 8, lanePixel / 8 + " + S(2 * t) + ");\n";
    c += "  const bool bOutOK" + S(t) + " = all(outPxPos" + S(t) + " < dstSize);\n";
    c += "  const int2 basePos" + S(t) + " = outPxPos" + S(t) + " * (int2)(stride_x, stride_y) - (int2)(padding_x, padding_y);\n";
  }
  c += "  const int w_tiles_per_kpos = num_output_channel_groups * num_input_channel_groups;\n";
  c += "\n";

  // ---- out_g pass selected by grid Z ----
  c += "  {\n";
  c += "    const int out_g = (int)get_group_id(2) * " + S(2 * NB) + ";\n";
  c += "    const int og_base = out_g + (int)subgroupID * " + NBs + ";\n";
  for (int i = 0; i < NB; ++i) {
    c += "    const int wcol" + S(i) + " = min(og_base + " + S(i) + ", num_output_channel_groups - 1) * num_input_channel_groups * 256;\n";
  }
  for (int t = 0; t < NT; ++t)
    for (int i = 0; i < NB; ++i)
      c += "    acc" + S(t) + "_" + S(i) + " = coop_mat_init(0.0f);\n";
  c += "\n";

  c += "    for (int oy = 0; oy < args.kernel_size_y; ++oy) {\n";
  c += "      for (int ox = 0; ox < args.kernel_size_x; ++ox) {\n";
  c += "        const int2 kOff = (int2)(ox * dilation_x, oy * dilation_y);\n";
  for (int t = 0; t < NT; ++t) {
    c += "        const int2 loadPos" + S(t) + " = basePos" + S(t) + " + kOff;\n";
    c += "        const bool bIn" + S(t) + " = all(loadPos" + S(t) + " >= (int2)(0, 0)) && all(loadPos" + S(t) + " <= srcMax);\n";
  }
  c += "        const int w_kpos = (oy * args.kernel_size_x + ox) * w_tiles_per_kpos * 256;\n";
  c += "        for (int cin_group = 0; cin_group < num_input_channel_groups; ++cin_group) {\n";
  c += "          const int channel_slice_start = cin_group * 4 + iSubBundledInputDim;\n";
  // ---- Stage NT input tiles ----
  for (int t = 0; t < NT; ++t) {
    const std::string ts = S(t);
    c += "          {\n";
    c += "            " + data_type_str + "4 va = (" + data_type_str + "4)(" + zero_lit + ");\n";
    c += "            " + data_type_str + "4 vb = (" + data_type_str + "4)(" + zero_lit + ");\n";
    c += "            if (bIn" + ts + ") {\n";
    c += "              va = args.src_tensor.Read(loadPos" + ts + ".x, loadPos" + ts + ".y, channel_slice_start);\n";
    c += "              vb = args.src_tensor.Read(loadPos" + ts + ".x, loadPos" + ts + ".y, channel_slice_start + 1);\n";
    c += "            }\n";
    c += "            vstore4(va, 0, mySh + " + S(t * 256) + " + sharedElem);\n";
    c += "            vstore4(vb, 0, mySh + " + S(t * 256) + " + sharedElem + 4);\n";
    c += "          }\n";
  }
  for (int t = 0; t < NT; ++t) {
    c += "          matA" + S(t) + " = coop_mat_load(mySh + " + S(t * 256) + ", CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR, 16);\n";
  }
  // ---- NB weight tiles, each reused across NT A tiles ----
  c += "          const int w_base = w_kpos + cin_group * 256;\n";
  for (int i = 0; i < NB; ++i) {
    c += "          matB = coop_mat_load(weights_buffer_half + w_base + wcol" + S(i) + ", CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR, 16);\n";
    for (int t = 0; t < NT; ++t) {
      const std::string a = "acc" + S(t) + "_" + S(i);
      c += "          " + a + " = coop_mat_mulAdd(matA" + S(t) + ", matB, " + a + ");\n";
    }
  }
  c += "        }  // cin_group\n";
  c += "      }  // ox\n";
  c += "    }  // oy\n";
  c += "\n";
  // ---- STORE: per out_g i (uniform branch), stage NT tiles through LDS, bias, write ----
  for (int i = 0; i < NB; ++i) {
    const std::string is = S(i);
    c += "    if (og_base + " + is + " < num_output_channel_groups) {\n";
    c += "      const int my_out_g = og_base + " + is + ";\n";
    c += "      const int bundled_ch_a = my_out_g * 4 + iSubBundledInputDim;\n";
    c += "      const int bundled_ch_b = bundled_ch_a + 1;\n";
    c += "      const bool chOK_a = bundled_ch_a * 4 < out_channels;\n";
    c += "      const bool chOK_b = bundled_ch_b * 4 < out_channels;\n";
    c += "      sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n";
    for (int t = 0; t < NT; ++t) {
      c += "      coop_mat_store(mySh + " + S(t * 256) + ", acc" + S(t) + "_" + is + ", CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR, 16);\n";
    }
    c += "      sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n";
    c += "      " + data_type_str + "4 bias_a = chOK_a ? args.biases.Read(bundled_ch_a) : (" + data_type_str + "4)(" + zero_lit + ");\n";
    c += "      " + data_type_str + "4 bias_b = chOK_b ? args.biases.Read(bundled_ch_b) : (" + data_type_str + "4)(" + zero_lit + ");\n";
    for (int t = 0; t < NT; ++t) {
      const std::string ts = S(t);
      c += "      {\n";
      c += "        " + data_type_str + "4 v_a = vload4(0, mySh + " + S(t * 256) + " + sharedElem) + bias_a;\n";
      c += "        " + data_type_str + "4 v_b = vload4(0, mySh + " + S(t * 256) + " + sharedElem + 4) + bias_b;\n";
      c += "        if (bOutOK" + ts + " && chOK_a) args.dst_tensor.Write(v_a, outPxPos" + ts + ".x, outPxPos" + ts + ".y, bundled_ch_a);\n";
      c += "        if (bOutOK" + ts + " && chOK_b) args.dst_tensor.Write(v_b, outPxPos" + ts + ".x, outPxPos" + ts + ".y, bundled_ch_b);\n";
      c += "      }\n";
    }
    c += "    }\n";
  }
  c += "  }  // out_g pass\n";
  c += "}\n";

  return c;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

bool IsCooperativeMatrixSupported(const GpuInfo& gpu_info) {
  return gpu_info.SupportsExtension("cl_khr_cooperative_matrix");
}

// Rearrange weights from OHWI to tiled [H][W][O_p/16][I_p/16][16 o][16 i] layout
static std::vector<half> RearrangeWeightsForCoopMat(
    const std::vector<float>& weights_data,
    int O, int H, int W, int I) {
  const int I_padded = (I + 15) / 16 * 16;
  const int O_padded = (O + 15) / 16 * 16;
  std::vector<half> result(H * W * O_padded * I_padded);
  std::memset(result.data(), 0, result.size() * sizeof(half));

  for (int o = 0; o < O; ++o) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int i = 0; i < I; ++i) {
          size_t src_idx = ((o * H + h) * W + w) * I + i;
          // Tiled: [H][W][OG][IG][16 o][16 i]
          const size_t og = o / 16, ig = i / 16;
          size_t dst_idx = ((((size_t)(h * W + w) * (O_padded / 16) + og) * (I_padded / 16) + ig) * 256)
                           + (o % 16) * 16 + (i % 16);
          result[dst_idx] = static_cast<half>(weights_data[src_idx]);
        }
      }
    }
  }
  return result;
}

// Rearrange weights from OHWI to tiled [H][W][O_p/16][I_p/16][16 o][16 i] layout (F32 — no half conversion)
static std::vector<float> RearrangeWeightsForCoopMatF32(
    const std::vector<float>& weights_data,
    int O, int H, int W, int I) {
  const int I_padded = (I + 15) / 16 * 16;
  const int O_padded = (O + 15) / 16 * 16;
  std::vector<float> result(H * W * O_padded * I_padded, 0.0f);

  for (int o = 0; o < O; ++o) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int i = 0; i < I; ++i) {
          size_t src_idx = ((o * H + h) * W + w) * I + i;
          const size_t og = o / 16, ig = i / 16;
          size_t dst_idx = ((((size_t)(h * W + w) * (O_padded / 16) + og) * (I_padded / 16) + ig) * 256)
                         + (o % 16) * 16 + (i % 16);
          result[dst_idx] = weights_data[src_idx];
        }
      }
    }
  }
  return result;
}

ConvGeneric CreateDirectConvolutionWithCooperativeMatrix(
    const OperationDef& definition,
    const Convolution2DAttributes& attr,
    const GpuInfo& gpu_info,
    const BHWC* dst_shape) {
  ConvGeneric operation(definition, attr, gpu_info, dst_shape);

  const int output_channels = attr.weights.shape.o;
  const int input_channels = attr.weights.shape.i;
  const int num_output_channel_groups = (output_channels + 15) / 16;
  const int num_input_channel_groups = (input_channels + 15) / 16;

  // Batch up to 4 output channel groups per subgroup (2 subgroups -> up to
  // 8 output groups per workgroup, avoiding register pressure at cout<128).
  int out_g_batch = (num_output_channel_groups + 1) / 2;
  if (out_g_batch > 4) out_g_batch = 4;
  if (out_g_batch < 1) out_g_batch = 1;

  const int num_tiles = 1;

  std::string kernel_code = GenerateConvolutionKernelCooperativeMatrix(
      definition, gpu_info, &operation, out_g_batch, num_tiles);

  static_cast<void>(operation.args_.SetInt("kernel_size_x", attr.weights.shape.w));
  static_cast<void>(operation.args_.SetInt("kernel_size_y", attr.weights.shape.h));
  static_cast<void>(operation.args_.SetInt("dilation_x", attr.dilations.w));
  static_cast<void>(operation.args_.SetInt("dilation_y", attr.dilations.h));
  static_cast<void>(operation.args_.SetInt("stride_x", attr.strides.w));
  static_cast<void>(operation.args_.SetInt("stride_y", attr.strides.h));
  static_cast<void>(operation.args_.SetInt("padding_x", attr.padding.prepended.w));
  static_cast<void>(operation.args_.SetInt("padding_y", attr.padding.prepended.h));

  const int output_channels_padded = num_output_channel_groups * 16;
  const int input_channels_padded = num_input_channel_groups * 16;

  static_cast<void>(operation.args_.SetInt("num_output_channel_groups", num_output_channel_groups));
  static_cast<void>(operation.args_.SetInt("num_input_channel_groups", num_input_channel_groups));
  static_cast<void>(operation.args_.SetInt("output_channels", output_channels));
  static_cast<void>(operation.args_.SetInt("input_channels", input_channels));
  static_cast<void>(operation.args_.SetInt("output_channels_padded", output_channels_padded));
  static_cast<void>(operation.args_.SetInt("input_channels_padded", input_channels_padded));

  static_cast<void>(operation.args_.SetInt("kernel_size_z", 1));

  static_cast<void>(operation.args_.SetInt("task_size_x", dst_shape ? dst_shape->w : 1));
  static_cast<void>(operation.args_.SetInt("task_size_y", dst_shape ? dst_shape->h : 1));
  static_cast<void>(operation.args_.SetInt("task_size_z", dst_shape ? dst_shape->c : 1));
  static_cast<void>(operation.args_.SetInt("task_size_b", 1));

  operation.code_ = kernel_code;
  operation.work_group_size_ = int3(8, 8, 1);
  operation.conv_params_.block_size = int4(64, 1, 1, 1);
  // Use a dummy layout since we handle weight upload manually
  operation.conv_params_.weights_layout = WeightsLayout::kOSpatialIOGroupI4O4;

  operation.conv_params_.linear_spatial = false;
  operation.conv_params_.linear_all = false;
  operation.conv_params_.fixed_work_group_size = true;
  operation.conv_params_.x_kernel_is_1 = false;
  operation.conv_params_.y_kernel_is_1 = false;

  int output_width = dst_shape ? dst_shape->w : 1;
  int output_height = dst_shape ? dst_shape->h : 1;

  // Each workgroup covers 8x2 pixels; both subgroups share the same pixel
  // tile, processing different output channel groups.
  int grid_x = DivideRoundUp(output_width, 8);
  int grid_y = DivideRoundUp(output_height, 2 * num_tiles);

  // Grid Z dispatches output channel groups in passes of 2*NB groups each.
  int grid_z = DivideRoundUp(num_output_channel_groups, 2 * out_g_batch);
  operation.grid_size_ = int3(grid_x * 8, grid_y * 8, grid_z);
  operation.tensor_to_grid_ = TensorToGrid::kCustom;
  operation.grid_dimension_ = 3;

  // ---- Manual weight upload in flat [H][W][O_padded][I_padded] layout ----
  {
    const int O = attr.weights.shape.o;
    const int H = attr.weights.shape.h;
    const int W = attr.weights.shape.w;
    const int I = attr.weights.shape.i;

    // Replace the weights buffer with our rearranged data
    BufferDescriptor desc;
    desc.element_size = 4;
    desc.memory_type = MemoryType::GLOBAL;
    if (definition.precision == CalculationsPrecision::F32) {
      std::vector<float> weight_data_f32 = RearrangeWeightsForCoopMatF32(
          attr.weights.data, O, H, W, I);
      desc.element_type = DataType::FLOAT32;
      desc.size = weight_data_f32.size() * sizeof(float);
      desc.data.resize(desc.size);
      std::memcpy(desc.data.data(), weight_data_f32.data(), desc.size);
    } else {
      std::vector<half> weight_data = RearrangeWeightsForCoopMat(
          attr.weights.data, O, H, W, I);
      desc.element_type = DataType::FLOAT16;
      desc.size = weight_data.size() * sizeof(half);
      desc.data.resize(desc.size);
      std::memcpy(desc.data.data(), weight_data.data(), desc.size);
    }
    operation.args_.AddObject("weights",
        std::make_unique<BufferDescriptor>(std::move(desc)));
  }

  // ---- Upload bias ----
  operation.UploadBias(attr.bias);

  return operation;
}

}  // namespace gpu
}  // namespace tflite
