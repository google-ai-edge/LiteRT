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

#include "tflite/delegates/gpu/common/tasks/conv_coop_mat.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/buffer_desc.h"
#include "tflite/delegates/gpu/common/task/gpu_tensor.h"
#include "tflite/delegates/gpu/common/types.h"
#include "tflite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {

std::string GenerateConvolutionKernelCooperativeMatrix(
    const OperationDef& definition, int out_g_batch, int num_tiles) {
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
  c += "    CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,\n";
  c += "    16, 16,\n";
  c += "    CLK_COOPERATIVE_MATRIX_A\n";
  c += ")));\n\n";
  c += "typedef " + data_type_str + " matB_t __attribute__((coop_mat(\n";
  c += "    CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,\n";
  c += "    16, 16,\n";
  c += "    CLK_COOPERATIVE_MATRIX_B\n";
  c += ")));\n\n";
  c += "typedef " + acc_type_str + " matC_t __attribute__((coop_mat(\n";
  c += "    CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,\n";
  c += "    16, 16,\n";
  c += "    CLK_COOPERATIVE_MATRIX_ACCUMULATOR\n";
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

// Rearrange weights from OHWI to tiled [H][W][O_p/16][I_p/16][16 o][16 i] layout
static std::vector<half> RearrangeWeightsForCoopMat(
    const std::vector<float>& weights_data,
    int O, int H, int W, int I) {
  const int I_padded = AlignByN(I, 16);
  const int O_padded = AlignByN(O, 16);
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
  const int I_padded = AlignByN(I, 16);
  const int O_padded = AlignByN(O, 16);
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

}  // namespace

// Static eligibility for the cooperative-matrix kernel (cl_khr_cooperative_matrix,
// 16x16 tiles, supports both F16 and F32 precision and accumulator modes).
bool IsConvCoopMatSupported(const GpuInfo& gpu_info,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           const BHWC& dst_shape) {
  // coop_mat_* builtins do not compile without the extension.
  if (!gpu_info.IsApiOpenCl() ||
      !gpu_info.SupportsExtension("cl_khr_cooperative_matrix")) {
    return false;
  }
  // The kernel supports F16 (half/half4) and F32 (float/float4) tensor types.
  if (definition.precision != CalculationsPrecision::F16 &&
      definition.precision != CalculationsPrecision::F32) {
    return false;
  }
  // The kernel addresses only batch zero and requires a 2D pixel plane.
  if (dst_shape.b != 1 || dst_shape.w <= 1 || dst_shape.h <= 1) {
    return false;
  }
  if (attr.groups != 1 || definition.src_tensors.size() != 1 ||
      definition.dst_tensors.size() != 1) {
    return false;
  }
  // Input channels must fill the K=16 tiles. cin=3 padded to 16 wastes 81%
  // of the MACs on zero-padding.
  if (attr.weights.shape.i <= 0 || attr.weights.shape.i % 16 != 0 ||
      attr.weights.shape.o <= 0 || dst_shape.c != attr.weights.shape.o) {
    return false;
  }
  return true;
}

// Per-layer routing heuristic. The cooperative matrix path loses to the
// standard FMA path when:
//  (a) cout < 64: at most 3 output channel groups, no input-tile reuse
//  (b) kh*kw*ceil(cin/16) < 4 K iterations: cooperative matrix epilogue
//      dominates
//  (c) 1x1 conv with < 24 workgroups: too few waves to hide latency
//  (d) any conv on tiny outputs: too few workgroups
//  (e) pointwise channel reduction with an in:out group ratio > 8:
//      rearrange/buffer overhead outweighs the few output tiles
bool IsGoodTaskSizeForConvCoopMat(const Convolution2DAttributes& attr,
                               const BHWC& dst_shape) {
  const int cin = attr.weights.shape.i;
  const int cout = attr.weights.shape.o;
  const int kh = attr.weights.shape.h;
  const int kw = attr.weights.shape.w;
  const int cin_groups = DivideRoundUp(cin, 16);
  const int cout_groups = DivideRoundUp(cout, 16);
  const int k_iters = kh * kw * cin_groups;
  const int out_g_batch = std::max(1, std::min(4, DivideRoundUp(cout_groups, 2)));
  const int work_groups = DivideRoundUp(dst_shape.w, 8) *
                          DivideRoundUp(dst_shape.h, 2) *
                          DivideRoundUp(cout_groups, 2 * out_g_batch);

  if (cout < 64) {  // (a)
    return false;
  }
  if (k_iters < 4) {  // (b)
    return false;
  }
  if (kh == 1 && kw == 1 && work_groups < 24) {  // (c)
    return false;
  }
  if (work_groups < 8) {  // (d)
    return false;
  }
  const bool is_pointwise =  // (e)
      kh == 1 && kw == 1 && attr.strides.w == 1 && attr.strides.h == 1 &&
      attr.dilations.w == 1 && attr.dilations.h == 1 &&
      attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0;
  return !(is_pointwise && cin_groups > 8 * cout_groups);
}

ConvCoopMat::ConvCoopMat(const OperationDef& definition,
                       const Convolution2DAttributes& attr)
    : GPUOperation(definition),
      out_g_batch_(std::max(1, std::min(
          4, DivideRoundUp(DivideRoundUp(attr.weights.shape.o, 16), 2)))) {
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  args_.AddInt("kernel_size_x", attr.weights.shape.w);
  args_.AddInt("kernel_size_y", attr.weights.shape.h);
  args_.AddInt("dilation_x", attr.dilations.w);
  args_.AddInt("dilation_y", attr.dilations.h);
  args_.AddInt("stride_x", attr.strides.w);
  args_.AddInt("stride_y", attr.strides.h);
  args_.AddInt("padding_x", attr.padding.prepended.w);
  args_.AddInt("padding_y", attr.padding.prepended.h);
  args_.AddInt("num_output_channel_groups");
  args_.AddInt("num_input_channel_groups");
  args_.AddInt("output_channels");
  code_ = GenerateConvolutionKernelCooperativeMatrix(
      definition, out_g_batch_, /*num_tiles=*/1);
  work_group_size_ = int3(8, 8, 1);
  tensor_to_grid_ = TensorToGrid::kCustom;
  grid_dimension_ = 3;

  // ---- Weight upload in tiled [H][W][O/16][I/16][16o][16i] layout ----
  {
    const int O = attr.weights.shape.o;
    const int H = attr.weights.shape.h;
    const int W = attr.weights.shape.w;
    const int I = attr.weights.shape.i;

    // Upload weights in cooperative matrix tile order
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
    args_.AddObject("weights",
        std::make_unique<BufferDescriptor>(std::move(desc)));
  }

  BufferDescriptor bias_desc;
  bias_desc.element_type = definition.GetDataType();
  bias_desc.element_size = 4;
  bias_desc.memory_type = MemoryType::GLOBAL;
  const int padded_channels = AlignByN(attr.weights.shape.o, 4);
  bias_desc.size = padded_channels * SizeOf(bias_desc.element_type);
  bias_desc.data.resize(bias_desc.size);
  for (int i = 0; i < padded_channels; ++i) {
    const float value = i < attr.bias.shape.v ? attr.bias.data[i] : 0.0f;
    if (bias_desc.element_type == DataType::FLOAT32) {
      std::memcpy(bias_desc.data.data() + i * sizeof(float), &value,
                  sizeof(float));
    } else {
      const half half_value = static_cast<half>(value);
      std::memcpy(bias_desc.data.data() + i * sizeof(half), &half_value,
                  sizeof(half));
    }
  }
  args_.AddObject("biases",
                  std::make_unique<BufferDescriptor>(std::move(bias_desc)));
}

absl::Status ConvCoopMat::BindArguments(ArgumentsBinder* args) {
  RETURN_IF_ERROR(args->SetInt("output_channels", dst_[0]->Channels()));
  RETURN_IF_ERROR(args->SetInt("num_output_channel_groups",
                               DivideRoundUp(dst_[0]->Channels(), 16)));
  RETURN_IF_ERROR(args->SetInt("num_input_channel_groups",
                               DivideRoundUp(src_[0]->Channels(), 16)));
  return absl::OkStatus();
}

int3 ConvCoopMat::GetGridSize() const {
  // A workgroup covers 8x2 pixels and two waves of output-channel tiles.
  return int3(DivideRoundUp(dst_[0]->Width(), 8) * work_group_size_.x,
              DivideRoundUp(dst_[0]->Height(), 2) * work_group_size_.y,
              DivideRoundUp(DivideRoundUp(dst_[0]->Channels(), 16),
                            2 * out_g_batch_));
}

ConvCoopMat CreateConvCoopMat(const OperationDef& definition,
                            const Convolution2DAttributes& attr) {
  return ConvCoopMat(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
