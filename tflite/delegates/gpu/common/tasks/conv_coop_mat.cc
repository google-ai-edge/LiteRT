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
#include <cstdint>
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
    const OperationDef& definition, int out_g_batch) {
  const int output_tiles = std::max(1, std::min(4, out_g_batch));
  const std::string type =
      definition.precision == CalculationsPrecision::F32 ? "float" : "half";
  auto S = [](int value) { return std::to_string(value); };
  std::string code;
  if (definition.precision == CalculationsPrecision::F16) {
    code += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }
  code += "#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable\n";
  code += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
  for (const auto& usage : {std::string("A"), std::string("B"),
                            std::string("ACCUMULATOR")}) {
    const std::string name = usage == "ACCUMULATOR" ? "C" : usage;
    code += "typedef " + type + " mat" + name +
            "_t __attribute__((coop_mat("
            "CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP, 16, 16, "
            "CLK_COOPERATIVE_MATRIX_" + usage + ")));\n";
  }
  code += "__attribute__((reqd_work_group_size(8, 8, 1)))\n";
  code += "MAIN_FUNCTION($0) {\n";
  code += R"(
  // Batch one uses coordinate zero. Avoid resource-name substitution of batch.
  args.src_tensor.SetBatchRef(0);
  args.dst_tensor.SetBatchRef(0);
  const uint subgroup_id = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint subgroup_size = get_sub_group_size();
  const uint subgroup_count = get_num_sub_groups();
)";
  // Two logical channel batches keep the host grid independent of the actual
  // subgroup width. Storage belongs to a logical batch, not a physical wave.
  code += "  __local " + type +
          " tileShared[512] __attribute__((aligned(16)));\n";
  code += "  __global const " + type + "* packed_weights = (__global const " +
          type + "*)weights_buffer;\n";
  code += R"(
  const int2 tile_origin = (int2)(get_group_id(0) * 8,
                                  get_group_id(1) * 2);
  const int2 dst_size = (int2)(args.dst_tensor.Width(), args.dst_tensor.Height());
  const int2 src_size = (int2)(args.src_tensor.Width(), args.src_tensor.Height());
  const int input_groups = args.num_input_channel_groups;
  const int output_groups = args.num_output_channel_groups;
  // Every subgroup executes its own sequence of complete collective operations.
  // Wave32: IDs 0 and 1 execute one batch each. Wave64: ID 0 executes both.
  for (uint slot = subgroup_id; slot < 2; slot += subgroup_count) {
)";
  code += "    const int output_base = ((int)get_group_id(2) * 2 + (int)slot) * " +
          S(output_tiles) + ";\n";
  code += R"(
    if (output_base >= output_groups) continue;
)";
  code += "    __local " + type + "* staging = tileShared + slot * 256;\n";
  code += "    matA_t a;\n    matB_t b;\n";
  for (int i = 0; i < output_tiles; ++i) {
    code += "    matC_t accum" + S(i) + " = coop_mat_init(0.0f);\n";
  }
  code += R"(
    for (int ky = 0; ky < args.kernel_size_y; ++ky) {
      for (int kx = 0; kx < args.kernel_size_x; ++kx) {
        const int weight_position =
            (ky * args.kernel_size_x + kx) * output_groups * input_groups * 256;
        for (int input_group = 0; input_group < input_groups; ++input_group) {
          // A has 64 four-element vectors. Striding by the actual subgroup size
          // gives two vectors per lane for wave32 and one for wave64.
          for (uint v = lane; v < 64; v += subgroup_size) {
            const int pixel = (int)(v / 4);
            const int input_slice = input_group * 4 + (int)(v % 4);
            const int2 out_pos = tile_origin + (int2)(pixel % 8, pixel / 8);
            const int2 in_pos = out_pos * (int2)(args.stride_x, args.stride_y) -
                (int2)(args.padding_x, args.padding_y) +
                (int2)(kx * args.dilation_x, ky * args.dilation_y);
)";
  code += "            " + type + "4 value = (" + type + "4)(0.0f);\n";
  code += R"(
            if (all(out_pos < dst_size) && all(in_pos >= (int2)(0, 0)) &&
                all(in_pos < src_size)) {
              value = args.src_tensor.Read(in_pos.x, in_pos.y, input_slice);
            }
            vstore4(value, 0, staging + v * 4);
          }
          sub_group_barrier(CLK_LOCAL_MEM_FENCE);
          a = coop_mat_load(staging, CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR, 16);
          sub_group_barrier(CLK_LOCAL_MEM_FENCE);
)";
  for (int i = 0; i < output_tiles; ++i) {
    code += "          b = coop_mat_load(packed_weights + weight_position + "
            "(min(output_base + " + S(i) +
            ", output_groups - 1) * input_groups + input_group) * 256, "
            "CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR, 16);\n";
    code += "          accum" + S(i) + " = coop_mat_mulAdd(a, b, accum" +
            S(i) + ");\n";
  }
  code += "        }\n      }\n    }\n";
  for (int i = 0; i < output_tiles; ++i) {
    code += "    if (output_base + " + S(i) + " < output_groups) {\n";
    code += "      sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n";
    code += "      coop_mat_store(staging, accum" + S(i) +
            ", CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR, 16);\n";
    code += "      sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n";
    code += R"(
      for (uint v = lane; v < 64; v += subgroup_size) {
        const int pixel = (int)(v / 4);
)";
    code += "        const int output_slice = (output_base + " + S(i) +
            ") * 4 + (int)(v % 4);\n";
    code += R"(
        const int2 out_pos = tile_origin + (int2)(pixel % 8, pixel / 8);
        if (all(out_pos < dst_size) && output_slice * 4 < args.output_channels) {
)";
    code += "          " + type +
            "4 value = vload4(0, staging + v * 4) + args.biases.Read(output_slice);\n";
    code += R"(
          args.dst_tensor.Write(value, out_pos.x, out_pos.y, output_slice);
        }
      }
    }
)";
  }
  code += "  }\n}\n";
  return code;
}


// Pack OHWI weights as [H][W][output tile][input tile][16 outputs][16 inputs].
// Column-major B loads then address each element as input + output * 16.
template <typename T>
std::vector<T> RearrangeWeightsForCoopMat(
    const std::vector<float>& weights_data, int output_channels,
    int kernel_height, int kernel_width, int input_channels) {
  const int input_tiles = DivideRoundUp(input_channels, 16);
  const int output_tiles = DivideRoundUp(output_channels, 16);
  const size_t elements = static_cast<size_t>(kernel_height) * kernel_width *
                          output_tiles * input_tiles * 256;
  std::vector<T> result(elements, T(0.0f));
  for (int o = 0; o < output_channels; ++o) {
    for (int h = 0; h < kernel_height; ++h) {
      for (int w = 0; w < kernel_width; ++w) {
        for (int i = 0; i < input_channels; ++i) {
          const size_t src_index =
              ((static_cast<size_t>(o) * kernel_height + h) * kernel_width + w) *
                  input_channels + i;
          const size_t dst_index =
              ((((static_cast<size_t>(h) * kernel_width + w) * output_tiles +
                 o / 16) * input_tiles + i / 16) * 256) +
              (o % 16) * 16 + i % 16;
          result[dst_index] = static_cast<T>(weights_data[src_index]);
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
  // Matrix capability discovery is independent of the physical lane mapping.
  // No explicit subgroup width is requested: use the queried default variants.
  if (!gpu_info.IsApiOpenCl() ||
      !gpu_info.SupportsExtension("cl_khr_cooperative_matrix") ||
      !gpu_info.SupportsExtension("cl_khr_subgroups")) {
    return false;
  }
  // The kernel supports F16 (half/half4) and F32 (float/float4) tensor types.
  if (definition.precision != CalculationsPrecision::F16 &&
      definition.precision != CalculationsPrecision::F32) {
    return false;
  }

  const DataType type = definition.GetDataType();
  const auto& cl_info = gpu_info.opencl_info;
  if (!cl_info.SupportsCooperativeMatrix(
          {/*m=*/16, /*n=*/16, /*k=*/16, type, type, type, type,
           /*saturating_accumulation=*/false})) {
    return false;
  }
  // Local arrays explicitly use 16-byte alignment; matrix origins are spaced
  // by 256 elements. Global weight buffers must meet the same requirement.
  const auto alignment = cl_info.cooperative_matrix_pointer_alignment;
  const auto stride_multiple = cl_info.cooperative_matrix_stride_multiple;
  if (alignment == 0 || 16 % alignment != 0 || stride_multiple == 0 ||
      (16 * SizeOf(type)) % stride_multiple != 0 ||
      cl_info.base_addr_align_in_bits < 8ULL * alignment ||
      (definition.precision == CalculationsPrecision::F16 &&
       !gpu_info.SupportsExtension("cl_khr_fp16"))) {
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
  if (definition.src_tensors[0].GetDataType() != type ||
      definition.dst_tensors[0].GetDataType() != type) {
    return false;
  }
  // Input channels must fill the K=16 tiles. cin=3 padded to 16 wastes 81%
  // of the MACs on zero-padding.
  if (attr.weights.shape.i <= 0 || attr.weights.shape.i % 16 != 0 ||
      attr.weights.shape.o <= 0 || dst_shape.c != attr.weights.shape.o ||
      attr.weights.shape.h <= 0 || attr.weights.shape.w <= 0 ||
      attr.strides.h <= 0 || attr.strides.w <= 0 ||
      attr.dilations.h <= 0 || attr.dilations.w <= 0) {
    return false;
  }
  return true;
}

// Conservative initial routing thresholds; performance must be measured on
// each supported device. Avoid shapes with limited matrix reuse or parallelism:
//  (a) fewer than 64 output channels
//  (b) fewer than four K iterations
//  (c) pointwise convolutions with fewer than 24 workgroups
//  (d) any convolution with fewer than eight workgroups
//  (e) pointwise channel reduction with an input:output group ratio above eight
bool IsGoodTaskSizeForConvCoopMat(const Convolution2DAttributes& attr,
                               const BHWC& dst_shape) {
  const int cin = attr.weights.shape.i;
  const int cout = attr.weights.shape.o;
  const int kh = attr.weights.shape.h;
  const int kw = attr.weights.shape.w;
  if (cin <= 0 || cout <= 0 || kh <= 0 || kw <= 0 ||
      dst_shape.w <= 0 || dst_shape.h <= 0) {
    return false;
  }
  const int cin_groups = DivideRoundUp(cin, 16);
  const int cout_groups = DivideRoundUp(cout, 16);
  const int64_t k_iters = int64_t{kh} * kw * cin_groups;
  const int out_g_batch = std::max(1, std::min(4, DivideRoundUp(cout_groups, 2)));
  const int64_t work_groups = int64_t{DivideRoundUp(dst_shape.w, 8)} *
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
  return !(is_pointwise && cin_groups > int64_t{8} * cout_groups);
}

ConvCoopMat::ConvCoopMat(const OperationDef& definition,
                       const Convolution2DAttributes& attr)
    : GPUOperation(definition),
      input_channels_(attr.weights.shape.i),
      output_channels_(attr.weights.shape.o),
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
      definition, out_g_batch_);
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
      std::vector<float> weight_data_f32 = RearrangeWeightsForCoopMat<float>(
          attr.weights.data, O, H, W, I);
      desc.element_type = DataType::FLOAT32;
      desc.size = weight_data_f32.size() * sizeof(float);
      desc.data.resize(desc.size);
      std::memcpy(desc.data.data(), weight_data_f32.data(), desc.size);
    } else {
      std::vector<half> weight_data = RearrangeWeightsForCoopMat<half>(
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
  // Packed weights fix the channel dimensions even when spatial sizes change.
  if (src_[0]->Batch() != 1 || dst_[0]->Batch() != 1 ||
      src_[0]->Channels() != input_channels_ ||
      dst_[0]->Channels() != output_channels_) {
    return absl::InvalidArgumentError(
        "ConvCoopMat requires batch 1 and the original channel dimensions");
  }
  RETURN_IF_ERROR(args->SetInt("output_channels", dst_[0]->Channels()));
  RETURN_IF_ERROR(args->SetInt("num_output_channel_groups",
                               DivideRoundUp(dst_[0]->Channels(), 16)));
  RETURN_IF_ERROR(args->SetInt("num_input_channel_groups",
                               DivideRoundUp(src_[0]->Channels(), 16)));
  return absl::OkStatus();
}

int3 ConvCoopMat::GetGridSize() const {
  // Two logical channel batches per workgroup, regardless of the actual number
  // of physical subgroups. This also keeps restored/cached dispatches unchanged.
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
