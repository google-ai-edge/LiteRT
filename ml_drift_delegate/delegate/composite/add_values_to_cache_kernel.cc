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

#include "ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

namespace litert::ml_drift {
namespace {

class AddValuesToCacheOp : public ::ml_drift::GPUOperation {
 public:
  AddValuesToCacheOp() = default;
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(update_width_, batch_size_, slices_);
  }

  // Move only
  AddValuesToCacheOp(AddValuesToCacheOp&& operation) = default;
  AddValuesToCacheOp& operator=(AddValuesToCacheOp&& operation) = default;
  AddValuesToCacheOp(const AddValuesToCacheOp&) = delete;
  AddValuesToCacheOp& operator=(const AddValuesToCacheOp&) = delete;

  int update_width_ = 1;
  int batch_size_ = 1;
  int slices_ = 1;
};

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
BuildAddValuesToCache(const ::ml_drift::OperationDef& op_def,
                      const AddValuesToCacheAttributes& attr) {
  const auto& src_k = op_def.src_tensors[0];
  const auto& src_v = op_def.src_tensors[1];
  const auto& cache_k = op_def.dst_tensors[0];
  const auto& cache_v = op_def.dst_tensors[1];
  const auto src_shape = src_k.GetBHWDCShape();

  AddValuesToCacheOp custom_op;
  custom_op.AllowFuseInputReorder(true);

  // K update [B, H, T, D] maps to (w = T, c = D); [B, H, D, T] maps to
  // (w = D, c = T).
  const auto src_v_shape = src_v.GetBHWDCShape();
  const bool k_update_transposed = attr.k_update_ts_idx == 3;
  const bool v_update_transposed = attr.v_update_ts_idx == 3;
  const int k_tokens = k_update_transposed ? src_shape.c : src_shape.w;
  const int k_head_dim = k_update_transposed ? src_shape.w : src_shape.c;
  const int v_tokens = v_update_transposed ? src_v_shape.c : src_v_shape.w;
  const int v_head_dim = v_update_transposed ? src_v_shape.w : src_v_shape.c;
  int num_heads = attr.head_size > 0 ? (k_head_dim / attr.head_size) : 1;
  const auto is_float_dtype = [](::ml_drift::DataType dtype) {
    return dtype == ::ml_drift::DataType::kFloat32 ||
           dtype == ::ml_drift::DataType::kFloat16;
  };
  const bool is_float_src = is_float_dtype(src_k.GetDataType()) &&
                            is_float_dtype(src_v.GetDataType());
  const bool is_uint8_cache =
      cache_k.GetDataType() == ::ml_drift::DataType::kUint8 &&
      cache_v.GetDataType() == ::ml_drift::DataType::kUint8;
  if (attr.kv_quant_bits != 0 && attr.kv_quant_bits != 4 &&
      attr.kv_quant_bits != 8) {
    return absl::InvalidArgumentError(
        "AddValuesToCache: unsupported kv_quant_bits.");
  }
  if (cache_k.GetDataType() != cache_v.GetDataType()) {
    return absl::InvalidArgumentError(
        "AddValuesToCache: cache_k and cache_v data types must match.");
  }
  const float scale_k = attr.scale_k.value_or(1.0f);
  const float scale_v = attr.scale_v.value_or(1.0f);
  if (attr.kv_quant_bits == 4) {
    if (!is_float_src || !is_uint8_cache) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with INT4 KV cache requires float src_k/src_v and "
          "kUint8 cache_k/cache_v.");
    }
    if (attr.head_size <= 0 || attr.head_size % 8 != 0 ||
        attr.cache_size <= 0 || attr.cache_size % 4 != 0) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with INT4 KV cache requires head_size divisible by "
          "8 and cache_size divisible by 4.");
    }
    if (!attr.scale_k.has_value() || !std::isfinite(scale_k) ||
        scale_k <= 0.0f || !attr.scale_v.has_value() ||
        !std::isfinite(scale_v) || scale_v <= 0.0f) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with INT4 KV cache requires finite positive "
          "scale_k and scale_v.");
    }
  } else if (attr.kv_quant_bits == 8 || is_uint8_cache) {
    if (!is_float_src || !is_uint8_cache) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with INT8 KV cache requires float src_k/src_v and "
          "kUint8 cache_k/cache_v.");
    }
    if (!std::isfinite(scale_k) || scale_k <= 0.0f || !std::isfinite(scale_v) ||
        scale_v <= 0.0f) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with INT8 KV cache requires finite positive "
          "scale_k and scale_v.");
    }
  }
  const bool is_int4_cache = attr.kv_quant_bits == 4;
  const bool is_int8_cache = !is_int4_cache && is_uint8_cache;
  if (is_int4_cache || is_int8_cache) {
    if (cache_k.GetStorageType() != ::ml_drift::TensorStorageType::kBuffer ||
        cache_v.GetStorageType() != ::ml_drift::TensorStorageType::kBuffer) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with quantized KV cache requires kBuffer storage.");
    }
    if (attr.kv_cache_batch_size <= 0 || attr.head_size <= 0 ||
        attr.cache_size <= 0 || src_shape.b != 1 || src_v_shape.b != 1 ||
        src_shape.h <= 0 || src_shape.h != src_v_shape.h || k_tokens <= 0 ||
        k_tokens != v_tokens || k_head_dim <= 0 || k_head_dim != v_head_dim ||
        k_head_dim % attr.head_size != 0) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with quantized KV cache has invalid input shapes.");
    }
    const auto cache_k_shape = cache_k.GetBHWDCShape();
    const auto cache_v_shape = cache_v.GetBHWDCShape();
    const int expected_head_bytes =
        is_int4_cache ? (attr.head_size / 2) : attr.head_size;
    const int expected_k_w =
        attr.k_cache_ts_idx == 2 ? attr.cache_size : expected_head_bytes;
    const int expected_k_c =
        attr.k_cache_ts_idx == 2 ? expected_head_bytes : attr.cache_size;
    const int expected_v_w =
        attr.v_cache_ts_idx == 3 ? expected_head_bytes : attr.cache_size;
    const int expected_v_c =
        attr.v_cache_ts_idx == 3 ? attr.cache_size : expected_head_bytes;
    const int64_t expected_batch =
        static_cast<int64_t>(attr.kv_cache_batch_size) * std::max(1, num_heads);
    const int64_t k_batch =
        static_cast<int64_t>(cache_k_shape.b) * cache_k_shape.h;
    const int64_t v_batch =
        static_cast<int64_t>(cache_v_shape.b) * cache_v_shape.h;
    if (expected_batch <= 0 || expected_batch != k_batch ||
        expected_batch != v_batch || cache_k_shape.w != expected_k_w ||
        cache_k_shape.c != expected_k_c || cache_v_shape.w != expected_v_w ||
        cache_v_shape.c != expected_v_c) {
      return absl::InvalidArgumentError(
          "AddValuesToCache with quantized KV cache has mismatched cache "
          "tensor shapes.");
    }
  }

  custom_op.update_width_ = k_tokens;
  custom_op.batch_size_ = attr.kv_cache_batch_size * std::max(1, num_heads);
  // In INT4 mode, each work-item packs 8 head-dim elements (vec4 slices 2*S and
  // 2*S + 1) into a single uchar4 (4 bytes = 8 nibbles), avoiding any sub-word
  // read-modify-write races across tokens or threads.
  custom_op.slices_ =
      is_int4_cache ? (attr.head_size / 8) : ((attr.head_size + 3) / 4);

  custom_op.args_.AddInt("update_width", custom_op.update_width_);
  custom_op.args_.AddInt("batch_size", custom_op.batch_size_);
  custom_op.args_.AddInt("slices", custom_op.slices_);
  custom_op.args_.AddInt("src_height", src_shape.h);
  custom_op.args_.AddInt("cache_size", attr.cache_size);
  custom_op.args_.AddInt("head_size", attr.head_size);
  custom_op.AddSrcTensor("src_k", src_k);
  custom_op.AddSrcTensor("src_v", src_v);
  ::ml_drift::BufferDescriptor params_buffer;
  params_buffer.element_type = ::ml_drift::DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("cache_k", cache_k);
  custom_op.AddDstTensor("cache_v", cache_v);

  bool is_ring_buffer = attr.is_ring_buffer.value_or(false);

  // Quantized cache cases: src_k and src_v are float and quantized in-shader.
  if (is_int4_cache) {
    constexpr float kInt4Min = -8.0f;
    constexpr float kInt4Max = 7.0f;
    custom_op.args_.AddFloat("min_k", kInt4Min * scale_k,
                             ::ml_drift::DataType::kFloat32);
    custom_op.args_.AddFloat("max_k", kInt4Max * scale_k,
                             ::ml_drift::DataType::kFloat32);
    custom_op.args_.AddFloat("inverse_scale_k", 1.0f / scale_k,
                             ::ml_drift::DataType::kFloat32);
    custom_op.args_.AddFloat("min_v", kInt4Min * scale_v,
                             ::ml_drift::DataType::kFloat32);
    custom_op.args_.AddFloat("max_v", kInt4Max * scale_v,
                             ::ml_drift::DataType::kFloat32);
    custom_op.args_.AddFloat("inverse_scale_v", 1.0f / scale_v,
                             ::ml_drift::DataType::kFloat32);
  } else if (is_int8_cache) {
    float max_value_k = static_cast<float>(INT8_MAX) * scale_k;
    float min_value_k = static_cast<float>(INT8_MIN) * scale_k;
    float inverse_scale_k = 1.0f / scale_k;
    custom_op.args_.AddFloat("min_k", min_value_k, src_k.GetDataType());
    custom_op.args_.AddFloat("max_k", max_value_k, src_k.GetDataType());
    custom_op.args_.AddFloat("inverse_scale_k", inverse_scale_k,
                             src_k.GetDataType());
    float max_value_v = static_cast<float>(INT8_MAX) * scale_v;
    float min_value_v = static_cast<float>(INT8_MIN) * scale_v;
    float inverse_scale_v = 1.0f / scale_v;
    custom_op.args_.AddFloat("min_v", min_value_v, src_v.GetDataType());
    custom_op.args_.AddFloat("max_v", max_value_v, src_v.GetDataType());
    custom_op.args_.AddFloat("inverse_scale_v", inverse_scale_v,
                             src_v.GetDataType());
  }

  std::string op_code = absl::Substitute(
      R"(
MAIN_FUNCTION($$0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.update_width || Y >= args.batch_size || S >= args.slices) {
    return;
  }
  int token_index_offset = args.params.Read($0);
  int active_tokens = args.params.Read($1);
  )",
      attr.token_index_offset_param_idx, attr.active_tokens_param_idx);
  if (is_ring_buffer) {
    op_code += R"(
  int update_length = args.params.Read(3);
  if (X >= update_length) {
    return;
  }
  int token_index = (token_index_offset + X) % args.cache_size;
  )";
  } else {
    op_code += R"(
  int token_index = token_index_offset + X;
  if (token_index >= args.cache_size || token_index >= active_tokens) {
    return;
  }
  )";
  }

  op_code += R"(
  int src_y = Y % args.src_height;  // broadcast Height dim used as Batch
  )";

  if (is_int4_cache) {
    auto emit_int4_read_and_pack = [&op_code](const std::string& name,
                                              int update_ts_idx) {
      if (update_ts_idx == 3) {
        // Update [B, H, D, T]: gather 8 head-dim values of token X.
        for (int i = 0; i < 8; ++i) {
          op_code += absl::Substitute(R"(
  float $0_$1 = 0.0f;
  if (S * 8 + $1 < args.head_size) {
    args.src_$0.ReadPerChannel<float>($0_$1, S * 8 + $1, src_y, X);
  }
  )",
                                      name, i);
        }
        op_code += absl::Substitute(R"(
  float4 value_$0_0 = ucl::Init<float4>($0_0, $0_1, $0_2, $0_3);
  float4 value_$0_1 = ucl::Init<float4>($0_4, $0_5, $0_6, $0_7);
  )",
                                    name);
      } else {
        // Update [B, H, T, D]: read two consecutive vec4 slices 2*S and 2*S+1.
        op_code += absl::Substitute(R"(
  float4 value_$0_0 = ucl::Convert<float4>(args.src_$0.Read(X, src_y, S * 2));
  float4 value_$0_1 = ucl::Convert<float4>(args.src_$0.Read(X, src_y, S * 2 + 1));
  )",
                                    name);
      }
      // Symmetric INT4 [-8, 7] quantized in FP32 and biased by +8 into [0, 15].
      // Low nibble stores slice 2*S (channels 8*S + 0..3); high nibble stores
      // slice 2*S + 1 (channels 8*S + 4..7), so downstream SDPA kernels unpack
      // two ready-to-dot float4 slices from one uchar4 with zero swizzling.
      op_code += absl::Substitute(R"(
  float4 clamped_$0_0 = min(ucl::Init<float4>(args.max_$0), max(ucl::Init<float4>(args.min_$0), value_$0_0));
  float4 clamped_$0_1 = min(ucl::Init<float4>(args.max_$0), max(ucl::Init<float4>(args.min_$0), value_$0_1));
  float4 q_$0_0 = min(ucl::Init<float4>(15.0f), max(ucl::Init<float4>(0.0f), round(clamped_$0_0 * ucl::Init<float4>(args.inverse_scale_$0)) + ucl::Init<float4>(8.0f)));
  float4 q_$0_1 = min(ucl::Init<float4>(15.0f), max(ucl::Init<float4>(0.0f), round(clamped_$0_1 * ucl::Init<float4>(args.inverse_scale_$0)) + ucl::Init<float4>(8.0f)));
  uint4 u_$0_0 = ucl::Convert<uint4>(q_$0_0);
  uint4 u_$0_1 = ucl::Convert<uint4>(q_$0_1);
  uchar4 final_value_$0 = ucl::Convert<uchar4>(u_$0_0 + u_$0_1 * ucl::Init<uint4>(16u));
  )",
                                  name);
    };
    emit_int4_read_and_pack("k", attr.k_update_ts_idx);
    emit_int4_read_and_pack("v", attr.v_update_ts_idx);
  } else {
    // Emits the read of one head-dim slice of token X from `name`.
    auto emit_read = [&op_code](const std::string& name, int update_ts_idx) {
      if (update_ts_idx == 3) {
        // Update [B, H, D, T]: width is the head dim and channels are tokens,
        // so gather the 4 head-dim values of token X.
        for (int i = 0; i < 4; ++i) {
          op_code += absl::Substitute(R"(
  float $0_$1 = 0.0f;
  if (S * 4 + $1 < args.head_size) {
    args.src_$0.ReadPerChannel<float>($0_$1, S * 4 + $1, src_y, X);
  }
  )",
                                      name, i);
        }
        op_code += absl::Substitute(R"(
  args.src_$0::type value_$0 = ucl::Convert<args.src_$0::type>(
      ucl::Init<float4>($0_0, $0_1, $0_2, $0_3));
  )",
                                    name);
      } else {
        // Update [B, H, T, D], the historical layout.
        op_code += absl::Substitute(R"(
  args.src_$0::type value_$0 = args.src_$0.Read(X, src_y, S);
  )",
                                    name);
      }
    };
    emit_read("k", attr.k_update_ts_idx);
    emit_read("v", attr.v_update_ts_idx);

    if (is_int8_cache) {
      const std::string src_k_type = ToUclDataType(src_k.GetDataType(), 4);
      const std::string src_v_type = ToUclDataType(src_v.GetDataType(), 4);
      op_code += absl::Substitute(R"(
  // quantize the slices
  $0 clamped_value_k = min(ucl::Init<$0>(args.max_k), max(ucl::Init<$0>(args.min_k), value_k));
  $0 quantized_value_k = round((clamped_value_k - ucl::Init<$0>(args.min_k)) * ucl::Init<$0>(args.inverse_scale_k));
  uchar4 final_value_k = ucl::Convert<uchar4>(quantized_value_k);

  $1 clamped_value_v = min(ucl::Init<$1>(args.max_v), max(ucl::Init<$1>(args.min_v), value_v));
  $1 quantized_value_v = round((clamped_value_v - ucl::Init<$1>(args.min_v)) * ucl::Init<$1>(args.inverse_scale_v));
  uchar4 final_value_v = ucl::Convert<uchar4>(quantized_value_v);
  )",
                                  src_k_type, src_v_type);
    } else {
      op_code += R"(
  args.src_k::type final_value_k = value_k;
  args.src_v::type final_value_v = value_v;
  )";
    }
  }
  op_code += R"(
  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    // (or head_size / 2 packed bytes for INT4, so i_slices == args.slices).
    // OGroup = o_slices -> actual layout SpatialIOGroupO4I4
    // SpatialIOGroupO4I4 Spatial ISlice OSlice O4 I4
    // index of vec4(I4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + O4Index
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i_slice = S;
    int k_sp = Y;
    int i_slices = args.slices;
    int o_slices = (args.cache_size + 3) / 4;
    int k_index = ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    args.cache_k.WriteLinear(final_value_k, k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    // (or head_size / 2 packed bytes for INT4, so o_slices == args.slices).
    // OGroup = o_slices -> actual layout SpatialIOGroupI4O4
    // SpatialIOI4O4 Spatial ISlice OSlice I4 O4
    // index of vec4(O4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + I4Index
    int v_o_slice = S;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int i_slices = (args.cache_size + 3) / 4;
    int o_slices = args.slices;
    int v_index = ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.cache_v.WriteLinear(final_value_v, v_index);
  }
})";

  custom_op.code_ = std::move(op_code);
  return std::make_unique<AddValuesToCacheOp>(std::move(custom_op));
}

}  // namespace

std::unique_ptr<::ml_drift::GPUOperation> CreateAddValuesToCache(
    const ::ml_drift::OperationDef& op_def,
    const AddValuesToCacheAttributes& attr) {
  absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>> op =
      BuildAddValuesToCache(op_def, attr);
  ABSL_CHECK_OK(op.status());
  return *std::move(op);
}

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateAddValuesToCacheFromNode(const ::ml_drift::OperationDef& op_def,
                               const ::ml_drift::Node& node) {
  if (op_def.src_tensors.size() != 3 || op_def.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "AddValuesToCache operation expects 3 inputs and 2 outputs.");
  }

  const auto& attr = std::any_cast<const AddValuesToCacheAttributes&>(
      node.operation.attributes);
  return BuildAddValuesToCache(op_def, attr);
}

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateAddValuesToCacheFromNode(const ::ml_drift::OperationDef& op_def,
                               const ::ml_drift::ir::IrOp& ir_op) {
  if (op_def.src_tensors.size() != 3 || op_def.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "AddValuesToCache operation expects 3 inputs and 2 outputs.");
  }

  const auto& attr =
      std::any_cast<const AddValuesToCacheAttributes&>(ir_op.attr);
  return BuildAddValuesToCache(op_def, attr);
}

}  // namespace litert::ml_drift
