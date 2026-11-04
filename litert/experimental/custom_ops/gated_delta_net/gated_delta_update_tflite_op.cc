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

#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/mutable_op_resolver.h"

namespace litert_torch {
namespace gdn_kernels {
namespace {

std::optional<::litert::gated_delta_net::GatedDeltaBlockParams>
ParseBlockParams(const TfLiteNode* node) {
  if (node->custom_initial_data == nullptr ||
      node->custom_initial_data_size <= 0) {
    return std::nullopt;
  }
  const flexbuffers::Map fb_map =
      flexbuffers::GetRoot(
          reinterpret_cast<const uint8_t*>(node->custom_initial_data),
          node->custom_initial_data_size)
          .AsMap();
  if (fb_map["num_k_heads"].IsNull() || fb_map["num_k_heads"].AsInt32() <= 0) {
    return std::nullopt;
  }
  ::litert::gated_delta_net::GatedDeltaBlockParams params;
  params.num_k_heads = fb_map["num_k_heads"].AsInt32();
  if (!fb_map["num_v_heads"].IsNull()) {
    params.num_v_heads = fb_map["num_v_heads"].AsInt32();
  }
  if (!fb_map["head_k_dim"].IsNull()) {
    params.head_k_dim = fb_map["head_k_dim"].AsInt32();
  }
  if (!fb_map["head_v_dim"].IsNull()) {
    params.head_v_dim = fb_map["head_v_dim"].AsInt32();
  }
  if (!fb_map["eps"].IsNull()) {
    params.eps = fb_map["eps"].AsFloat();
  }
  if (!fb_map["norm_eps"].IsNull()) {
    params.norm_eps = fb_map["norm_eps"].AsFloat();
  }
  return params;
}

}  // namespace

// ============================================================================
// 1. Lower Triangular Inversion Kernel (gdn_tril_inv)
// ============================================================================

TfLiteStatus PrepareTrilInv(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus EvalTrilInv(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  const float* in_data = input->data.f;
  float* out_data = output->data.f;
  const int dims_count = input->dims->size;
  TF_LITE_ENSURE(context, dims_count >= 2);
  const int C = input->dims->data[dims_count - 1];
  TF_LITE_ENSURE_EQ(context, input->dims->data[dims_count - 2], C);

  const int total_elements = tflite::GetTensorShape(input).FlatSize();
  ::litert::gated_delta_net::ComputeTrilInv(in_data, out_data, total_elements,
                                            C);
  return kTfLiteOk;
}

TfLiteRegistration* GetTrilInvRegistration() {
  static TfLiteRegistration reg = {nullptr, nullptr, PrepareTrilInv,
                                   EvalTrilInv};
  return &reg;
}

// ============================================================================
// 2. GatedDeltaUpdate Kernel (gated_delta_update)
// ============================================================================

TfLiteStatus PrepareGatedDeltaUpdate(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size == 6 || node->inputs->size == 7);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  if (const auto block = ParseBlockParams(node); block.has_value()) {
    TF_LITE_ENSURE(context, block->num_k_heads > 0 &&
                                block->num_v_heads >= block->num_k_heads &&
                                block->num_v_heads % block->num_k_heads == 0 &&
                                block->head_k_dim > 0 && block->head_v_dim > 0);
    const TfLiteTensor* conv_out = tflite::GetInput(context, node, 0);
    const TfLiteTensor* zba = tflite::GetInput(context, node, 1);
    const TfLiteTensor* a_log = tflite::GetInput(context, node, 2);
    const TfLiteTensor* dt_bias = tflite::GetInput(context, node, 3);
    const TfLiteTensor* norm_weight = tflite::GetInput(context, node, 4);
    const TfLiteTensor* rec_state = tflite::GetInput(context, node, 5);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);
    TfLiteTensor* new_rec = tflite::GetOutput(context, node, 1);

    TF_LITE_ENSURE_EQ(context, conv_out->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, zba->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, a_log->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, dt_bias->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, norm_weight->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, rec_state->type, kTfLiteFloat32);

    const int K_dim = block->num_k_heads * block->head_k_dim;
    const int V_dim = block->num_v_heads * block->head_v_dim;
    const int conv_dim = 2 * K_dim + V_dim;
    const int zba_dim = V_dim + 2 * block->num_v_heads;
    TF_LITE_ENSURE_EQ(context, conv_out->dims->size, 3);
    const int B = conv_out->dims->data[0];
    const int T = conv_out->dims->data[1];
    TF_LITE_ENSURE_EQ(context, conv_out->dims->data[2], conv_dim);
    TF_LITE_ENSURE_EQ(context, zba->dims->size, 3);
    TF_LITE_ENSURE_EQ(context, zba->dims->data[0], B);
    TF_LITE_ENSURE_EQ(context, zba->dims->data[1], T);
    TF_LITE_ENSURE(context, zba->dims->data[2] == zba_dim ||
                                zba->dims->data[2] == conv_dim + zba_dim);
    TF_LITE_ENSURE_EQ(context, tflite::GetTensorShape(a_log).FlatSize(),
                      block->num_v_heads);
    TF_LITE_ENSURE_EQ(context, tflite::GetTensorShape(dt_bias).FlatSize(),
                      block->num_v_heads);
    TF_LITE_ENSURE_EQ(context, tflite::GetTensorShape(norm_weight).FlatSize(),
                      block->head_v_dim);
    TF_LITE_ENSURE_EQ(context, rec_state->dims->size, 4);
    TF_LITE_ENSURE_EQ(context, rec_state->dims->data[0], B);
    TF_LITE_ENSURE_EQ(context, rec_state->dims->data[1], block->num_v_heads);
    TF_LITE_ENSURE_EQ(context, rec_state->dims->data[2], block->head_k_dim);
    TF_LITE_ENSURE_EQ(context, rec_state->dims->data[3], block->head_v_dim);

    if (node->inputs->size == 7) {
      const TfLiteTensor* valid_mask = tflite::GetInput(context, node, 6);
      TF_LITE_ENSURE(context, valid_mask->type == kTfLiteFloat32 ||
                                  valid_mask->type == kTfLiteInt32 ||
                                  valid_mask->type == kTfLiteBool);
      TF_LITE_ENSURE_EQ(context, tflite::GetTensorShape(valid_mask).FlatSize(),
                        B * T);
    }

    TfLiteIntArray* out_shape0 = TfLiteIntArrayCreate(3);
    out_shape0->data[0] = B;
    out_shape0->data[1] = T;
    out_shape0->data[2] = V_dim;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, out_shape0));

    TfLiteIntArray* out_shape1 = TfLiteIntArrayCopy(rec_state->dims);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, new_rec, out_shape1));
    return kTfLiteOk;
  }

  const TfLiteTensor* q_t = tflite::GetInput(context, node, 0);
  const TfLiteTensor* k_t = tflite::GetInput(context, node, 1);
  const TfLiteTensor* v_t = tflite::GetInput(context, node, 2);
  const TfLiteTensor* rec_state = tflite::GetInput(context, node, 5);
  TfLiteTensor* core_out = tflite::GetOutput(context, node, 0);
  TfLiteTensor* new_rec = tflite::GetOutput(context, node, 1);

  TF_LITE_ENSURE_EQ(context, q_t->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, k_t->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, v_t->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, rec_state->type, kTfLiteFloat32);
  if (node->inputs->size == 7) {
    const TfLiteTensor* valid_len = tflite::GetInput(context, node, 6);
    TF_LITE_ENSURE(context, valid_len->type == kTfLiteInt32 ||
                                valid_len->type == kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, tflite::GetTensorShape(valid_len).FlatSize(),
                      v_t->dims->data[0]);
  }

  TfLiteIntArray* out_shape0 = TfLiteIntArrayCopy(v_t->dims);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, core_out, out_shape0));

  TfLiteIntArray* out_shape1 = TfLiteIntArrayCopy(rec_state->dims);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, new_rec, out_shape1));

  return kTfLiteOk;
}

TfLiteStatus EvalGatedDeltaUpdate(TfLiteContext* context, TfLiteNode* node) {
  if (const auto block = ParseBlockParams(node); block.has_value()) {
    const TfLiteTensor* conv_out = tflite::GetInput(context, node, 0);
    const TfLiteTensor* zba = tflite::GetInput(context, node, 1);
    const TfLiteTensor* a_log = tflite::GetInput(context, node, 2);
    const TfLiteTensor* dt_bias = tflite::GetInput(context, node, 3);
    const TfLiteTensor* norm_weight = tflite::GetInput(context, node, 4);
    const TfLiteTensor* rec_state = tflite::GetInput(context, node, 5);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);
    TfLiteTensor* new_rec = tflite::GetOutput(context, node, 1);

    const int B = conv_out->dims->data[0];
    const int T = conv_out->dims->data[1];
    std::vector<float> mask_vec;
    const float* mask_ptr = nullptr;
    if (node->inputs->size == 7) {
      const TfLiteTensor* mask_tensor = tflite::GetInput(context, node, 6);
      if (mask_tensor->type == kTfLiteFloat32) {
        mask_ptr = mask_tensor->data.f;
      } else {
        mask_vec.resize(B * T);
        if (mask_tensor->type == kTfLiteInt32) {
          for (int i = 0; i < B * T; ++i) {
            mask_vec[i] = static_cast<float>(mask_tensor->data.i32[i]);
          }
        } else {
          for (int i = 0; i < B * T; ++i) {
            mask_vec[i] = mask_tensor->data.b[i] ? 1.0f : 0.0f;
          }
        }
        mask_ptr = mask_vec.data();
      }
    }

    ::litert::gated_delta_net::ComputeGatedDeltaUpdateBlock(
        conv_out->data.f, zba->data.f, a_log->data.f, dt_bias->data.f,
        norm_weight->data.f, rec_state->data.f, mask_ptr, output->data.f,
        new_rec->data.f, B, T, *block, zba->dims->data[2]);
    return kTfLiteOk;
  }

  const TfLiteTensor* q_t = tflite::GetInput(context, node, 0);
  const TfLiteTensor* k_t = tflite::GetInput(context, node, 1);
  const TfLiteTensor* v_t = tflite::GetInput(context, node, 2);
  const TfLiteTensor* beta_t = tflite::GetInput(context, node, 3);
  const TfLiteTensor* g_t = tflite::GetInput(context, node, 4);
  const TfLiteTensor* rec_state = tflite::GetInput(context, node, 5);
  TfLiteTensor* core_out = tflite::GetOutput(context, node, 0);
  TfLiteTensor* new_rec = tflite::GetOutput(context, node, 1);

  TF_LITE_ENSURE(context, q_t->dims->size >= 4);
  const int B = v_t->dims->data[0];
  const int H = v_t->dims->data[1];
  const int H_k = q_t->dims->data[1];
  const int N = v_t->dims->data[2];
  const int D_k = q_t->dims->data[3];
  const int D_v = v_t->dims->data[3];

  std::vector<int> valid_len_vec;
  const int* valid_len_ptr = nullptr;
  if (node->inputs->size == 7) {
    const TfLiteTensor* valid_len_tensor = tflite::GetInput(context, node, 6);
    const int num_elems = tflite::GetTensorShape(valid_len_tensor).FlatSize();
    TF_LITE_ENSURE_EQ(context, num_elems, B);
    valid_len_vec.resize(B);
    for (int b = 0; b < B; ++b) {
      valid_len_vec[b] = (valid_len_tensor->type == kTfLiteInt32)
                             ? valid_len_tensor->data.i32[b]
                             : static_cast<int>(valid_len_tensor->data.f[b]);
    }
    valid_len_ptr = valid_len_vec.data();
  }

  // Dispatch to only recurrent implementation for now
  ::litert::gated_delta_net::ComputeGatedDeltaUpdateRecurrent(
      q_t->data.f, k_t->data.f, v_t->data.f, beta_t->data.f, g_t->data.f,
      rec_state->data.f, core_out->data.f, new_rec->data.f, B, H, N, D_k, D_v,
      H_k, valid_len_ptr);
  return kTfLiteOk;
}

TfLiteRegistration* GetGatedDeltaUpdateRegistration() {
  static TfLiteRegistration reg = {nullptr, nullptr, PrepareGatedDeltaUpdate,
                                   EvalGatedDeltaUpdate};
  return &reg;
}

void TfLiteRegisterer(tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("custom_call.gdn_tril_inv", GetTrilInvRegistration());
  resolver->AddCustom("gdn_tril_inv", GetTrilInvRegistration());
  resolver->AddCustom("custom_call.gated_delta_update",
                      GetGatedDeltaUpdateRegistration());
  resolver->AddCustom("gated_delta_update", GetGatedDeltaUpdateRegistration());
}

}  // namespace gdn_kernels
}  // namespace litert_torch
