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

#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/mutable_op_resolver.h"

namespace litert_torch {
namespace gdn_kernels {

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
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 6);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);
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

  TfLiteIntArray* out_shape0 = TfLiteIntArrayCopy(v_t->dims);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, core_out, out_shape0));

  TfLiteIntArray* out_shape1 = TfLiteIntArrayCopy(rec_state->dims);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, new_rec, out_shape1));

  return kTfLiteOk;
}

TfLiteStatus EvalGatedDeltaUpdate(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* q_t = tflite::GetInput(context, node, 0);
  const TfLiteTensor* k_t = tflite::GetInput(context, node, 1);
  const TfLiteTensor* v_t = tflite::GetInput(context, node, 2);
  const TfLiteTensor* beta_t = tflite::GetInput(context, node, 3);
  const TfLiteTensor* g_t = tflite::GetInput(context, node, 4);
  const TfLiteTensor* rec_state = tflite::GetInput(context, node, 5);
  TfLiteTensor* core_out = tflite::GetOutput(context, node, 0);
  TfLiteTensor* new_rec = tflite::GetOutput(context, node, 1);

  TF_LITE_ENSURE(context, q_t->dims->size >= 4);
  const int B = q_t->dims->data[0];
  const int H = q_t->dims->data[1];
  const int N = q_t->dims->data[2];
  const int D_k = q_t->dims->data[3];
  const int D_v = v_t->dims->data[3];

  // Dispatch to only recurrent implementation for now
  ::litert::gated_delta_net::ComputeGatedDeltaUpdateRecurrent(
      q_t->data.f, k_t->data.f, v_t->data.f, beta_t->data.f, g_t->data.f,
      rec_state->data.f, core_out->data.f, new_rec->data.f, B, H, N, D_k, D_v);
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
