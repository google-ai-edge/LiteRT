/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <jni.h>

#include <algorithm>
#include <cstddef>

#include "tflite/core/c/common.h"
#include "tflite/kernels/kernel_util.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_InterpreterTest_getNativeHandleForDelegate(
    JNIEnv* env, jclass clazz) {
  // A simple op which outputs a tensor with values of 7.
  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare =
          [](TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input;
            TF_LITE_ENSURE_OK(context,
                              tflite::GetInputSafe(context, node, 0, &input));
            TfLiteTensor* output;
            TF_LITE_ENSURE_OK(context,
                              tflite::GetOutputSafe(context, node, 0, &output));
            TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
            output->type = kTfLiteFloat32;
            return context->ResizeTensor(context, output, output_dims);
          },
      .invoke =
          [](TfLiteContext* context, TfLiteNode* node) {
            TfLiteTensor* output;
            TF_LITE_ENSURE_OK(context,
                              tflite::GetOutputSafe(context, node, 0, &output));
            std::fill(output->data.f,
                      output->data.f + tflite::NumElements(output), 7.0f);
            return kTfLiteOk;
          },
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "",
      .version = 1,
  };
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .Prepare = [](TfLiteContext* context,
                    TfLiteDelegate* delegate) -> TfLiteStatus {
        TfLiteIntArray* execution_plan;
        TF_LITE_ENSURE_STATUS(
            context->GetExecutionPlan(context, &execution_plan));
        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, registration, execution_plan, delegate);
        // Now bind delegate buffer handles for all tensors.
        for (size_t i = 0; i < context->tensors_size; ++i) {
          context->tensors[i].delegate = delegate;
          context->tensors[i].buffer_handle = static_cast<int>(i);
        }
        return kTfLiteOk;
      },
      .CopyFromBufferHandle = nullptr,
      .CopyToBufferHandle = nullptr,
      .FreeBufferHandle = nullptr,
      .flags = kTfLiteDelegateFlagsAllowDynamicTensors,
  };
  return reinterpret_cast<jlong>(&delegate);
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_InterpreterTest_getNativeHandleForInvalidDelegate(
    JNIEnv* env, jclass clazz) {
  // A simple delegate that fails during preparation.
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .Prepare = [](TfLiteContext* context, TfLiteDelegate* delegate)
          -> TfLiteStatus { return kTfLiteError; },
      .CopyFromBufferHandle = nullptr,
      .CopyToBufferHandle = nullptr,
      .FreeBufferHandle = nullptr,
      .flags = kTfLiteDelegateFlagsNone,
  };
  return reinterpret_cast<jlong>(&delegate);
}

namespace {

struct HeapDelegateState {
  TfLiteDelegate delegate;
  int prepared_kernels = 0;
  int active_kernels = 0;
  int prepared_buffers = 0;
  int active_buffers = 0;
};

}  // namespace

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapperTest_createHeapDelegate(
    JNIEnv* env, jclass clazz) {
  static TfLiteRegistration registration = {
      .init = [](TfLiteContext* context, const char* buffer,
                 size_t length) -> void* {
        const TfLiteDelegateParams* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* state = static_cast<HeapDelegateState*>(params->delegate->data_);
        ++state->prepared_kernels;
        ++state->active_kernels;
        return state;
      },
      .free =
          [](TfLiteContext* context, void* buffer) {
            auto* state = static_cast<HeapDelegateState*>(buffer);
            --state->active_kernels;
          },
      .prepare =
          [](TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input;
            TF_LITE_ENSURE_OK(context,
                              tflite::GetInputSafe(context, node, 0, &input));
            TfLiteTensor* output;
            TF_LITE_ENSURE_OK(context,
                              tflite::GetOutputSafe(context, node, 0, &output));
            TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
            output->type = kTfLiteFloat32;
            return context->ResizeTensor(context, output, output_dims);
          },
      .invoke =
          [](TfLiteContext* context, TfLiteNode* node) {
            TfLiteTensor* output;
            TF_LITE_ENSURE_OK(context,
                              tflite::GetOutputSafe(context, node, 0, &output));
            std::fill(output->data.f,
                      output->data.f + tflite::NumElements(output), 7.0f);
            return kTfLiteOk;
          },
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "",
      .version = 1,
  };

  auto* state = new HeapDelegateState();
  state->delegate = TfLiteDelegateCreate();
  state->delegate.data_ = state;
  state->delegate.Prepare = [](TfLiteContext* context,
                               TfLiteDelegate* delegate) -> TfLiteStatus {
    auto* state = static_cast<HeapDelegateState*>(delegate->data_);
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
    TF_LITE_ENSURE_STATUS(context->ReplaceNodeSubsetsWithDelegateKernels(
        context, registration, execution_plan, delegate));
    for (size_t i = 0; i < context->tensors_size; ++i) {
      context->tensors[i].delegate = delegate;
      context->tensors[i].buffer_handle = static_cast<int>(i);
      ++state->prepared_buffers;
      ++state->active_buffers;
    }
    return kTfLiteOk;
  };
  state->delegate.FreeBufferHandle = [](TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle* handle) {
    auto* state = static_cast<HeapDelegateState*>(delegate->data_);
    --state->active_buffers;
    *handle = kTfLiteNullBufferHandle;
  };
  state->delegate.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  return reinterpret_cast<jlong>(&state->delegate);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapperTest_deleteHeapDelegate(
    JNIEnv* env, jclass clazz, jlong delegate_handle) {
  auto* delegate = reinterpret_cast<TfLiteDelegate*>(delegate_handle);
  auto* state = static_cast<HeapDelegateState*>(delegate->data_);
  bool all_released_before_delete =
      (state->prepared_kernels > 0 && state->prepared_buffers > 0 &&
       state->active_kernels == 0 && state->active_buffers == 0);
  delete state;
  return all_released_before_delete ? JNI_TRUE : JNI_FALSE;
}

}  // extern "C"
