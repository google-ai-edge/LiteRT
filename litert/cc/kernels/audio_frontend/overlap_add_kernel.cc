// Copyright 2025 Google LLC.
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

#include "litert/cc/kernels/audio_frontend/overlap_add_kernel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/c/c_api_types.h"
#include "tflite/schema/schema_generated.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/src/overlap_add.h"

namespace litert {
namespace audio_frontend {

namespace {
TfLiteType ConvertElementTypeToTfLiteType(ElementType element_type) {
  switch (element_type) {
    case ElementType::Float32:
      return kTfLiteFloat32;
    case ElementType::Int16:
      return kTfLiteInt16;
    default:
      return kTfLiteNoType;
  }
}

size_t GetElementSize(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteInt16:
      return sizeof(int16_t);
    default:
      return 0;
  }
}
}  // namespace

Expected<void> OverlapAddKernel::Init(const void* init_data,
                                      size_t init_data_size) {
  if (init_data == nullptr || init_data_size == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OverlapAddKernel: init_data is null or empty.");
  }

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(init_data);
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(buffer_t, init_data_size).AsMap();
  frame_step_ = m["frame_step"].AsInt32();
  auto tensor_type = static_cast<tflite::TensorType>(m["T"].AsInt32());

  switch (tensor_type) {
    case tflite::TensorType_FLOAT32:
      fft_type_ = kTfLiteFloat32;
      break;
    case tflite::TensorType_INT16:
      fft_type_ = kTfLiteInt16;
      break;
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "OverlapAddKernel: Unsupported tensor type.");
  }

  return {};
}

Expected<void> OverlapAddKernel::Destroy() {
  if (state_buffers_ != nullptr) {
    for (int i = 0; i < outer_dims_; ++i) {
      if (state_buffers_[i] != nullptr) {
        delete[] state_buffers_[i];
      }
    }
    delete[] state_buffers_;
    state_buffers_ = nullptr;
  }
  return {};
}

Expected<void> OverlapAddKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if (input_layouts.size() != 1 || output_layouts.size() != 1) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OverlapAddKernel: Expected 1 input and 1 output.");
  }

  const Layout& input_layout = input_layouts[0];
  if (input_layout.Dimensions().size() < 2) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "OverlapAddKernel: Input tensor must have at least 2 dimensions.");
  }

  frame_size_ = input_layout.Dimensions().back();
  n_frames_ = input_layout.Dimensions()[input_layout.Dimensions().size() - 2];
  LITERT_ASSIGN_OR_RETURN(size_t input_num_elements,
                          input_layout.NumElements());
  outer_dims_ = input_num_elements / (frame_size_ * n_frames_);

  // The output layout has one less dimension than the input.
  std::vector<int> output_dims;
  for (size_t i = 0; i < input_layout.Dimensions().size() - 1; ++i) {
    output_dims.push_back(input_layout.Dimensions()[i]);
  }
  output_dims.back() = frame_step_ * n_frames_;
  output_layouts[0] =
      Layout(Dimensions(output_dims.begin(), output_dims.end()));

  // Allocate state buffers here because outer_dims_ is now known.
  if (state_buffers_ != nullptr) {
    // Already allocated, destroy and reallocate if dimensions changed.
    LITERT_RETURN_IF_ERROR(Destroy());
  }

  if (outer_dims_ > 0) {
    state_buffers_ = new uint8_t*[outer_dims_];
    if (state_buffers_ == nullptr) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OverlapAddKernel: Failed to allocate state_buffers_.");
    }
    for (int i = 0; i < outer_dims_; ++i) {
      size_t buffer_size = frame_size_ * GetElementSize(fft_type_);
      state_buffers_[i] = new uint8_t[buffer_size];
      if (state_buffers_[i] == nullptr) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "OverlapAddKernel: Failed to allocate state buffer.");
      }
      memset(state_buffers_[i], 0, buffer_size);
    }
  }

  return {};
}

Expected<void> OverlapAddKernel::Run(const std::vector<TensorBuffer>& inputs,
                                     std::vector<TensorBuffer>& outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OverlapAddKernel: Expected 1 input and 1 output.");
  }

  const TensorBuffer& input = inputs[0];
  TensorBuffer& output = outputs[0];

  LITERT_ASSIGN_OR_RETURN(auto input_tensor_type, input.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_tensor_type, output.TensorType());

  TfLiteType input_tflite_type =
      ConvertElementTypeToTfLiteType(input_tensor_type.ElementType());
  TfLiteType output_tflite_type =
      ConvertElementTypeToTfLiteType(output_tensor_type.ElementType());

  if (input_tflite_type != fft_type_ || output_tflite_type != fft_type_) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OverlapAddKernel: Tensor type mismatch.");
  }

  for (int i = 0; i < outer_dims_; ++i) {
    for (int frame = 0; frame < n_frames_; ++frame) {
      size_t input_offset = (i * n_frames_ + frame) * frame_size_;
      size_t output_offset = (i * n_frames_ + frame) * frame_step_;

      switch (fft_type_) {
        case kTfLiteFloat32: {
          LITERT_ASSIGN_OR_RETURN(auto input_lock,
                                  TensorBufferScopedLock::Create<const float>(
                                      input, TensorBuffer::LockMode::kRead));
          LITERT_ASSIGN_OR_RETURN(auto output_lock,
                                  TensorBufferScopedLock::Create<float>(
                                      output, TensorBuffer::LockMode::kWrite));
          float* buffer = reinterpret_cast<float*>(state_buffers_[i]);
          OverlapAdd(&input_lock.second[input_offset], buffer, frame_size_,
                     &output_lock.second[output_offset], frame_step_);
          break;
        }
        case kTfLiteInt16: {
          LITERT_ASSIGN_OR_RETURN(auto input_lock,
                                  TensorBufferScopedLock::Create<const int16_t>(
                                      input, TensorBuffer::LockMode::kRead));
          LITERT_ASSIGN_OR_RETURN(auto output_lock,
                                  TensorBufferScopedLock::Create<int16_t>(
                                      output, TensorBuffer::LockMode::kWrite));
          int16_t* buffer = reinterpret_cast<int16_t*>(state_buffers_[i]);
          OverlapAdd(&input_lock.second[input_offset], buffer, frame_size_,
                     &output_lock.second[output_offset], frame_step_);
          break;
        }
        default:
          return Unexpected(kLiteRtStatusErrorInvalidArgument,
                            "OverlapAddKernel: Unsupported fft_type_.");
      }
    }
  }

  return {};
}

}  // namespace audio_frontend
}  // namespace litert
