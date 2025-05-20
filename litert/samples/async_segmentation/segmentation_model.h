/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_SEGMENTATION_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_SEGMENTATION_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

// EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

class SegmentationModel {
 public:
  // Enum to specify accelerator type
  enum class AcceleratorType { CPU, GPU, NPU };

  explicit SegmentationModel(
      bool use_gl_buffers,
      AcceleratorType accelerator_type = AcceleratorType::GPU)
      : use_gl_buffers_(use_gl_buffers),
        current_accelerator_(accelerator_type) {};
  ~SegmentationModel() = default;

  // Initializes the LiteRT model from a given path.
  bool InitializeModel(const std::string& model_path,
                       std::string npu_library_path = "");

  bool RunSegmentation(bool run_async = true);

  GLuint GetInputGlBufferId(int index) {
    LITERT_ASSIGN_OR_ABORT(auto buffer, input_buffers_[index].GetGlBuffer());
    return buffer.id;
  }
  GLuint GetOutputGlBufferId(int index) {
    LITERT_ASSIGN_OR_ABORT(auto buffer, output_buffers_[index].GetGlBuffer());
    return buffer.id;
  }

  litert::TensorBuffer& GetInputBuffer(int index) {
    return input_buffers_[index];
  }

  litert::TensorBuffer& GetOutputBuffer(int index) {
    return output_buffers_[index];
  }

  bool UseGlBuffers() { return use_gl_buffers_; }

 private:
  // Whether to use GL buffers for input/output. Currently this is only used
  // for the GPU accelerator.
  bool use_gl_buffers_;
  litert::Model model_;
  litert::CompiledModel compiled_model_;
  AcceleratorType current_accelerator_ = AcceleratorType::CPU;
  std::unique_ptr<litert::Environment> env_;

  std::vector<litert::TensorBuffer> input_buffers_;
  std::vector<litert::TensorBuffer> output_buffers_;
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_SEGMENTATION_MODEL_H_
