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

#include <string>
#include <vector>

// EGL
#include "third_party/GL/gl/include/EGL/egl.h"
#include "third_party/GL/gl/include/EGL/eglext.h"
#include "third_party/GL/gl/include/GLES3/gl3.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/samples/async_segmentation/image_processor.h"

class SegmentationModel {
 public:
  explicit SegmentationModel(ImageProcessor* image_processor)
      : image_processor_(image_processor) {};
  ~SegmentationModel() = default;

  // Initializes the LiteRT model from a given path.
  bool InitializeModel(const std::string& model_path);

  // Takes an SSBO ID as input
  bool RunSegmentation(GLuint preprocessed_input_buffer_id, int input_width,
                       int input_height,
                       std::vector<GLuint>& output_buffer_ids);

 private:
  bool CreateMaskBuffers(std::vector<float> data, int input_width,
                         int input_height,
                         std::vector<GLuint>& output_buffer_ids);
  litert::Model model_;
  litert::CompiledModel compiled_model_;
  ImageProcessor* image_processor_;

  std::vector<litert::TensorBuffer> input_buffers_;
  std::vector<litert::TensorBuffer> output_buffers_;
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_SEGMENTATION_MODEL_H_
