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

#include "litert/samples/async_segmentation/segmentation_model.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "third_party/GL/gl/include/GLES2/gl2.h"
#include "third_party/GL/gl/include/GLES3/gl3.h"
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/samples/async_segmentation/image_processor.h"

bool SegmentationModel::InitializeModel(const std::string& model_path,
                                        AcceleratorType accelerator_type,
                                        std::string npu_library_path) {
  current_accelerator_ = accelerator_type;
  std::cout << "SegmentationModel: Initializing model ... Path: " << model_path
            << std::endl;
  LITERT_ASSIGN_OR_ABORT(model_, litert::Model::CreateFromFile(model_path));
  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));

  switch (current_accelerator_) {
    case AcceleratorType::CPU: {
      LITERT_ASSIGN_OR_ABORT(
          compiled_model_,
          litert::CompiledModel::Create(env, model_, kLiteRtHwAcceleratorCpu));
      break;
    }
    case AcceleratorType::GPU: {
      LITERT_ASSIGN_OR_ABORT(
          compiled_model_,
          litert::CompiledModel::Create(env, model_, kLiteRtHwAcceleratorGpu));
      break;
    }
    case AcceleratorType::NPU: {
      // Environment setup.
      const std::vector<litert::Environment::Option> environment_options = {
        litert::Environment::Option{
            litert::Environment::OptionTag::DispatchLibraryDir,
            absl::string_view(npu_library_path),
        },
      };
      LITERT_ASSIGN_OR_ABORT(env,
                            litert::Environment::Create(environment_options));
      LITERT_ASSIGN_OR_ABORT(
          compiled_model_,
          litert::CompiledModel::Create(env, model_, kLiteRtHwAcceleratorNpu));
      break;
    }
  }

  LITERT_ASSIGN_OR_ABORT(auto signatures, model_.GetSignatures());

  size_t signature_index = 0;

  LITERT_ASSIGN_OR_ABORT(input_buffers_,
                         compiled_model_.CreateInputBuffers(signature_index));

  LITERT_ASSIGN_OR_ABORT(output_buffers_,
      compiled_model_.CreateOutputBuffers(signature_index));

  std::cout << "SegmentationModel: Model initialized." << std::endl;
  return true;
}

bool SegmentationModel::CreateMaskBuffers(
    std::vector<float> data, int input_width, int input_height,
    std::vector<GLuint>& output_buffer_ids) {
  std::cout << "SegmentationModel: Creating mask buffers ..." << std::endl;
  int mask_width = input_width;
  int mask_height = input_height;
  std::vector<std::vector<float>> out_masks_data;
  out_masks_data.assign(6, std::vector<float>(mask_width * mask_height,
                                              0));  // 6 single-channel masks

  // Generate 6 distinct masks
  for (int y = 0; y < mask_height; ++y) {
    for (int x = 0; x < mask_width; ++x) {
      for (int i = 0; i < 6; ++i) {
        // Create different patterns for each mask
        out_masks_data[i][y * mask_width + x] =
            data[y * mask_width * 6 + x * 6 + i];
      }
    }
  }

  output_buffer_ids.reserve(6);
  for (int i = 0; i < 6; ++i) {
    GLuint mask_ssbo_id = image_processor_->CreateOpenGLBuffer(
        out_masks_data[i].data(), out_masks_data[i].size() * sizeof(float));
    if (mask_ssbo_id == 0) {
      std::cerr << "Failed to create OpenGL buffer for mask " << i << std::endl;
      return false;
    }
    output_buffer_ids.push_back(mask_ssbo_id);
  }
  return true;
}

bool SegmentationModel::RunSegmentation(
    GLuint preprocessed_input_buffer_id, int input_width, int input_height,
    std::vector<GLuint>& output_buffer_ids) {
  std::cout << "SegmentationModel: Running segmentation on preprocessed "
               "buffer on accelerator: ";
  switch (current_accelerator_) {
    case AcceleratorType::GPU:
      std::cout << "GPU";
      break;
    case AcceleratorType::NPU:
      std::cout << "NPU";
      break;
    default:
      std::cout << "CPU";
      break;
  }
  std::cout << std::endl;
  if (input_width != 256 || input_height != 256) {
    std::cerr << "SegmentationModel: Error - input to runSegmentation should "
                 "be preprocessed (256x256)."
              << std::endl;
    return false;
  }
  if (preprocessed_input_buffer_id == 0) {
    std::cerr << "SegmentationModel: Invalid preprocessed input buffer ID."
              << std::endl;
    return false;
  }

  // 1. Read preprocessed input GL buffer to CPU float buffer.
  size_t buffer_data_size = input_width * input_height * 3 * sizeof(float);
  std::vector<float> float_input_buffer(input_width * input_height * 3);
  if (!image_processor_->ReadBufferData(preprocessed_input_buffer_id, 0,
                                        buffer_data_size,
                                        float_input_buffer.data())) {
    std::cerr << "SegmentationModel: Failed to read preprocessed input SSBO to "
                 "CPU float buffer."
              << std::endl;
    return false;
  }

  // 2. Copy input cpu data into model input tensor
  input_buffers_[0].Write<float>(
      absl::MakeSpan(float_input_buffer.data(), float_input_buffer.size()));
  std::cout << "SegmentationModel: Preparing input tensor for "
               "LiteRT model."
            << std::endl;
  // 3. Run model
  compiled_model_.Run(0, input_buffers_, output_buffers_);
  std::cout << "SegmentationModel: Executing LiteRT model." << std::endl;
  // 4. Get output tensor and read to CPU buffer
  LITERT_ASSIGN_OR_ABORT(auto packed_size, output_buffers_[0].PackedSize());
  auto output_size = input_width * input_height * 6;
  std::cout << "SegmentationModel: Output size: " << output_size
            << " packed_size:" << " " << packed_size << std::endl;
  std::vector<float> output_cpu_buffer;
  output_cpu_buffer.resize(output_size);
  output_buffers_[0].Read<float>(
      absl::MakeSpan(output_cpu_buffer.data(), packed_size / sizeof(float)));
  // 5. Create mask buffers
  return CreateMaskBuffers(output_cpu_buffer, input_width, input_height,
                           output_buffer_ids);
}
