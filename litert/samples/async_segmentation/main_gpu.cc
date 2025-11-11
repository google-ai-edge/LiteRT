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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <GLES2/gl2.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/samples/async_segmentation/image_processor.h"
#include "litert/samples/async_segmentation/image_utils.h"
#include "litert/samples/async_segmentation/timing_utils.h"

namespace {

litert::Options CreateGpuOptions(bool use_gl_buffers) {
  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
  LITERT_ASSIGN_OR_ABORT(auto& gpu_options, options.GetGpuOptions());
  if (use_gl_buffers) {
    LITERT_ABORT_IF_ERROR(
        gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32));
    LITERT_ABORT_IF_ERROR(gpu_options.SetBufferStorageType(
        litert::GpuOptions::BufferStorageType::kBuffer));
    LITERT_ABORT_IF_ERROR(gpu_options.EnableExternalTensorsMode(true));
  } else {
    LITERT_ABORT_IF_ERROR(gpu_options.EnableExternalTensorsMode(false));
  }

  options.SetHardwareAccelerators(litert::HwAccelerators::kGpu |
                                  litert::HwAccelerators::kCpu);
  return options;
}

litert::Expected<std::vector<litert::TensorBuffer>> CreateGlInputBuffers(
    litert::Environment& env, litert::CompiledModel& compiled_model,
    litert::Model& model, int signature_index) {
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          model.GetSignatureInputNames(signature_index));
  std::vector<litert::TensorBuffer> input_buffers;
  for (int i = 0; i < input_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBufferRequirements input_buffer_requirements,
        compiled_model.GetInputBufferRequirements(signature_index, i));
    LITERT_ASSIGN_OR_RETURN(litert::RankedTensorType ranked_tensor_type,
                            model.GetInputTensorType(signature_index, i));
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                            input_buffer_requirements.BufferSize());

    LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                            litert::TensorBuffer::CreateManaged(
                                env, litert::TensorBufferType::kGlBuffer,
                                ranked_tensor_type, buffer_size));

    input_buffers.push_back(std::move(input_buffer));
  }
  return input_buffers;
}

litert::Expected<std::vector<litert::TensorBuffer>> CreateGlOutputBuffers(
    litert::Environment& env, litert::CompiledModel& compiled_model,
    litert::Model& model, int signature_index) {
  LITERT_ASSIGN_OR_RETURN(auto output_names,
                          model.GetSignatureOutputNames(signature_index));
  std::vector<litert::TensorBuffer> output_buffers;
  output_buffers.reserve(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBufferRequirements output_buffer_requirements,
        compiled_model.GetOutputBufferRequirements(signature_index, i));
    LITERT_ASSIGN_OR_RETURN(litert::RankedTensorType ranked_tensor_type,
                            model.GetOutputTensorType(signature_index, i));
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                            output_buffer_requirements.BufferSize());

    LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                            litert::TensorBuffer::CreateManaged(
                                env, litert::TensorBufferType::kGlBuffer,
                                ranked_tensor_type, buffer_size));

    output_buffers.push_back(std::move(output_buffer));
  }
  return output_buffers;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " <model_path> <input_image_path> <output_image_path> "
                 "[use_gl_buffers "
                 "(true|false)]"
              << std::endl;
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string input_file = argv[2];
  const std::string output_file = argv[3];
  bool use_gl_buffers = false;

  if (argc == 5) {
    std::string use_gl_buffers_arg = argv[4];
    absl::c_transform(use_gl_buffers_arg, use_gl_buffers_arg.begin(),
                      ::tolower);
    if (use_gl_buffers_arg == "true") {
      use_gl_buffers = true;
    } else if (use_gl_buffers_arg != "false") {
      std::cerr << "Warning: Unknown value for use_gl_buffers '"
                << use_gl_buffers_arg << "'. Defaulting to false." << std::endl;
    }
  }

  std::vector<RGBAColor> mask_colors = {
      {1.0f, 0.0f, 0.0f, 0.1f}, {0.0f, 1.0f, 0.0f, 0.1f},
      {0.0f, 0.0f, 1.0f, 0.1f}, {1.0f, 1.0f, 0.0f, 0.1f},
      {1.0f, 0.0f, 1.0f, 0.1f}, {0.0f, 1.0f, 1.0f, 0.1f}};
  ImageProcessor processor;
  if (!processor.InitializeGL(
          "shaders/passthrough_shader.vert", "shaders/mask_blend_compute.glsl",
          "shaders/resize_compute.glsl", "shaders/preprocess_compute.glsl",
          "shaders/deinterleave_masks.glsl")) {
    std::cerr << "Failed to initialize ImageProcessor." << std::endl;
    return 1;
  }

  // Initialize LiteRT environment
  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));

  // Initialize LiteRT model
  LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(model_path));

  // Compile the model for the GPU
  litert::Options options = CreateGpuOptions(use_gl_buffers);
  LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                         litert::CompiledModel::Create(env, model, options));

  // Create input and output buffers.
  // When using GL buffers, the input and output buffers are created as GL
  // buffers. This allows for zero-copy pre- and post-processing, as the data
  // can be manipulated on the GPU without being copied to the CPU.
  std::vector<litert::TensorBuffer> input_buffers;
  std::vector<litert::TensorBuffer> output_buffers;
  if (use_gl_buffers) {
    LITERT_ASSIGN_OR_ABORT(input_buffers,
                           CreateGlInputBuffers(env, compiled_model, model, 0));
    LITERT_ASSIGN_OR_ABORT(
        output_buffers, CreateGlOutputBuffers(env, compiled_model, model, 0));
  } else {
    LITERT_ASSIGN_OR_ABORT(input_buffers, compiled_model.CreateInputBuffers());
    LITERT_ASSIGN_OR_ABORT(output_buffers,
                           compiled_model.CreateOutputBuffers());
  }

  // ================= PRE-PROCESSING =================
  // Load and preprocess the image
  ProfilingTimestamps profiling_timestamps;
  profiling_timestamps.load_image_start_time = absl::Now();
  int width_orig = 0, height_orig = 0, channels_file = 0, loaded_channels = 3;
  GLuint tex_id_orig = 0;
  auto img_data_cpu = ImageUtils::LoadImage(input_file, width_orig, height_orig,
                                            channels_file, loaded_channels);
  if (!img_data_cpu) {
    std::cerr << "Failed to load image file: " << input_file << std::endl;
    return 1;
  }
  profiling_timestamps.load_image_end_time =
      profiling_timestamps.e2e_start_time =
          profiling_timestamps.pre_process_start_time = absl::Now();
  tex_id_orig = processor.CreateOpenGLTexture(img_data_cpu, width_orig,
                                              height_orig, loaded_channels);
  if (!tex_id_orig) {
    std::cerr << "Failed to create OpenGL texture for image" << std::endl;
    ImageUtils::FreeImageData(img_data_cpu);
    return 1;
  }
  ImageUtils::FreeImageData(img_data_cpu);

  int preprocessed_width = 256, preprocessed_height = 256;
  int num_channels = use_gl_buffers ? 4 : 3;
  GLuint preprocessed_buffer_id;
  if (use_gl_buffers) {
    LITERT_ASSIGN_OR_ABORT(auto gl_buffer, input_buffers[0].GetGlBuffer());
    preprocessed_buffer_id = gl_buffer.id;
  } else {
    preprocessed_buffer_id = processor.CreateOpenGLBuffer(
        nullptr, preprocessed_width * preprocessed_height * num_channels *
                     sizeof(float));
  }

  if (!processor.PreprocessInputForSegmentation(
          tex_id_orig, width_orig, height_orig, preprocessed_width,
          preprocessed_height, preprocessed_buffer_id, num_channels)) {
    std::cerr << "Failed to preprocess input image for segmentation."
              << std::endl;
    return 1;
  }

  if (!use_gl_buffers) {
    std::vector<float> preprocessed_buffer_data(
        preprocessed_width * preprocessed_height * num_channels);
    if (!processor.ReadBufferData(preprocessed_buffer_id, 0,
                                  preprocessed_width * preprocessed_height *
                                      num_channels * sizeof(float),
                                  preprocessed_buffer_data.data())) {
      std::cerr << "Failed to read preprocessed input SSBO." << std::endl;
      return 1;
    }
    LITERT_ABORT_IF_ERROR(
        input_buffers[0].Write(absl::MakeConstSpan(preprocessed_buffer_data)));
  }
  profiling_timestamps.pre_process_end_time =
      profiling_timestamps.inference_start_time = absl::Now();
  // ================= INFERENCE =================
  // Run inference

  bool async = false;
  LITERT_ABORT_IF_ERROR(
      compiled_model.RunAsync(0, input_buffers, output_buffers, async));
  profiling_timestamps.inference_end_time =
      profiling_timestamps.post_process_start_time = absl::Now();

  // ================= POST-PROCESSING =================
  // Post-process the results
  if (output_buffers[0].HasEvent()) {
    LITERT_ASSIGN_OR_ABORT(auto event, output_buffers[0].GetEvent());
    event.Wait();
  }

  std::vector<GLuint> mask_buffer_ids;
  mask_buffer_ids.reserve(6);
  for (int i = 0; i < 6; ++i) {
    mask_buffer_ids.push_back(processor.CreateOpenGLBuffer(
        nullptr, preprocessed_width * preprocessed_height * sizeof(float)));
  }

  if (use_gl_buffers) {
    LITERT_ASSIGN_OR_ABORT(auto gl_buffer, output_buffers[0].GetGlBuffer());
    if (!processor.DeinterleaveMasks(gl_buffer.id, mask_buffer_ids)) {
      std::cerr << "Failed to deinterleave masks." << std::endl;
      return 1;
    }
  } else {
    std::vector<float> deinterleaved_masks_data(preprocessed_width *
                                                preprocessed_height * 6);
    LITERT_ABORT_IF_ERROR(
        output_buffers[0].Read(absl::MakeSpan(deinterleaved_masks_data)));
    if (!processor.DeinterleaveMasksCpu(deinterleaved_masks_data.data(),
                                        preprocessed_width, preprocessed_height,
                                        mask_buffer_ids)) {
      std::cerr << "Failed to deinterleave masks on CPU." << std::endl;
      return 1;
    }
  }

  int out_blend_width = 0, out_blend_height = 0;
  GLuint final_blended_ssbo_id = processor.ApplyColoredMasks(
      tex_id_orig, width_orig, height_orig, mask_buffer_ids, mask_colors,
      out_blend_width, out_blend_height);
  if (!final_blended_ssbo_id) {
    std::cerr << "Failed to apply colored masks and blend." << std::endl;
    return 1;
  }

  std::vector<float> final_blended_float_data(out_blend_width *
                                              out_blend_height * 4);
  if (!processor.ReadBufferData(final_blended_ssbo_id, 0,
                                final_blended_float_data.size() * sizeof(float),
                                final_blended_float_data.data())) {
    processor.DeleteOpenGLBuffer(final_blended_ssbo_id);
    std::cerr << "Failed to read blended image data from SSBO." << std::endl;
    return 1;
  }

  profiling_timestamps.post_process_end_time =
      profiling_timestamps.e2e_end_time =
          profiling_timestamps.save_image_start_time = absl::Now();
  // Save the output image
  std::vector<unsigned char> final_blended_uchar_data(out_blend_width *
                                                      out_blend_height * 4);
  for (size_t i = 0; i < final_blended_float_data.size(); ++i) {
    final_blended_uchar_data[i] = static_cast<unsigned char>(
        std::max(0.0f, std::min(1.0f, final_blended_float_data[i])) * 255.0f);
  }
  if (!ImageUtils::SaveImage(output_file, out_blend_width, out_blend_height, 4,
                             final_blended_uchar_data.data())) {
    std::cerr << "Failed to save final blended image." << std::endl;
    return 1;
  }
  std::cout << "Successfully saved final blended image to " << output_file
            << std::endl;

  profiling_timestamps.save_image_end_time = absl::Now();
  PrintTiming(profiling_timestamps);

  processor.DeleteOpenGLTexture(tex_id_orig);
  if (!use_gl_buffers) {
    processor.DeleteOpenGLBuffer(preprocessed_buffer_id);
  }
  for (GLuint id : mask_buffer_ids) {
    processor.DeleteOpenGLBuffer(id);
  }
  processor.DeleteOpenGLBuffer(final_blended_ssbo_id);

  return 0;
}
