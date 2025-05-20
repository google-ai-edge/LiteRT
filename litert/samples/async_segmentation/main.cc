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
#include <vector>

#include <GLES2/gl2.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/samples/async_segmentation/image_processor.h"
#include "litert/samples/async_segmentation/image_utils.h"
#include "litert/samples/async_segmentation/segmentation_model.h"

namespace {
bool Initialize(ImageProcessor& processor, SegmentationModel& segmenter,
                SegmentationModel::AcceleratorType accelerator_choice) {
  if (!processor.InitializeGL(
          "shaders/passthrough_shader.vert", "shaders/mask_blend_compute.glsl",
          "shaders/resize_compute.glsl", "shaders/preprocess_compute.glsl",
          "shaders/deinterleave_masks.glsl")) {
    std::cerr << "Failed to initialize ImageProcessor." << std::endl;
    return false;
  }
  // We currently only support NPU model for Qualcomm devices (SM8750)
  std::string model_path =
      accelerator_choice == SegmentationModel::AcceleratorType::NPU
          ? "./models/selfie_multiclass_256x256_SM8750.tflite"
          : "./models/selfie_multiclass_256x256.tflite";
  std::string npu_library_path =
      accelerator_choice == SegmentationModel::AcceleratorType::NPU
          ? "/data/local/tmp/async_segmentation_android/npu/"
          : "";
  if (!segmenter.InitializeModel(model_path, npu_library_path)) {
    std::cerr << "Failed to initialize SegmentationModel." << std::endl;
    processor.ShutdownGL();  // ImageProcessor destructor will handle this
    return false;
  }
  return true;
}

bool LoadImage(const std::string& path, ImageProcessor& processor, int& width,
               int& height, int& channels_in_file, GLuint& texture_id,
               int desired_channels = 4) {
  auto img_data_cpu = ImageUtils::LoadImage(path, width, height,
                                            channels_in_file, desired_channels);
  std::cout << "Loaded image 1: " << path << " (" << width << "x" << height
            << " ChannelsInFile: " << channels_in_file
            << " LoadedAsChannels: " << desired_channels << ")" << std::endl;
  texture_id = processor.CreateOpenGLTexture(img_data_cpu, width, height,
                                             desired_channels);

  if (!texture_id) {
    std::cerr << "Failed to create OpenGL texture for image 1" << std::endl;
    ImageUtils::FreeImageData(img_data_cpu);
    return false;
  }
  ImageUtils::FreeImageData(img_data_cpu);
  img_data_cpu = nullptr;
  return true;
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " <input_image_path> <output_image_path> [cpu|gpu|npu (optional)]"
        << std::endl;
    return 1;
  }

  const std::string input_file(argv[1]);
  const std::string output_file(argv[2]);

  SegmentationModel::AcceleratorType accelerator_choice =
      SegmentationModel::AcceleratorType::CPU;

  if (argc == 4) {
    std::string accelerator_arg = argv[3];
    std::transform(accelerator_arg.begin(), accelerator_arg.end(),
                   accelerator_arg.begin(), ::tolower);
    if (accelerator_arg == "gpu") {
      accelerator_choice = SegmentationModel::AcceleratorType::GPU;
    } else if (accelerator_arg == "npu") {
      accelerator_choice = SegmentationModel::AcceleratorType::NPU;
    } else if (accelerator_arg != "cpu") {
      std::cerr << "Warning: Unknown accelerator '" << argv[3]
                << "'. Defaulting to CPU." << std::endl;
    }
  }
  // Define 6 distinct colors for the masks (RGBA)
  std::vector<RGBAColor> mask_colors = {
      {1.0f, 0.0f, 0.0f, 0.1f},  // Red (semi-transparent)
      {0.0f, 1.0f, 0.0f, 0.1f},  // Green
      {0.0f, 0.0f, 1.0f, 0.1f},  // Blue
      {1.0f, 1.0f, 0.0f, 0.1f},  // Yellow
      {1.0f, 0.0f, 1.0f, 0.1f},  // Magenta
      {0.0f, 1.0f, 1.0f, 0.1f}   // Cyan
  };
  ImageProcessor processor;
  // Whether to use GL buffers for input/output. Currently this is only used
  // for the GPU accelerator.
  bool use_gl_buffers =
      accelerator_choice == SegmentationModel::AcceleratorType::GPU;
  SegmentationModel segmenter(
      use_gl_buffers,
      accelerator_choice);  // Create segmentation model instance
  if (!Initialize(processor, segmenter, accelerator_choice)) {
    return 1;
  }

  int preprocessed_width = 256, preprocessed_height = 256;
  int width1_orig = 0, height1_orig = 0, channels1_file = 0,
      loaded_channels1 = 3;
  GLuint tex_id_orig = 0;

  auto start_time = absl::Now();
  // --- Load Image for Segmentation ---
  if (!LoadImage(input_file, processor, width1_orig, height1_orig,
                 channels1_file, tex_id_orig, loaded_channels1)) {
    std::cerr << "Failed to load image." << std::endl;
    return 1;
  }
  // Record end time
  auto end_time = absl::Now();
  auto duration = end_time - start_time;
  std::cout << "LoadImage took " << duration << std::endl;

  // --- Preprocess Image for Segmentation ---
  // When using GL buffers we need to set the number of channels to 4
  // (RGBA with alpha channel set as 0) so that memory layout is compatible with
  // GPU accelerator.
  start_time = absl::Now();
  int num_channels = use_gl_buffers ? 4 : 3;
  // When using GL buffers we can directly use the input buffer of the
  // segmentation model for preprocessing.
  GLuint preprocessed_buffer_id =
      use_gl_buffers ? segmenter.GetInputGlBufferId(0)
                     : processor.CreateOpenGLBuffer(
                           nullptr, preprocessed_width * preprocessed_height *
                                        num_channels * sizeof(float));

  if (!processor.PreprocessInputForSegmentation(
          tex_id_orig, width1_orig, height1_orig, preprocessed_width,
          preprocessed_height, preprocessed_buffer_id, num_channels)) {
    std::cerr << "Failed to preprocess input image for segmentation."
              << std::endl;
    return 1;
  }
  std::cout << "Preprocessed image to " << preprocessed_width << "x"
            << preprocessed_height << std::endl;

  // If not using GL buffers, we need to read the preprocessed GL buffer to
  // the input buffer of the segmentation model on CPU.
  if (!use_gl_buffers) {
    std::vector<float> preprocessed_buffer_data(
        preprocessed_width * preprocessed_height * num_channels);
    if (!processor.ReadBufferData(preprocessed_buffer_id, 0,
                                  preprocessed_width * preprocessed_height *
                                      num_channels * sizeof(float),
                                  preprocessed_buffer_data.data())) {
      std::cerr << "SegmentationModel: Failed to read preprocessed input SSBO."
                << std::endl;
      return 1;
    }
    litert::TensorBuffer& input_buffer = segmenter.GetInputBuffer(0);
    if (!input_buffer
             .Write<float>(absl::MakeConstSpan(preprocessed_buffer_data))
             .HasValue()) {
      std::cerr << "SegmentationModel: Failed to write preprocessed input to "
                   "TensorBuffer on CPU."
                << std::endl;
      return 1;
    }
  }

  // --- Run Segmentation ---
  if (!segmenter.RunSegmentation()) {
    std::cerr << "Failed to run segmentation." << std::endl;
    return 1;
  }

  litert::TensorBuffer& output_buffer = segmenter.GetOutputBuffer(0);
  if (output_buffer.HasEvent()) {
    LITERT_ASSIGN_OR_ABORT(auto event, output_buffer.GetEvent());
    event.Wait();
  }

  // --- Post-Processing: Deinterleave masks from output buffer ---
  // Generate 6 mask buffers for the 6 mask colors.
  std::vector<GLuint> mask_buffer_ids;
  mask_buffer_ids.reserve(6);
  for (int i = 0; i < 6; ++i) {
    mask_buffer_ids.push_back(processor.CreateOpenGLBuffer(
        nullptr, preprocessed_width * preprocessed_height * sizeof(float)));
  }
  if (use_gl_buffers) {
    // Deinterleave masks from the output buffer of the segmentation model on
    // GPU.
    GLuint output_buffer_id = segmenter.GetOutputGlBufferId(0);
    if (!processor.DeinterleaveMasks(output_buffer_id, mask_buffer_ids)) {
      std::cerr << "Failed to deinterleave masks." << std::endl;
      return 1;
    }
  } else {
    // We need to deinterleave masks from the output buffer of the segmentation
    // model on CPU.
    std::vector<float> deinterleaved_masks_data(preprocessed_width *
                                                preprocessed_height * 6);
    litert::TensorBuffer& output_buffer = segmenter.GetOutputBuffer(0);
    if (!output_buffer.Read<float>(absl::MakeSpan(deinterleaved_masks_data))
             .HasValue()) {
      std::cerr << "Failed to read deinterleaved masks from output "
                   "TensorBuffer on CPU."
                << std::endl;
      return 1;
    }
    if (!processor.DeinterleaveMasksCpu(deinterleaved_masks_data.data(),
                                        preprocessed_width, preprocessed_height,
                                        mask_buffer_ids)) {
      std::cerr << "Failed to deinterleave masks." << std::endl;
      return 1;
    }
  }

  // --- Post-Processing: Apply Colored Masks and Blend ---
  int out_blend_width = 0, out_blend_height = 0;
  std::vector<unsigned char> final_blended_pixel_data;

  GLuint final_blended_ssbo_id = 0;
  final_blended_ssbo_id = processor.ApplyColoredMasks(
      tex_id_orig, width1_orig, height1_orig, mask_buffer_ids, mask_colors,
      out_blend_width, out_blend_height);
  if (!final_blended_ssbo_id) {
    std::cerr << "Failed to apply colored masks and blend." << std::endl;
    return 1;
  }
  std::cout << "Applied colored masks to SSBO ID: " << final_blended_ssbo_id
            << std::endl;

  // --- Read data from the SSBO and save to image file ---
  std::vector<float> final_blended_float_data(out_blend_width *
                                              out_blend_height * 4);
  if (!processor.ReadBufferData(final_blended_ssbo_id, 0,
                                final_blended_float_data.size() * sizeof(float),
                                final_blended_float_data.data())) {
    processor.DeleteOpenGLBuffer(final_blended_ssbo_id);
    std::cerr << "Failed to read blended image data from SSBO." << std::endl;
    return 1;
  }
  end_time = absl::Now();
  duration = end_time - start_time;
  std::cout << "Full segmentation pipeline took " << duration << std::endl;

  start_time = absl::Now();
  // Convert float [0,1] to uchar [0,255]
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
  end_time = absl::Now();
  duration = end_time - start_time;
  std::cout << "SaveImage took " << duration << std::endl;

  // ---- Clean Up ----
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
