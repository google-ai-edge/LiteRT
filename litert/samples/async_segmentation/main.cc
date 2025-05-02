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
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "third_party/GL/gl/include/GLES2/gl2.h"
#include "litert/samples/async_segmentation/image_processor.h"
#include "litert/samples/async_segmentation/image_utils.h"
#include "litert/samples/async_segmentation/segmentation_model.h"

namespace {
bool Initialize(ImageProcessor& processor, SegmentationModel& segmenter) {
  if (!processor.InitializeGL(
          "shaders/passthrough_shader.vert", "shaders/mask_blend_compute.glsl",
          "shaders/resize_compute.glsl", "shaders/preprocess_compute.glsl")) {
    std::cerr << "Failed to initialize ImageProcessor." << std::endl;
    return false;
  }
  if (!segmenter.InitializeModel("./models/selfie_multiclass_256x256.tflite")) {
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
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_image1_path> <output_image_path>" << std::endl;
    return 1;
  }

  const std::string input_file(argv[1]);
  const std::string output_file(argv[2]);

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
  SegmentationModel segmenter =
      SegmentationModel(&processor);  // Create segmentation model instance
  if (!Initialize(processor, segmenter)) {
    return 1;
  }

  GLuint preprocessed_buffer_id = 0;  // For the output of preprocessing
  int preprocessed_width = 256, preprocessed_height = 256;
  int width1_orig = 0, height1_orig = 0, channels1_file = 0,
      loaded_channels1 = 3;
  GLuint tex_id_orig = 0;

  // --- Load Image for Segmentation ---
  if (!LoadImage(input_file, processor, width1_orig, height1_orig,
                 channels1_file, tex_id_orig, loaded_channels1)) {
    std::cerr << "Failed to load image." << std::endl;
    return 1;
  }

  // --- Preprocess Image for Segmentation ---
  preprocessed_buffer_id = processor.PreprocessInputForSegmentation(
      tex_id_orig, width1_orig, height1_orig, preprocessed_width,
      preprocessed_height);
  if (!preprocessed_buffer_id) {
    std::cerr << "Failed to preprocess input image for segmentation."
              << std::endl;
    return 1;
  }
  std::cout << "Preprocessed image to " << preprocessed_width << "x"
            << preprocessed_height << std::endl;

  // --- Run Segmentation ---
  std::vector<GLuint> mask_buffer_ids;  // Vector of 6 mask buffers
  if (!segmenter.RunSegmentation(preprocessed_buffer_id, preprocessed_width,
                                 preprocessed_height, mask_buffer_ids)) {
    std::cerr << "Failed to run segmentation." << std::endl;
    return 1;
  };

  // --- Apply Colored Masks and Blend ---
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

  // ---- Clean Up ----
  processor.DeleteOpenGLTexture(tex_id_orig);
  processor.DeleteOpenGLBuffer(preprocessed_buffer_id);
  for (GLuint id : mask_buffer_ids) {
    processor.DeleteOpenGLBuffer(id);
  }
  processor.DeleteOpenGLBuffer(final_blended_ssbo_id);

  return 0;
}
