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

#include "litert/samples/async_segmentation/image_utils.h"

#include <algorithm>  // For std::transform
#include <cctype>     // For std::tolower
#include <cstddef>
#include <iostream>   // For std::cerr, std::cout
#include <string>
#include <vector>     // For RGB conversion buffer in saveImage for JPG

// STB Image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // from @stblib
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"  // from @stblib
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // from @stblib

unsigned char* ImageUtils::LoadImage(const std::string& path, int& width,
                                     int& height, int& channels_in_file,
                                     int desired_channels) {
  unsigned char* data = stbi_load(path.c_str(), &width, &height,
                                  &channels_in_file, desired_channels);
  if (!data) {
    // It's good practice to throw or return a clear error indicator.
    // stbi_failure_reason() gives more info.
    std::cerr << "ImageUtils::loadImage: Failed to load image '" << path
              << "' - " << stbi_failure_reason() << std::endl;
    return nullptr;
  }
  std::cout << "ImageUtils::loadImage: '" << path
  << "' - " << width << "x" << height << " channels: " << channels_in_file
            << " desired: " << desired_channels << std::endl;
return data;
}

void ImageUtils::FreeImageData(unsigned char* data) {
  if (data) {
    stbi_image_free(data);
  }
}

bool ImageUtils::SaveImage(const std::string& path, int width, int height,
                           int channels_in_data, const void* data) {
  if (!data || width <= 0 || height <= 0) {
    std::cerr << "ImageUtils::saveImage: Invalid image data or dimensions."
              << std::endl;
    return false;
  }

  std::string ext;
  size_t dot_pos = path.rfind('.');
  if (dot_pos != std::string::npos && dot_pos < path.length() - 1) {
    ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }

  int success = 0;
  if (ext == ".png") {
    success = stbi_write_png(path.c_str(), width, height, channels_in_data,
                             data, width * channels_in_data);
  } else if (ext == ".jpg" || ext == ".jpeg") {
    if (channels_in_data == 4) {  // stbi_write_jpg needs 3 channels (RGB)
      std::vector<unsigned char> rgbData(width * height * 3);
      const unsigned char* rgba_data_ptr =
          static_cast<const unsigned char*>(data);
      for (int i = 0; i < width * height; ++i) {
        rgbData[i * 3 + 0] = rgba_data_ptr[i * 4 + 0];  // R
        rgbData[i * 3 + 1] = rgba_data_ptr[i * 4 + 1];  // G
        rgbData[i * 3 + 2] = rgba_data_ptr[i * 4 + 2];  // B
      }
      success = stbi_write_jpg(path.c_str(), width, height, 3, rgbData.data(),
                               90);  // 90 is quality
    } else if (channels_in_data == 3) {
      success = stbi_write_jpg(path.c_str(), width, height, 3, data, 90);
    } else {
      std::cerr
          << "ImageUtils::saveImage: Cannot save JPG for '" << path
          << "'. Unsupported channel count " << channels_in_data
          << " (must be 3 for JPG output, or 4 for RGBA input to be converted)."
          << std::endl;
    }
  } else if (ext == ".bmp") {
    success =
        stbi_write_bmp(path.c_str(), width, height, channels_in_data, data);
  } else if (ext == ".tga") {
    success =
        stbi_write_tga(path.c_str(), width, height, channels_in_data, data);
  } else {
    std::cerr << "ImageUtils::saveImage: Unsupported output file extension '"
              << ext << "' for path '" << path
              << "'. Attempting to save as PNG." << std::endl;
    std::string png_path = (dot_pos != std::string::npos)
                               ? path.substr(0, dot_pos) + ".png"
                               : path + ".png";
    success = stbi_write_png(png_path.c_str(), width, height, channels_in_data,
                             data, width * channels_in_data);
    if (success)
      std::cout << "ImageUtils::saveImage: Saved as PNG to '" << png_path << "'"
                << std::endl;
  }

  if (!success) {
    std::cerr << "ImageUtils::saveImage: Failed to write image to '" << path
              << "'" << std::endl;
  }
  return success != 0;
}

std::vector<float> ImageUtils::ResizeImageCpu(const unsigned char* input,
                                              int input_width, int input_height,
                                              int input_channels,
                                              int output_width,
                                              int output_height) {
  std::vector<unsigned char> resized_image(output_width * output_height *
                                           input_channels);
  stbir_resize_uint8(input, input_width, input_height, 0, resized_image.data(),
                     output_width, output_height, 0, input_channels);
  std::vector<float> resized_image_float(resized_image.size());
  for (size_t i = 0; i < resized_image.size(); ++i) {
    resized_image_float[i] = resized_image[i] / 255.0f;
  }
  return resized_image_float;
}

std::vector<float> ImageUtils::PreprocessInputForSegmentationCpu(
    const unsigned char* input, int input_width, int input_height,
    int input_channels, int output_width, int output_height) {
  auto resized_image =
      ResizeImageCpu(input, input_width, input_height, input_channels,
                     output_width, output_height);
  // The model expects the input to be normalized to [-1, 1]
  for (size_t i = 0; i < resized_image.size(); ++i) {
    resized_image[i] = resized_image[i] * 2.0f - 1.0f;
  }
  return resized_image;
}

std::vector<std::vector<float>> ImageUtils::DeinterleaveMasksCpu(
    const float* data, int mask_width, int mask_height) {
  std::vector<std::vector<float>> out_masks_data;
  out_masks_data.assign(6, std::vector<float>(mask_width * mask_height, 0));

  for (int y = 0; y < mask_height; ++y) {
    for (int x = 0; x < mask_width; ++x) {
      for (int i = 0; i < 6; ++i) {
        out_masks_data[i][y * mask_width + x] =
            data[y * mask_width * 6 + x * 6 + i];
      }
    }
  }
  return out_masks_data;
}

std::vector<unsigned char> ImageUtils::ApplyColoredMasksCpu(
    const unsigned char* original_image, int original_width,
    int original_height, int original_channels,
    const std::vector<std::vector<float>>& masks,
    const std::vector<RGBAColor>& mask_colors) {
  std::vector<unsigned char> blended_image(original_width * original_height *
                                           original_channels);
  for (int y = 0; y < original_height; ++y) {
    for (int x = 0; x < original_width; ++x) {
      float blended_r =
          original_image[(y * original_width + x) * original_channels + 0] /
          255.0f;
      float blended_g =
          original_image[(y * original_width + x) * original_channels + 1] /
          255.0f;
      float blended_b =
          original_image[(y * original_width + x) * original_channels + 2] /
          255.0f;

      for (size_t i = 0; i < masks.size(); ++i) {
        float mask_value = masks[i][y * original_width + x];
        blended_r =
            blended_r * (1 - mask_value) + mask_colors[i].r * mask_value;
        blended_g =
            blended_g * (1 - mask_value) + mask_colors[i].g * mask_value;
        blended_b =
            blended_b * (1 - mask_value) + mask_colors[i].b * mask_value;
      }

      blended_image[(y * original_width + x) * original_channels + 0] =
          static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, blended_r)) *
                                     255.0f);
      blended_image[(y * original_width + x) * original_channels + 1] =
          static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, blended_g)) *
                                     255.0f);
      blended_image[(y * original_width + x) * original_channels + 2] =
          static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, blended_b)) *
                                     255.0f);
      if (original_channels == 4) {
        blended_image[(y * original_width + x) * original_channels + 3] =
            original_image[(y * original_width + x) * original_channels + 3];
      }
    }
  }
  return blended_image;
}
