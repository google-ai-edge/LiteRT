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
#include "third_party/stblib/stb_image.h"
#include "third_party/stblib/stb_image_write.h"

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
