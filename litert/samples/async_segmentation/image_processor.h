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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_IMAGE_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_IMAGE_PROCESSOR_H_

#include <stdbool.h>
#include <cstddef>
#include <string>
#include <vector>

// EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include "litert/samples/async_segmentation/image_utils.h"

class ImageProcessor {
 public:
  ImageProcessor();
  ~ImageProcessor();

  bool InitializeGL(const std::string& passthrough_vert_shader_path,
                    const std::string& mask_blend_compute_shader_path,
                    const std::string& resize_compute_shader_path,
                    const std::string& preprocess_compute_shader_path,
                    const std::string& deinterleave_masks_shader_path);
  void ShutdownGL();

  GLuint CreateOpenGLTexture(const unsigned char* image_data, int width,
                             int height, int channels);
  GLuint CreateOpenGLTexture(const float* image_data, int width, int height,
                             int channels);
  void DeleteOpenGLTexture(GLuint texture_id);
  // Creates an SSBO and uploads data.
  GLuint CreateOpenGLBuffer(const void* data, size_t data_size,
                            GLenum usage = GL_STATIC_DRAW);
  void DeleteOpenGLBuffer(GLuint buffer_id);

  // Returns an SSBO ID containing the blended image data (RGBA float [0,1])
  GLuint ApplyColoredMasks(GLuint original_image_tex_id, int original_width,
                           int original_height,
                           const std::vector<GLuint>& mask_buffer_ids,
                           const std::vector<RGBAColor>& mask_colors,
                           int& out_width, int& out_height);

  bool ReadTexturePixels(GLuint texture_id, int width, int height,
                         void* pixel_data, GLenum format, GLenum type);

  bool PreprocessInputForSegmentation(GLuint input_tex_id, int input_width,
                                        int input_height, int output_width,
                                        int output_height,
                                        GLuint preprocessed_buffer_id,
                                        int num_channels);

  // Reads data from an SSBO into a pre-allocated CPU buffer.
  bool ReadBufferData(GLuint buffer_id, size_t offset, size_t data_size,
                      void* out_data);
  GLuint ResizeTextureOpenGL(GLuint src_tex_id, int src_width, int src_height,
                             int target_width, int target_height);

  bool DeinterleaveMasksCpu(float* data, int mask_width, int mask_height,
                            std::vector<GLuint>& output_buffer_ids);

  bool DeinterleaveMasks(GLuint input_buffer_id,
                         std::vector<GLuint>& output_buffer_ids);

 private:
  EGLDisplay egl_display_ = EGL_NO_DISPLAY;
  EGLSurface egl_surface_ = EGL_NO_SURFACE;
  EGLContext egl_context_ = EGL_NO_CONTEXT;

  GLuint mask_blend_compute_shader_program_ = 0;
  GLuint resize_compute_shader_program_ = 0;
  GLuint preprocess_compute_shader_program_ = 0;
  GLuint deinterleave_masks_shader_program_ = 0;
  GLuint fbo_ = 0;

  void CleanupEGLContext();
  void CleanupEGLSurface();
  void CleanupGLResources();

  bool SetupComputeShader(const std::string& compute_shader_path,
                          GLuint& program_id);
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_ASYNC_SEGMENTATION_IMAGE_PROCESSOR_H_
