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

#include "litert/samples/async_segmentation/image_processor.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <GLES3/gl32.h>
#include "absl/log/absl_check.h"  // from @com_google_absl

namespace {

void GlCheckErrorDetail(const char* file, int line,
                        const char* operation = "") {
  GLenum error_code;
  while ((error_code = glGetError()) != GL_NO_ERROR) {
    std::string error_string;
    switch (error_code) {
      case GL_INVALID_ENUM:
        error_string = "INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        error_string = "INVALID_VALUE";
        break;
      case GL_INVALID_OPERATION:
        error_string = "INVALID_OPERATION";
        break;
      case GL_OUT_OF_MEMORY:
        error_string = "OUT_OF_MEMORY";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        error_string = "INVALID_FRAMEBUFFER_OPERATION";
        break;
      default:
        error_string = "UNKNOWN_ERROR_CODE_0x" + std::to_string(error_code);
        break;
    }
    std::cerr << "GL_ERROR (" << operation << "): " << error_string << " - "
              << file << ":" << line << std::endl;
  }
}
#define GlCheckErrorOp(op) GlCheckErrorDetail(__FILE__, __LINE__, op)
#define GlCheckError() GlCheckErrorDetail(__FILE__, __LINE__)

void EglCheckErrorDetail(const char* file, int line,
                         const char* operation = "") {
  EGLint egl_error_code = eglGetError();
  if (egl_error_code != EGL_SUCCESS) {
    std::cerr << "EGL_ERROR (" << operation << "): code 0x" << std::hex
              << egl_error_code << " at " << file << ":" << line << std::endl;
  }
}
#define EglCheckErrorOp(op) EglCheckErrorDetail(__FILE__, __LINE__, op)
#define EglCheckError() EglCheckErrorDetail(__FILE__, __LINE__)

std::string LoadShaderSourceFromFile(const std::string& filepath) {
  std::ifstream file_stream(filepath);
  if (!file_stream.is_open()) {
    ABSL_CHECK(false)
        << "LoadShaderSourceFromFile: Failed to open shader file: " + filepath;
  }
  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  return buffer.str();
}

GLuint CompileShaderInternal(GLenum type, const std::string& source_code) {
  GLuint shader_id = glCreateShader(type);
  const char* source_code_ptr = source_code.c_str();
  glShaderSource(shader_id, 1, &source_code_ptr, nullptr);
  glCompileShader(shader_id);
  GLint success_status;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success_status);
  if (!success_status) {
    GLchar info_log[1024];
    glGetShaderInfoLog(shader_id, sizeof(info_log), nullptr, info_log);
    std::string shader_type_string = (type == GL_VERTEX_SHADER)     ? "VERTEX"
                                     : (type == GL_FRAGMENT_SHADER) ? "FRAGMENT"
                                                                    : "COMPUTE";
    glDeleteShader(shader_id);
    ABSL_CHECK(false) << "CompileShaderInternal: SHADER " + shader_type_string +
                             " compilation failed:\n" + info_log;
  }
  return shader_id;
}

}  // anonymous namespace

bool ImageProcessor::SetupComputeShader(const std::string& compute_shader_path,
                                        GLuint& program_id) {
  std::cout << "ImageProcessor::SetupComputeShader: " << compute_shader_path
            << std::endl;
  std::string compute_source = LoadShaderSourceFromFile(compute_shader_path);
  GLuint compute_shader_id = 0;
  program_id = 0;

  compute_shader_id = CompileShaderInternal(GL_COMPUTE_SHADER, compute_source);
  program_id = glCreateProgram();
  glAttachShader(program_id, compute_shader_id);
  glLinkProgram(program_id);

  GLint link_success_status;
  glGetProgramiv(program_id, GL_LINK_STATUS, &link_success_status);
  if (!link_success_status) {
    GLchar info_log[1024];
    glGetProgramInfoLog(program_id, sizeof(info_log), nullptr, info_log);
    ABSL_CHECK(false) << "ImageProcessor::SetupComputeShader: COMPUTE SHADER " +
                             compute_shader_path + "PROGRAM linking failed:\n" +
                             info_log;
  }
  if (glGetError() != GL_NO_ERROR) {
    std::cerr << "Unknown error during compute shader setup." << std::endl;
    if (compute_shader_id) glDeleteShader(compute_shader_id);
    if (program_id) {
      glDeleteProgram(program_id);
      program_id = 0;
    }
    return false;
  }

  glDeleteShader(compute_shader_id);
  return program_id != 0;
}

ImageProcessor::ImageProcessor() = default;

ImageProcessor::~ImageProcessor() { ShutdownGL(); }

bool ImageProcessor::InitializeGL(
    const std::string& passthrough_vert_shader_path,
    const std::string& mask_blend_compute_shader_path,
    const std::string& resize_compute_shader_path,
    const std::string& preprocess_compute_shader_path,
    const std::string& deinterleave_masks_shader_path) {
  egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (egl_display_ == EGL_NO_DISPLAY) {
    std::cerr << "ImageProcessor: Failed to get EGL display." << std::endl;
    EglCheckErrorOp("eglGetDisplay");
    return false;
  }
  EGLint major_version, minor_version;
  if (!eglInitialize(egl_display_, &major_version, &minor_version)) {
    std::cerr << "ImageProcessor: Failed to initialize EGL." << std::endl;
    EglCheckErrorOp("eglInitialize");
    return false;
  }
  std::cout << "ImageProcessor: EGL Initialized. Version: " << major_version
            << "." << minor_version << std::endl;

  EGLint const attribute_list[] = {EGL_SURFACE_TYPE,
                                   EGL_PBUFFER_BIT,
                                   EGL_RENDERABLE_TYPE,
                                   EGL_OPENGL_ES3_BIT_KHR,
                                   EGL_RED_SIZE,
                                   8,
                                   EGL_GREEN_SIZE,
                                   8,
                                   EGL_BLUE_SIZE,
                                   8,
                                   EGL_ALPHA_SIZE,
                                   8,
                                   EGL_DEPTH_SIZE,
                                   0,
                                   EGL_STENCIL_SIZE,
                                   0,
                                   EGL_NONE};
  EGLConfig egl_config;
  EGLint num_configs;
  if (!eglChooseConfig(egl_display_, attribute_list, &egl_config, 1,
                       &num_configs) ||
      num_configs == 0) {
    std::cerr
        << "ImageProcessor: Failed to choose/find EGL config for GLES 3.x."
        << std::endl;
    EglCheckErrorOp("eglChooseConfig");
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  EGLint const pbuffer_attributes[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
  egl_surface_ =
      eglCreatePbufferSurface(egl_display_, egl_config, pbuffer_attributes);
  if (egl_surface_ == EGL_NO_SURFACE) {
    std::cerr << "ImageProcessor: Failed to create EGL Pbuffer surface."
              << std::endl;
    EglCheckErrorOp("eglCreatePbufferSurface");
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  EGLint const context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
                                       EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
  egl_context_ = eglCreateContext(egl_display_, egl_config, EGL_NO_CONTEXT,
                                  context_attributes);
  if (egl_context_ == EGL_NO_CONTEXT) {
    EGLint const context_attributes_30[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
                                            EGL_NONE};
    std::cerr
        << "ImageProcessor: Failed to create EGL 3.1 context. Trying 3.0..."
        << std::endl;
    EglCheckErrorOp("eglCreateContext 3.1");
    egl_context_ = eglCreateContext(egl_display_, egl_config, EGL_NO_CONTEXT,
                                    context_attributes_30);
    if (egl_context_ == EGL_NO_CONTEXT) {
      std::cerr << "ImageProcessor: Failed to create EGL 3.0 context."
                << std::endl;
      EglCheckErrorOp("eglCreateContext 3.0");
      CleanupEGLSurface();
      eglTerminate(egl_display_);
      egl_display_ = EGL_NO_DISPLAY;
      return false;
    }
  }

  if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
    std::cerr << "ImageProcessor: Failed to make EGL context current."
              << std::endl;
    EglCheckErrorOp("eglMakeCurrent");
    CleanupEGLContext();
    CleanupEGLSurface();
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  std::cout << "ImageProcessor: OpenGL ES Version: " << glGetString(GL_VERSION)
            << std::endl;
  const char* version_str =
      reinterpret_cast<const char*>(glGetString(GL_VERSION));
  if (version_str == nullptr ||
      (strstr(version_str, "OpenGL ES 3.1") == nullptr &&
       strstr(version_str, "OpenGL ES 3.2") == nullptr)) {
    std::cerr << "ImageProcessor: WARNING - OpenGL ES 3.1+ not fully supported "
                 "by context. Compute shaders may not work."
              << std::endl;
  }

  if (!SetupComputeShader(mask_blend_compute_shader_path,
                          mask_blend_compute_shader_program_)) {
    std::cerr << "ImageProcessor: Failed to setup mask blend compute shader."
              << std::endl;
    ShutdownGL();
    return false;
  }
  if (!SetupComputeShader(resize_compute_shader_path,
                          resize_compute_shader_program_)) {
    std::cerr << "ImageProcessor: Failed to setup resize compute shader."
              << std::endl;
    ShutdownGL();
    return false;
  }
  if (!SetupComputeShader(preprocess_compute_shader_path,
                          preprocess_compute_shader_program_)) {
    std::cerr << "ImageProcessor: Failed to setup preprocess compute shader."
              << std::endl;
    ShutdownGL();
    return false;
  }

  if (!SetupComputeShader(deinterleave_masks_shader_path,
                          deinterleave_masks_shader_program_)) {
    std::cerr << "Image Processor: Failed to setup deinterleave masks shader."
              << std::endl;
    ShutdownGL();
    return false;
  }

  glGenFramebuffers(1, &fbo_);
  GlCheckErrorOp("glGenFramebuffers for reusable FBO");

  return true;
}

void ImageProcessor::ShutdownGL() {
  CleanupGLResources();
  if (egl_display_ != EGL_NO_DISPLAY) {
    eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                   EGL_NO_CONTEXT);
    CleanupEGLContext();
    CleanupEGLSurface();
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
  }
}

GLuint ImageProcessor::CreateOpenGLTexture(const unsigned char* image_data,
                                           int width, int height,
                                           int channels) {
  if (!image_data || width <= 0 || height <= 0) {
    std::cerr << "ImageProcessor::CreateOpenGLTexture (uchar): Invalid input "
                 "data or dimensions."
              << std::endl;
    return 0;
  }

  GLenum format;
  GLenum internal_format;

  if (channels == 1) {
    format = GL_RED;
    internal_format = GL_R8;
  } else if (channels == 3) {
    format = GL_RGB;
    internal_format = GL_RGB8;
  } else if (channels == 4) {
    format = GL_RGBA;
    internal_format = GL_RGBA8;
  } else {
    std::cerr << "ImageProcessor::CreateOpenGLTexture (uchar): Unsupported "
                 "channel count: "
              << channels << ". Only 1, 3, or 4 are supported." << std::endl;
    return 0;
  }

  GLuint texture_id = 0;
  glGenTextures(1, &texture_id);
  if (!texture_id) {
    std::cerr << "ImageProcessor::CreateOpenGLTexture (uchar): Failed to "
                 "generate texture ID."
              << std::endl;
    GlCheckErrorOp("glGenTextures");
    return 0;
  }
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format,
               GL_UNSIGNED_BYTE, image_data);
  GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    GlCheckErrorDetail(__FILE__, __LINE__, "glTexImage2D (uchar)");
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &texture_id);
    return 0;
  }

  glGenerateMipmap(GL_TEXTURE_2D);
  GlCheckErrorOp("glGenerateMipmap (uchar)");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return texture_id;
}

GLuint ImageProcessor::CreateOpenGLTexture(const float* image_data, int width,
                                           int height, int channels) {
  if (!image_data || width <= 0 || height <= 0) {
    std::cerr << "ImageProcessor::CreateOpenGLTexture (float): Invalid input "
                 "data or dimensions."
              << std::endl;
    return 0;
  }

  GLenum format;
  GLenum internal_format;

  if (channels == 1) {
    format = GL_RED;
    internal_format = GL_R16F;
  } else if (channels == 3) {
    format = GL_RGB;
    internal_format = GL_RGB16F;
  } else if (channels == 4) {
    format = GL_RGBA;
    internal_format = GL_RGBA16F;
  } else {
    std::cerr << "ImageProcessor::CreateOpenGLTexture (float): Unsupported "
                 "channel count: "
              << channels << ". Only 1, 3, or 4 are supported." << std::endl;
    return 0;
  }

  GLuint texture_id = 0;
  glGenTextures(1, &texture_id);
  if (!texture_id) {
    std::cerr << "ImageProcessor::CreateOpenGLTexture (float): Failed to "
                 "generate texture ID."
              << std::endl;
    GlCheckErrorOp("glGenTextures");
    return 0;
  }
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format,
               GL_FLOAT, image_data);
  GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    GlCheckErrorDetail(__FILE__, __LINE__, "glTexImage2D (float)");
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &texture_id);
    return 0;
  }

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return texture_id;
}

GLuint ImageProcessor::CreateOpenGLBuffer(const void* data, size_t data_size,
                                          GLenum usage) {
  if (data_size <= 0) {
    std::cerr << "ImageProcessor::CreateOpenGLBuffer: Invalid data size."
              << std::endl;
    return 0;
  }
  GLuint buffer_id = 0;
  glGenBuffers(1, &buffer_id);
  GlCheckErrorOp("CreateOpenGLBuffer - glGenBuffers");
  if (!buffer_id) {
    std::cerr
        << "ImageProcessor::CreateOpenGLBuffer: Failed to generate buffer."
        << std::endl;
    return 0;
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id);
  GlCheckErrorOp("CreateOpenGLBuffer - glBindBuffer");
  glBufferData(GL_SHADER_STORAGE_BUFFER, data_size, data, usage);
  GlCheckErrorOp("CreateOpenGLBuffer - glBufferData");
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return buffer_id;
}

void ImageProcessor::DeleteOpenGLTexture(GLuint texture_id) {
  if (texture_id != 0) {
    glDeleteTextures(1, &texture_id);
    GlCheckErrorOp("glDeleteTextures");
  }
}

void ImageProcessor::DeleteOpenGLBuffer(GLuint buffer_id) {
  if (buffer_id != 0) {
    glDeleteBuffers(1, &buffer_id);
    GlCheckErrorOp("glDeleteBuffers");
  }
}

GLuint ImageProcessor::ResizeTextureOpenGL(GLuint src_tex_id, int src_width,
                                           int src_height, int target_width,
                                           int target_height) {
  if (src_tex_id == 0 || src_width <= 0 || src_height <= 0 ||
      target_width <= 0 || target_height <= 0) {
    std::cerr << "ImageProcessor::ResizeTextureOpenGL: Invalid parameters."
              << std::endl;
    return 0;
  }
  if (resize_compute_shader_program_ == 0) {
    std::cerr << "ImageProcessor::ResizeTextureOpenGL: Resize compute shader "
                 "not initialized."
              << std::endl;
    return 0;
  }

  GLuint resized_texture_id = 0;
  glGenTextures(1, &resized_texture_id);
  glBindTexture(GL_TEXTURE_2D, resized_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target_width, target_height, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  GlCheckErrorOp("ResizeTextureOpenGL - create target texture");

  glUseProgram(resize_compute_shader_program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, src_tex_id);
  glUniform1i(
      glGetUniformLocation(resize_compute_shader_program_, "inputTexture"), 0);
  glBindImageTexture(0, resized_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                     GL_RGBA8);
  GlCheckErrorOp("ResizeTextureOpenGL - bind image textures and sampler");

  glDispatchCompute((target_width + 7) / 8, (target_height + 7) / 8, 1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                  GL_TEXTURE_FETCH_BARRIER_BIT);
  GlCheckErrorOp("ResizeTextureOpenGL - dispatch compute & barrier");

  return resized_texture_id;
}

bool ImageProcessor::PreprocessInputForSegmentation(
    GLuint input_tex_id, int input_width, int input_height, int output_width,
    int output_height, GLuint preprocessed_buffer_id, int num_channels) {
  if (input_tex_id == 0 || input_width <= 0 || input_height <= 0) {
    std::cerr << "ImageProcessor::PreprocessInputForSegmentation: Invalid "
                 "input parameters."
              << std::endl;
    return false;
  }
  if (preprocess_compute_shader_program_ == 0) {
    std::cerr << "ImageProcessor::PreprocessInputForSegmentation: Preprocess "
                 "compute shader not initialized."
              << std::endl;
    return false;
  }

  glUseProgram(preprocess_compute_shader_program_);
  GlCheckErrorOp("PreprocessInputForSegmentation - glUseProgram");

  glUniform1i(
      glGetUniformLocation(preprocess_compute_shader_program_, "num_channels"),
      num_channels);
  GlCheckErrorOp("PreprocessInputForSegmentation - set inputTexture uniform");

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, input_tex_id);
  GlCheckErrorOp("PreprocessInputForSegmentation - glBindTexture");
  glUniform1i(
      glGetUniformLocation(preprocess_compute_shader_program_, "inputTexture"),
      0);
  GlCheckErrorOp("PreprocessInputForSegmentation - set inputTexture uniform");

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, preprocessed_buffer_id);
  GlCheckErrorOp("PreprocessInputForSegmentation - glBindBufferBase for SSBO");

  glDispatchCompute((output_width + 7) / 8, (output_height + 7) / 8, 1);
  GlCheckErrorOp("PreprocessInputForSegmentation - glDispatchCompute");
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
  GlCheckErrorOp("PreprocessInputForSegmentation - glMemoryBarrier");

  glFinish();
  GlCheckErrorOp("PreprocessInputForSegmentation - glFinish");

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glUseProgram(0);

  return true;
}

GLuint ImageProcessor::ApplyColoredMasks(
    GLuint original_image_tex_id, int original_width, int original_height,
    const std::vector<GLuint>& mask_buffer_ids,
    const std::vector<RGBAColor>& mask_colors, int& out_width,
    int& out_height) {
  GlCheckErrorOp("ApplyColoredMasks - Entry");

  if (mask_blend_compute_shader_program_ == 0) {
    std::cerr << "ImageProcessor::ApplyColoredMasks: Blend compute shader not "
                 "initialized."
              << std::endl;
    return 0;
  }
  if (original_image_tex_id == 0 || original_width <= 0 ||
      original_height <= 0) {
    std::cerr << "ImageProcessor::ApplyColoredMasks: Invalid original image "
                 "texture or dimensions."
              << std::endl;
    return 0;
  }
  if (mask_buffer_ids.size() != 6 || mask_colors.size() != 6) {
    std::cerr << "ImageProcessor::ApplyColoredMasks: Requires exactly 6 mask "
                 "buffers and 6 colors."
              << std::endl;
    return 0;
  }

  out_width = original_width;
  out_height = original_height;

  GLuint output_ssbo_id = 0;
  glGenBuffers(1, &output_ssbo_id);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_ssbo_id);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               out_width * out_height * 4 * sizeof(float), nullptr,
               GL_STATIC_DRAW);  // RGBA float output
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  GlCheckErrorOp("ApplyColoredMasks - create output SSBO");

  glUseProgram(mask_blend_compute_shader_program_);
  GlCheckErrorOp("ApplyColoredMasks - glUseProgram for blend");

  GLint loc;

  // Bind output SSBO to SSBO binding point 0 (matches shader layout binding=0
  // for output_blend_buffer)
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, output_ssbo_id);
  GlCheckErrorOp(
      "ApplyColoredMasks - glBindBufferBase for output SSBO to unit 0");

  // Bind base image as sampler to TEXTURE UNIT 1 (sampler uniform "baseTexture"
  // set to 1)
  loc = glGetUniformLocation(mask_blend_compute_shader_program_, "baseTexture");
  if (loc == -1) {
    std::cerr << "Uniform baseTexture not found!" << std::endl;
    DeleteOpenGLBuffer(output_ssbo_id);
    return 0;
  }
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, original_image_tex_id);
  glUniform1i(loc, 1);
  GlCheckErrorOp("ApplyColoredMasks - baseTexture setup");

  // Bind mask SSBOs to SSBO binding points 2 through 7
  for (int i = 0; i < 6; ++i) {
    if (i < mask_buffer_ids.size()) {
      std::string buffer_uniform_name =
          "MaskBuffer" + std::to_string(i);  // This matches shader block name
      // We don't set uniforms for SSBOs like this, they are bound via
      // glBindBufferBase
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, mask_buffer_ids[i]);

      std::string color_uniform_name = "maskColor" + std::to_string(i);
      loc = glGetUniformLocation(mask_blend_compute_shader_program_,
                                 color_uniform_name.c_str());
      if (loc == -1) {
        std::cerr << "Uniform " << color_uniform_name << " not found!"
                  << std::endl;
        DeleteOpenGLBuffer(output_ssbo_id);
        return 0;
      }
      glUniform4f(loc, mask_colors[i].r, mask_colors[i].g, mask_colors[i].b,
                  mask_colors[i].a);
    }
  }
  GlCheckErrorOp(
      "ApplyColoredMasks - After setting color uniforms and binding mask "
      "SSBOs");

  glDispatchCompute((out_width + 7) / 8, (out_height + 7) / 8, 1);
  GlCheckErrorOp("ApplyColoredMasks - glDispatchCompute");
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
  GlCheckErrorOp("ApplyColoredMasks - glMemoryBarrier");

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);  // Unbind output SSBO
  for (int i = 0; i < 6; ++i) {                      // Unbind mask SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, 0);
  }
  glUseProgram(0);

  return output_ssbo_id;
}

bool ImageProcessor::ReadTexturePixels(GLuint texture_id, int width, int height,
                                       void* pixel_data, GLenum format,
                                       GLenum type) {
  if (texture_id == 0 || width <= 0 || height <= 0 || fbo_ == 0 ||
      pixel_data == nullptr) {
    std::cerr << "ImageProcessor::ReadTexturePixels: Invalid parameters or FBO "
                 "not initialized."
              << std::endl;
    return false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         texture_id, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr
        << "ImageProcessor::ReadTexturePixels: Framebuffer is not complete!"
        << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return false;
  }

  glReadBuffer(GL_COLOR_ATTACHMENT0);
  int pack_alignment = 4;
  if (format == GL_RED && type == GL_UNSIGNED_BYTE) {
    pack_alignment = 1;
  } else if (type == GL_HALF_FLOAT) {
    if (format == GL_RGB) {
      pack_alignment = 2;
    } else if (format == GL_RGBA) {
      pack_alignment = 4;
    }
  }

  glPixelStorei(GL_PACK_ALIGNMENT, pack_alignment);
  glReadPixels(0, 0, width, height, format, type, pixel_data);
  glPixelStorei(GL_PACK_ALIGNMENT, 4);
  GlCheckErrorOp("ReadTexturePixels - glReadPixels");

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  return true;
}

bool ImageProcessor::ReadBufferData(GLuint buffer_id, size_t offset,
                                    size_t data_size, void* out_data) {
  if (buffer_id == 0 || data_size == 0 || out_data == nullptr) {
    std::cerr << "ImageProcessor::ReadBufferData: Invalid parameters."
              << std::endl;
    return false;
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id);
  GlCheckErrorOp("ReadBufferData - glBindBuffer");

  GLint buffer_actual_size;
  glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE,
                         &buffer_actual_size);
  if (offset + data_size > static_cast<size_t>(buffer_actual_size)) {
    std::cerr << "ImageProcessor::ReadBufferData: Requested read range exceeds "
                 "buffer size. "
              << offset + data_size << " > " << buffer_actual_size << " bytes."
              << std::endl;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return false;
  }

  void* mapped_ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset,
                                      data_size, GL_MAP_READ_BIT);
  if (!mapped_ptr) {
    std::cerr << "ImageProcessor::ReadBufferData: Failed to map SSBO."
              << std::endl;
    GlCheckErrorOp("ReadBufferData - glMapBufferRange");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return false;
  }

  memcpy(out_data, mapped_ptr, data_size);

  if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)) {
    std::cerr << "ImageProcessor::ReadBufferData: Failed to unmap SSBO."
              << std::endl;
    GlCheckErrorOp("ReadBufferData - glUnmapBuffer");
    // Data might have been copied, but unmapping failed. Consider the operation
    // partially successful or failed.
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return true;
}

bool ImageProcessor::DeinterleaveMasksCpu(
    float* data, int mask_width, int mask_height,
    std::vector<GLuint>& output_buffer_ids) {
  std::cout << "DeinterleaveMasksCpu: Deinterleaving masks on CPU ..."
            << std::endl;
  std::vector<std::vector<float>> out_masks_data;
  out_masks_data.assign(6, std::vector<float>(mask_width * mask_height,
                                              0));  // 6 single-channel masks

  // Generate 6 distinct masks.
  for (int y = 0; y < mask_height; ++y) {
    for (int x = 0; x < mask_width; ++x) {
      for (int i = 0; i < 6; ++i) {
        // Create different patterns for each mask
        out_masks_data[i][y * mask_width + x] =
            data[y * mask_width * 6 + x * 6 + i];
      }
    }
  }
  for (int i = 0; i < 6; ++i) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer_ids[i]);
    GlCheckErrorOp("DeinterleaveMasksCpu - glBindBuffer");
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    mask_width * mask_height * sizeof(float),
                    out_masks_data[i].data());
    GlCheckErrorOp("DeinterleaveMasksCpu - glBufferSubData");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  }
  return true;
}

bool ImageProcessor::DeinterleaveMasks(GLuint input_buffer_id,
                                       std::vector<GLuint>& output_buffer_ids) {
  int mask_width = 256;
  int mask_height = 256;
  if (output_buffer_ids.size() != 6) {
    std::cerr << "ImageProcessor::DeinterleaveMasks: Requires exactly 6 output "
                 "buffers."
              << std::endl;
    return false;
  }
  if (deinterleave_masks_shader_program_ == 0) {
    std::cerr << "ImageProcessor::DeinterleaveMasks: Deinterleave masks shader "
                 "not initialized."
              << std::endl;
    return false;
  }

  glUseProgram(deinterleave_masks_shader_program_);
  GlCheckErrorOp("DeinterleaveMasks - glUseProgram");

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buffer_id);
  for (int i = 0; i < output_buffer_ids.size(); ++i) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1 + i, output_buffer_ids[i]);
  }
  GlCheckErrorOp("DeinterleaveMasks - glBindBufferBase");

  glUniform1i(
      glGetUniformLocation(deinterleave_masks_shader_program_, "mask_width"),
      mask_width);
  glUniform1i(
      glGetUniformLocation(deinterleave_masks_shader_program_, "mask_height"),
      mask_height);

  GlCheckErrorOp(
      "DeinterleaveMasks - set mask_width and mask_height "
      "uniform");

  glDispatchCompute((mask_width + 7) / 8, (mask_height + 7) / 8, 1);
  GlCheckErrorOp("DeinterleaveMasks - glDispatchCompute");
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  GlCheckErrorOp("DeinterleaveMasks - glMemoryBarrier");

  glFinish();
  GlCheckErrorOp("DeinterleaveMasks - glFinish");

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
  for (int i = 0; i < output_buffer_ids.size(); ++i) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1 + i, 0);
  }
  glUseProgram(0);

  return true;
}

void ImageProcessor::CleanupEGLContext() {
  if (egl_display_ != EGL_NO_DISPLAY && egl_context_ != EGL_NO_CONTEXT) {
    eglDestroyContext(egl_display_, egl_context_);
    egl_context_ = EGL_NO_CONTEXT;
  }
}

void ImageProcessor::CleanupEGLSurface() {
  if (egl_display_ != EGL_NO_DISPLAY && egl_surface_ != EGL_NO_SURFACE) {
    eglDestroySurface(egl_display_, egl_surface_);
    egl_surface_ = EGL_NO_SURFACE;
  }
}

void ImageProcessor::CleanupGLResources() {
  if (fbo_) {
    glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
  }
  if (mask_blend_compute_shader_program_) {
    glDeleteProgram(mask_blend_compute_shader_program_);
    mask_blend_compute_shader_program_ = 0;
  }
  if (resize_compute_shader_program_) {
    glDeleteProgram(resize_compute_shader_program_);
    resize_compute_shader_program_ = 0;
  }
  if (preprocess_compute_shader_program_) {
    glDeleteProgram(preprocess_compute_shader_program_);
    preprocess_compute_shader_program_ = 0;
  }
  if (deinterleave_masks_shader_program_) {
    glDeleteProgram(deinterleave_masks_shader_program_);
    deinterleave_masks_shader_program_ = 0;
  }
  GlCheckErrorOp("CleanupGLResources");
}
