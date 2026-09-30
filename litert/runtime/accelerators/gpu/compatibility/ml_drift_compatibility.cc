// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/runtime/accelerators/gpu/compatibility/ml_drift_compatibility.h"

#include <cstddef>
#include <map>
#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "litert/c/internal/litert_logging.h"
#include "tflite/experimental/acceleration/compatibility/android_info.h"
#include "tflite/experimental/acceleration/compatibility/canonicalize_value.h"
#include "tflite/experimental/acceleration/compatibility/database_generated.h"
#include "tflite/experimental/acceleration/compatibility/devicedb.h"
#include "tflite/experimental/acceleration/compatibility/mldrift_compatibility_binary.h"
#include "tflite/experimental/acceleration/compatibility/variables.h"

#if defined(__ANDROID__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <thread>  // NOLINT: only used on Android, where std::thread is allowed
#endif             // defined(__ANDROID__)

namespace litert::ml_drift {

using ::tflite::acceleration::AndroidInfo;

namespace {

// Builds the database lookup variables, normalized the same way the database
// values were normalized when the database was generated.
std::map<std::string, std::string> BuildCanonicalVariables(
    const AndroidInfo& android_info, absl::string_view gl_renderer,
    int gles_major, int gles_minor) {
  namespace acceleration = ::tflite::acceleration;
  std::map<std::string, std::string> variables = {
      {acceleration::kAndroidSdkVersion, android_info.android_sdk_version},
      {acceleration::kDeviceModel, android_info.model},
      {acceleration::kDeviceName, android_info.device},
      {acceleration::kManufacturer, android_info.manufacturer},
      {acceleration::kGPUModel, std::string(gl_renderer)},
      {acceleration::kOpenGLESVersion,
       absl::StrCat(gles_major, ".", gles_minor)},
  };
  for (auto& [key, value] : variables) {
    value = acceleration::CanonicalizeValueWithKey(key, value);
  }
  return variables;
}

#if defined(__ANDROID__)
struct GlInfo {
  std::string renderer;
  GLint major_version = 0;
  GLint minor_version = 0;
};

// Reads the GL renderer and GLES version through a temporary 1x1 pbuffer
// context. Leaves `gl_info` empty if any EGL step fails.
void ReadGlInfo(GlInfo* gl_info) {
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display == EGL_NO_DISPLAY ||
      eglInitialize(display, nullptr, nullptr) != EGL_TRUE) {
    return;
  }

  // GL_MAJOR_VERSION and GL_MINOR_VERSION require a GLES 3 context.
  const EGLint config_attributes[] = {EGL_RENDERABLE_TYPE,
                                      EGL_OPENGL_ES3_BIT_KHR, EGL_SURFACE_TYPE,
                                      EGL_PBUFFER_BIT, EGL_NONE};
  const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
  EGLConfig config;
  EGLint num_configs = 0;
  EGLContext context = EGL_NO_CONTEXT;
  EGLSurface surface = EGL_NO_SURFACE;
  if (eglChooseConfig(display, config_attributes, &config, 1, &num_configs) ==
          EGL_TRUE &&
      num_configs > 0) {
    context =
        eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    surface = eglCreatePbufferSurface(display, config, surface_attributes);
  }

  if (context != EGL_NO_CONTEXT && surface != EGL_NO_SURFACE &&
      eglMakeCurrent(display, surface, surface, context) == EGL_TRUE) {
    const GLubyte* renderer = glGetString(GL_RENDERER);
    if (renderer != nullptr) {
      gl_info->renderer = reinterpret_cast<const char*>(renderer);
    }
    glGetIntegerv(GL_MAJOR_VERSION, &gl_info->major_version);
    glGetIntegerv(GL_MINOR_VERSION, &gl_info->minor_version);
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  }

  if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
  if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
  // Pairs the eglInitialize above (b/479543747). Android's libEGL ref-counts
  // display initialization, so this does not terminate a display that is still
  // initialized elsewhere in the process.
  eglTerminate(display);
}
#endif  // defined(__ANDROID__)

}  // namespace

bool IsMlDriftGpuSupported(const unsigned char* compatibility_binary,
                           size_t compatibility_binary_len,
                           const AndroidInfo& android_info,
                           absl::string_view gl_renderer, int gles_major,
                           int gles_minor) {
  if (!compatibility_binary || compatibility_binary_len == 0) {
    return false;
  }
  flatbuffers::Verifier verifier(compatibility_binary,
                                 compatibility_binary_len);
  if (!tflite::acceleration::VerifyDeviceDatabaseBuffer(verifier)) {
    LITERT_LOG(LITERT_WARNING,
               "IsMlDriftGpuSupported: Failed to parse compatibility binary.");
    return false;
  }

  std::map<std::string, std::string> variables = BuildCanonicalVariables(
      android_info, gl_renderer, gles_major, gles_minor);
  tflite::acceleration::UpdateVariablesFromDatabase(
      &variables, *flatbuffers::GetRoot<tflite::acceleration::DeviceDatabase>(
                      compatibility_binary));
  return variables[tflite::acceleration::gpu::kStatus] ==
         tflite::acceleration::gpu::kStatusSupported;
}

bool IsMlDriftGpuSupported(const AndroidInfo& android_info,
                           absl::string_view gl_renderer, int gles_major,
                           int gles_minor) {
  return IsMlDriftGpuSupported(
      g_tflite_acceleration_mldrift_compatibility_binary,
      g_tflite_acceleration_mldrift_compatibility_binary_len, android_info,
      gl_renderer, gles_major, gles_minor);
}

bool IsMlDriftGpuSupportedOnThisDevice() {
#if defined(__ANDROID__)
  static const bool is_supported = []() {
    AndroidInfo android_info;
    auto android_status =
        tflite::acceleration::RequestAndroidInfo(&android_info);
    if (!android_status.ok()) {
      LITERT_LOG(
          LITERT_WARNING,
          "IsMlDriftGpuSupportedOnThisDevice: RequestAndroidInfo failed: %s",
          android_status.ToString().c_str());
      return false;
    }

    // Read GL info on a separate thread so that the EGL context and bindings of
    // the calling thread are not modified.
    GlInfo gl_info;
    std::thread(ReadGlInfo, &gl_info).join();
    if (gl_info.renderer.empty()) {
      LITERT_LOG(LITERT_WARNING,
                 "IsMlDriftGpuSupportedOnThisDevice: Failed to read the GL "
                 "renderer.");
    }

    return IsMlDriftGpuSupported(android_info, gl_info.renderer,
                                 gl_info.major_version, gl_info.minor_version);
  }();
  return is_supported;
#else
  return true;
#endif  // defined(__ANDROID__)
}

}  // namespace litert::ml_drift
