
// Copyright 2025 Google LLC.
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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/delegates/gpu/gl/egl_environment.h"

namespace {

using ::testing::HasSubstr;

TEST(AngleTest, CheckAngle) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  ASSERT_TRUE(env->gpu_info().IsApiOpenGl());
  EXPECT_THAT(env->gpu_info().opengl_info.vendor_name, HasSubstr("Google"));
  EXPECT_THAT(env->gpu_info().opengl_info.version, HasSubstr("ANGLE"));
  EXPECT_THAT(env->gpu_info().opengl_info.renderer_name, HasSubstr("ANGLE"));
}

}  // namespace
