// Copyright 2024 Google LLC.
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

#include "litert/runtime/dispatch/dispatch_opaque_options.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::internal {
namespace {

TEST(DispatchDelegateOptionsTest, Create) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);
  ASSERT_TRUE(options->GetAllocBase());
  ASSERT_TRUE(options->GetAllocBaseFd());
}

TEST(DispatchDelegateOptionsTest, CreateFromOpaqueOptions) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);

  auto other =
      DispatchDelegateOptions::Create(dynamic_cast<OpaqueOptions&>(*options));
  ASSERT_TRUE(other);

  auto alloc_base = options->GetAllocBase();
  ASSERT_TRUE(alloc_base);

  auto other_alloc_base = other->GetAllocBase();
  ASSERT_TRUE(other_alloc_base);
  ASSERT_EQ(*alloc_base, *other_alloc_base);
}

TEST(DispatchDelegateOptionsTest, SetAllocBase) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);

  int dummy_val = 1;
  void* dummy_addr = &dummy_val;

  ASSERT_TRUE(options->SetAllocBase(dummy_addr));
  auto alloc_base = options->GetAllocBase();
  ASSERT_TRUE(alloc_base);
  ASSERT_EQ(*alloc_base, dummy_addr);
}

TEST(DispatchDelegateOptionsTest, SetAllocBaseFd) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);

  int dummy_fd = 1;

  ASSERT_TRUE(options->SetAllocBaseFd(dummy_fd));
  auto alloc_base_fd = options->GetAllocBaseFd();
  ASSERT_TRUE(alloc_base_fd);
  ASSERT_EQ(*alloc_base_fd, dummy_fd);
}

TEST(DispatchDelegateOptionsTest, SetAllocBaseSize) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);

  constexpr size_t kAllocBaseSize = 1234;
  ASSERT_TRUE(options->SetAllocBaseSize(kAllocBaseSize));
  auto alloc_base_size = options->GetAllocBaseSize();
  ASSERT_TRUE(alloc_base_size);
  EXPECT_EQ(*alloc_base_size, kAllocBaseSize);
}

TEST(DispatchDelegateOptionsTest, SetModelSourcePath) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);

  constexpr absl::string_view kPath = "/tmp/model.tflite";
  ASSERT_TRUE(options->SetModelSourcePath(kPath));
  auto model_source_path = options->GetModelSourcePath();
  ASSERT_TRUE(model_source_path);
  EXPECT_EQ(*model_source_path, kPath);
}

TEST(DispatchDelegateOptionsTest, SetDispatchManifest) {
  auto options = DispatchDelegateOptions::Create();
  ASSERT_TRUE(options);

  constexpr absl::string_view kManifest = "manifest";
  ASSERT_TRUE(options->SetDispatchManifest(
      BufferRef<uint8_t>(kManifest.data(), kManifest.size())));

  auto manifest = options->GetDispatchManifest();
  ASSERT_TRUE(manifest);
  EXPECT_EQ(manifest->StrView(), kManifest);
}

}  // namespace
}  // namespace litert::internal
