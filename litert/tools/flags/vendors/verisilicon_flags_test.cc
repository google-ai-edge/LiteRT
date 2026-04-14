// Copyright 2025 Vivante Corporation.
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

#include "litert/tools/flags/vendors/verisilicon_flags.h"

#include <gtest/gtest.h>

#include <string>

#include "absl/flags/flag.h"           // from @com_google_absl
#include "absl/flags/marshalling.h"    // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_verisilicon_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_verisilicon_options.h"

namespace litert::verisilicon {
namespace {

TEST(UpdateVerisiliconOptionsFromFlagsTest, DefaultValue) {
  Expected<VerisiliconOptions> options = VerisiliconOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateVerisiliconOptionsFromFlags(options.Value()).HasValue());
  auto device_idx = options.Value().GetDeviceIndex();
  EXPECT_EQ(device_idx.Value(), 0);
}

TEST(UpdateVerisiliconOptionsFromFlagsTest, SetFlagToDevice1) {
  absl::SetFlag(&FLAGS_verisilicon_device_index, 1);
  Expected<VerisiliconOptions> options = VerisiliconOptions::Create();

  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateVerisiliconOptionsFromFlags(options.Value()).HasValue());
  auto device_idx = options.Value().GetDeviceIndex();
  EXPECT_EQ(device_idx.Value(), 1);
}

TEST(UpdateVerisiliconOptionsFromFlagsTest, SetFlagToCore1) {
  absl::SetFlag(&FLAGS_verisilicon_core_index, 1);
  Expected<VerisiliconOptions> options = VerisiliconOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateVerisiliconOptionsFromFlags(options.Value()).HasValue());
  auto core_idx = options.Value().GetCoreIndex();
  EXPECT_EQ(core_idx.Value(), 1);
}

}  // namespace

}  // namespace litert::verisilicon
