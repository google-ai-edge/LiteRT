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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/cc/options/compiler_options.h"

#include <gtest/gtest.h>
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(CompilerOptionsTest, CreateSetAndGetDummyOptionWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(::litert::CompilerOptions options,
                              ::litert::CompilerOptions::Create());
  LITERT_EXPECT_OK(options.SetDummyOption(true));
  EXPECT_TRUE(options.GetDummyOption());
}

}  // namespace
}  // namespace litert
