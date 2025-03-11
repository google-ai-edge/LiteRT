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

#include "litert/core/accelerator_model_compilation_data.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_accelerator_options.h"
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {
using testing::Eq;
using testing::StrEq;

TEST(ModelCompilationDataTest, CreateSetsUpAllNecessaryFields) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto ptr, ModelCompilationData::Create());

  const char* identifier = nullptr;
  LITERT_EXPECT_OK(
      LiteRtGetAcceleratorCompilationOptionsIdentifier(ptr.get(), &identifier));
  EXPECT_THAT(identifier, StrEq(ModelCompilationData::kIdentifier));

  LiteRtApiVersion version;
  LITERT_EXPECT_OK(
      LiteRtGetAcceleratorCompilationOptionsVersion(ptr.get(), &version));
  EXPECT_THAT(LiteRtCompareApiVersion(version, ModelCompilationData::kVersion),
              Eq(0));
}

}  // namespace

}  // namespace litert
