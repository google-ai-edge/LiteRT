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

#include "litert/cc/options/litert_compiler_options.h"

#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert {

Expected<CompilerOptions> CompilerOptions::Create() {
  LrtCompilerOptions* options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtCreateCompilerOptions(&options));
  return CompilerOptions(options);
}

CompilerOptions::CompilerOptions(LrtCompilerOptions* options)
    : options_(options) {}

Expected<void> CompilerOptions::SetPartitionStrategy(
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  LITERT_RETURN_IF_ERROR(LrtSetCompilerOptionsPartitionStrategy(
      options_.get(), partition_strategy));
  return {};
}

Expected<LiteRtCompilerOptionsPartitionStrategy>
CompilerOptions::GetPartitionStrategy() const {
  LiteRtCompilerOptionsPartitionStrategy strategy;
  LITERT_RETURN_IF_ERROR(
      LrtGetCompilerOptionsPartitionStrategy(options_.get(), &strategy));
  return strategy;
}

Expected<void> CompilerOptions::SetDummyOption(bool dummy_option) {
  LITERT_RETURN_IF_ERROR(
      LrtSetCompilerOptionsDummyOption(options_.get(), dummy_option));
  return {};
}

Expected<bool> CompilerOptions::GetDummyOption() const {
  bool dummy_option;
  LITERT_RETURN_IF_ERROR(
      LrtGetCompilerOptionsDummyOption(options_.get(), &dummy_option));
  return dummy_option;
}

}  // namespace litert
