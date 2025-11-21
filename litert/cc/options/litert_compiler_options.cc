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

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert {

Expected<CompilerOptions> CompilerOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateCompilerOptions(&options));
  return CompilerOptions(options, litert::OwnHandle::kYes);
}

Expected<CompilerOptions> CompilerOptions::Create(OpaqueOptions& original) {
  const auto id = original.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return CompilerOptions(original.Get(), OwnHandle::kNo);
}

Expected<void> CompilerOptions::SetPartitionStrategy(
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  LiteRtCompilerOptions compiler_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCompilerOptions(Get(), &compiler_options));
  LITERT_RETURN_IF_ERROR(LiteRtSetCompilerOptionsPartitionStrategy(
      compiler_options, partition_strategy));
  return {};
}

Expected<LiteRtCompilerOptionsPartitionStrategy>
CompilerOptions::GetPartitionStrategy() const {
  LiteRtCompilerOptions compiler_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCompilerOptions(Get(), &compiler_options));
  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  LITERT_RETURN_IF_ERROR(LiteRtGetCompilerOptionsPartitionStrategy(
      compiler_options, &partition_strategy));
  return partition_strategy;
}

Expected<void> CompilerOptions::SetDummyOption(bool dummy_option) {
  LiteRtCompilerOptions compiler_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCompilerOptions(Get(), &compiler_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDummyCompilerOptions(compiler_options, dummy_option));
  return {};
}

Expected<bool> CompilerOptions::GetDummyOption() const {
  LiteRtCompilerOptions compiler_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCompilerOptions(Get(), &compiler_options));
  bool dummy_option;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDummyCompilerOptions(compiler_options, &dummy_option));
  return dummy_option;
}

}  // namespace litert
