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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_COMPILER_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_COMPILER_OPTIONS_H_

#include <memory>

#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT compiler options.
class CompilerOptions {
public:
  /// @brief Creates a new `CompilerOptions` instance with default values.
  static Expected<CompilerOptions> Create() {
    LrtCompilerOptions *options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateCompilerOptions(&options));
    return CompilerOptions(options);
  }

  /// @brief Sets the partition strategy.
  Expected<void> SetPartitionStrategy(
      LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
    LITERT_RETURN_IF_ERROR(LrtSetCompilerOptionsPartitionStrategy(
        options_.get(), partition_strategy));
    return {};
  }

  /// @brief Gets the partition strategy.
  Expected<LiteRtCompilerOptionsPartitionStrategy>
  GetPartitionStrategy() const {
    LiteRtCompilerOptionsPartitionStrategy strategy;
    LITERT_RETURN_IF_ERROR(
        LrtGetCompilerOptionsPartitionStrategy(options_.get(), &strategy));
    return strategy;
  }

  /// @brief Sets the dummy option for testing.
  Expected<void> SetDummyOption(bool dummy_option) {
    LITERT_RETURN_IF_ERROR(
        LrtSetCompilerOptionsDummyOption(options_.get(), dummy_option));
    return {};
  }

  /// @brief Gets the dummy option.
  Expected<bool> GetDummyOption() const {
    bool dummy_option;
    LITERT_RETURN_IF_ERROR(
        LrtGetCompilerOptionsDummyOption(options_.get(), &dummy_option));
    return dummy_option;
  }

  /// @brief Returns the underlying C handle.
  LrtCompilerOptions *Get() { return options_.get(); }
  const LrtCompilerOptions *Get() const { return options_.get(); }

private:
  explicit CompilerOptions(LrtCompilerOptions *options) : options_(options) {}

  struct Deleter {
    void operator()(LrtCompilerOptions *ptr) const {
      LrtDestroyCompilerOptions(ptr);
    }
  };
  std::unique_ptr<LrtCompilerOptions, Deleter> options_;
};

} // namespace litert

#endif // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_COMPILER_OPTIONS_H_
