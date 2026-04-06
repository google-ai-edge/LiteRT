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

#pragma once

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_samsung_options.h"
#include "litert/cc/litert_expected.h"

namespace litert::samsung {

class SamsungOptions {
 public:
  SamsungOptions() = delete;
  explicit SamsungOptions(LrtSamsungOptions options)
      : options_(options, LrtDestroySamsungOptions) {}

  // Delete copy constructor and assignment, keep move constructor
  SamsungOptions(const SamsungOptions &) = delete;
  SamsungOptions &operator=(const SamsungOptions &) = delete;
  SamsungOptions(SamsungOptions &&) = default;
  SamsungOptions &operator=(SamsungOptions &&) = default;

  ~SamsungOptions() = default;

  static const char *Discriminator();

  static Expected<SamsungOptions> Create();

  LrtSamsungOptions Release() { return options_.release(); }
  LrtSamsungOptions Get() const { return options_.get(); }

  LiteRtStatus GetOpaqueOptionsData(const char **identifier, void **payload,
                                    void (**payload_deleter)(void *)) const;

  /// @brief This option hints whether current model is LLM. This influences
  /// compilation behavior. Defaults to `false`.
  Expected<void> SetEnableLargeModelSupport(bool large_model_support);
  Expected<bool> GetEnableLargeModelSupport() const;

 private:
  std::unique_ptr<LrtSamsungOptionsT, void (*)(LrtSamsungOptions)> options_;
};

}  // namespace litert::samsung
