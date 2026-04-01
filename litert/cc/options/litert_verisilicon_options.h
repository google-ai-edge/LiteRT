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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_VERISILICON_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_VERISILICON_OPTIONS_H_

#include <memory>
#include <string>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_verisilicon_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"

namespace litert::verisilicon {

// Wraps a LiteRtVerisiliconOptions object for convenience.
class VerisiliconOptions {
 public:
  VerisiliconOptions() : options_(nullptr) {}

  explicit VerisiliconOptions(LrtVerisiliconOptions options): options_(options) {}
  ~VerisiliconOptions() {
    if (options_) {
      LrtDestroyVerisiliconOptions(options_);
    }
  }
  VerisiliconOptions(VerisiliconOptions&& other) noexcept : options_(other.options_) {
    other.options_ = nullptr;
  }
  VerisiliconOptions& operator=(VerisiliconOptions&& other) noexcept {
    if (this != &other) {
      if (options_) LrtDestroyVerisiliconOptions(options_);
      options_ = other.options_;
      other.options_ = nullptr;
    }
    return *this;
  }

  // Delete copy constructor and assignment
  VerisiliconOptions(const VerisiliconOptions&) = delete;
  VerisiliconOptions& operator=(const VerisiliconOptions&) = delete;


  LrtVerisiliconOptions Get() const { return options_; }
  LrtVerisiliconOptions Release() {
    auto* res = options_;
    options_ = nullptr;
    return res; }
  static const char* Discriminator() {
    return LrtVerisiliconOptionsGetIdentifier();
  }
  static Expected<VerisiliconOptions> Create();

  Expected<void> SetDeviceIndex(unsigned int device_index);
  Expected<unsigned int> GetDeviceIndex() const;
  Expected<void> SetCoreIndex(unsigned int core_index);
  Expected<unsigned int> GetCoreIndex() const;
  Expected<void> SetTimeOut(unsigned int time);
  Expected<unsigned int> GetTimeOut() const;
  Expected<void> SetProfileLevel(unsigned int level);
  Expected<unsigned int> GetProfileLevel() const;
  Expected<void> SetDumpNBG(bool enable);
  Expected<bool> GetDumpNBG() const;

 private:
  LrtVerisiliconOptions options_;
};

}  // namespace litert::verisilicon

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_VERISILICON_OPTIONS_H_
