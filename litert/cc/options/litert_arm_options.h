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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
//

#ifndef ODML_LITERT_LITERT_CC_OPTIONS_LITERT_ARM_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_OPTIONS_LITERT_ARM_OPTIONS_H_

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_arm_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_concrete_options_base.h"

namespace litert::arm {

class ArmOptions : public ConcreteOptionsBase {
 public:
  ArmOptions() = delete;
  explicit ArmOptions(LrtArmOptions options) : options_(options) {}

  ArmOptions(const ArmOptions&) = delete;
  ArmOptions& operator=(const ArmOptions&) = delete;
  ArmOptions(ArmOptions&&) = default;
  ArmOptions& operator=(ArmOptions&&) = default;

  static const char* Discriminator() { return LrtArmOptionsGetIdentifier(); }

  static Expected<ArmOptions> Create() {
    LrtArmOptions options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateArmOptions(&options));
    return ArmOptions(options);
  }

  Expected<void> SetEnableJustInTime(bool enable_just_in_time) {
    internal::AssertOk(LrtArmOptionsSetEnableJustInTime, Get(),
                       enable_just_in_time);
    return {};
  }

  Expected<bool> GetEnableJustInTime() const {
    bool enable_just_in_time = false;
    internal::AssertOk(LrtArmOptionsGetEnableJustInTime, Get(),
                       &enable_just_in_time);
    return enable_just_in_time;
  }

  LrtArmOptions Get() const { return options_.get(); }
  LrtArmOptions Release() { return options_.release(); }

  LiteRtStatus GetOpaqueOptionsData(
      const char** identifier, void** payload,
      void (**payload_deleter)(void*)) const override {
    return LrtGetOpaqueArmOptionsData(Get(), identifier, payload,
                                      payload_deleter);
  }

 private:
  struct Deleter {
    void operator()(LrtArmOptions options) const {
      LrtDestroyArmOptions(options);
    }
  };

  std::unique_ptr<LrtArmOptionsT, Deleter> options_;
};

}  // namespace litert::arm

#endif  // ODML_LITERT_LITERT_CC_OPTIONS_LITERT_ARM_OPTIONS_H_
