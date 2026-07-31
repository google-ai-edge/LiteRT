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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//

#ifndef ODML_LITERT_LITERT_CC_OPTIONS_LITERT_ARM_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_OPTIONS_LITERT_ARM_OPTIONS_H_

#include "litert/c/options/litert_arm_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::arm {

class ArmOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  ArmOptions() = delete;

  static const char* Discriminator() { return LiteRtArmOptionsGetIdentifier(); }

  static Expected<ArmOptions> Create() {
    LiteRtOpaqueOptions options = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtArmOptionsCreate(&options));
    return ArmOptions(options, OwnHandle::kYes);
  }

  static Expected<ArmOptions> Create(OpaqueOptions& options) {
    LiteRtArmOptions arm_options = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtArmOptionsGet(options.Get(), &arm_options));
    return ArmOptions(options.Get(), OwnHandle::kNo);
  }

  Expected<void> SetEnableJustInTime(bool enable_just_in_time) {
    internal::AssertOk(LiteRtArmOptionsSetEnableJustInTime, Data(),
                       enable_just_in_time);
    return {};
  }

  Expected<bool> GetEnableJustInTime() const {
    bool enable_just_in_time = false;
    internal::AssertOk(LiteRtArmOptionsGetEnableJustInTime, Data(),
                       &enable_just_in_time);
    return enable_just_in_time;
  }

  LiteRtStatus GetOpaqueOptionsData(const char** identifier, void** payload,
                                    void (**payload_deleter)(void*)) const {
    return LrtGetOpaqueArmOptionsData(Get(), identifier, payload,
                                      payload_deleter);
  }

 private:
  LiteRtArmOptions Data() const {
    LiteRtArmOptions arm_options = nullptr;
    internal::AssertOk(LiteRtArmOptionsGet, Get(), &arm_options);
    return arm_options;
  }
};

}  // namespace litert::arm

#endif  // ODML_LITERT_LITERT_CC_OPTIONS_LITERT_ARM_OPTIONS_H_
