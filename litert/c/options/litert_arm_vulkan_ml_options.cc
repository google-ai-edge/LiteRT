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

#include "litert/c/options/litert_arm_vulkan_ml_options.h"

#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtArmVulkanMLOptionsT {
  bool enable_just_in_time = false;
  LrtArmVulkanMLSoftmaxLutType softmax_lut_type = kLrtArmVulkanMLSoftmaxLutAuto;
  LrtArmVulkanMLOptimizationMode optimization_mode =
      kLrtArmVulkanMLOptimizationQualified;
};

const char* LrtArmVulkanMLOptionsGetIdentifier() { return "ArmVulkanML"; }

LiteRtStatus LrtCreateArmVulkanMLOptions(LrtArmVulkanMLOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *options = new LrtArmVulkanMLOptionsT;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtCreateArmVulkanMLOptionsFromToml(
    const char* toml_payload, LrtArmVulkanMLOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LrtCreateArmVulkanMLOptions(options));

  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  LrtArmVulkanMLOptionsT& options_ref = **options;
  auto status = litert::internal::ParseToml(
      toml_payload,
      [&options_ref](absl::string_view key,
                     absl::string_view value) -> LiteRtStatus {
        if (key == "enable_just_in_time") {
          LITERT_ASSIGN_OR_RETURN(options_ref.enable_just_in_time,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "softmax_lut_type") {
          if (value == "auto") options_ref.softmax_lut_type = kLrtArmVulkanMLSoftmaxLutAuto;
          else if (value == "int8") options_ref.softmax_lut_type = kLrtArmVulkanMLSoftmaxLutInt8;
          else if (value == "int16") options_ref.softmax_lut_type = kLrtArmVulkanMLSoftmaxLutInt16;
          else return kLiteRtStatusErrorInvalidArgument;
        } else if (key == "optimization_mode") {
          // ParseToml has already unquoted scalar string values.
          auto mode = value;
          if (mode == "off") {
            options_ref.optimization_mode = kLrtArmVulkanMLOptimizationOff;
          } else if (mode == "exact") {
            options_ref.optimization_mode = kLrtArmVulkanMLOptimizationExact;
          } else if (mode == "qualified") {
            options_ref.optimization_mode = kLrtArmVulkanMLOptimizationQualified;
          } else {
            return kLiteRtStatusErrorInvalidArgument;
          }
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyArmVulkanMLOptions(*options);
    *options = nullptr;
  }

  return status;
}

void LrtDestroyArmVulkanMLOptions(LrtArmVulkanMLOptions options) {
  delete options;
}

LiteRtStatus LrtGetOpaqueArmVulkanMLOptionsData(
    LrtArmVulkanMLOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *identifier = LrtArmVulkanMLOptionsGetIdentifier();

  std::string toml = absl::StrCat(
      "enable_just_in_time = ",
      (options->enable_just_in_time ? "true" : "false"), "\noptimization_mode = \"",
      options->optimization_mode == kLrtArmVulkanMLOptimizationOff ? "off" :
      options->optimization_mode == kLrtArmVulkanMLOptimizationExact ? "exact" : "qualified",
      "\"\nsoftmax_lut_type = \"",
      options->softmax_lut_type == kLrtArmVulkanMLSoftmaxLutInt8 ? "int8" :
      options->softmax_lut_type == kLrtArmVulkanMLSoftmaxLutInt16 ? "int16" : "auto",
      "\"\n");

  litert::internal::MakeCStringPayload(toml, payload, payload_deleter);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtArmVulkanMLOptionsSetEnableJustInTime(
    LrtArmVulkanMLOptions options, bool enable_just_in_time) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_just_in_time = enable_just_in_time;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtArmVulkanMLOptionsGetEnableJustInTime(
    LrtArmVulkanMLOptions options, bool* enable_just_in_time) {
  if (options == nullptr || enable_just_in_time == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_just_in_time = options->enable_just_in_time;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtArmVulkanMLOptionsSetOptimizationMode(
    LrtArmVulkanMLOptions options, LrtArmVulkanMLOptimizationMode mode) {
  if (options == nullptr ||
      (mode != kLrtArmVulkanMLOptimizationOff &&
       mode != kLrtArmVulkanMLOptimizationExact &&
       mode != kLrtArmVulkanMLOptimizationQualified)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->optimization_mode = mode;
  return kLiteRtStatusOk;
}
LiteRtStatus LrtArmVulkanMLOptionsGetOptimizationMode(
    LrtArmVulkanMLOptions options, LrtArmVulkanMLOptimizationMode* mode) {
  if (options == nullptr || mode == nullptr) return kLiteRtStatusErrorInvalidArgument;
  *mode = options->optimization_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtArmVulkanMLOptionsSetSoftmaxLutType(
    LrtArmVulkanMLOptions options, LrtArmVulkanMLSoftmaxLutType type) {
  if (!options || (type != kLrtArmVulkanMLSoftmaxLutAuto &&
      type != kLrtArmVulkanMLSoftmaxLutInt8 && type != kLrtArmVulkanMLSoftmaxLutInt16))
    return kLiteRtStatusErrorInvalidArgument;
  options->softmax_lut_type = type;
  return kLiteRtStatusOk;
}
LiteRtStatus LrtArmVulkanMLOptionsGetSoftmaxLutType(
    LrtArmVulkanMLOptions options, LrtArmVulkanMLSoftmaxLutType* type) {
  if (!options || !type) return kLiteRtStatusErrorInvalidArgument;
  *type = options->softmax_lut_type;
  return kLiteRtStatusOk;
}
