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

#include "litert/vendors/intel_openvino//dispatch/openvino_shared_core.h"

#include <exception>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"

OpenVINOSharedCore::OpenVINOSharedCore()
    : core_(std::make_shared<ov::Core>()) {}

OpenVINOSharedCore::~OpenVINOSharedCore() = default;

// static
OpenVINOSharedCore* OpenVINOSharedCore::GetInstance() {
  static OpenVINOSharedCore* instance = new OpenVINOSharedCore();
  return instance;
}

const std::vector<std::string>& OpenVINOSharedCore::GetAvailableDevices() {
  std::call_once(available_devices_once_, [this]() {
    try {
      available_devices_ = core_->get_available_devices();
    } catch (const std::exception&) {
      available_devices_.clear();
    }
  });
  return available_devices_;
}
