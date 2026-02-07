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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_

#include <memory>

#include "openvino/runtime/core.hpp"

class OpenVINOSharedCore {
 public:
  OpenVINOSharedCore(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore(OpenVINOSharedCore&&) = delete;
  OpenVINOSharedCore& operator=(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore& operator=(OpenVINOSharedCore&&) = delete;

  static OpenVINOSharedCore* GetInstance();

  // Return the core shared_pointer.
  std::shared_ptr<ov::Core> getCore() const { return core_; }

 private:
  OpenVINOSharedCore();
  ~OpenVINOSharedCore();

  std::shared_ptr<ov::Core> core_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
