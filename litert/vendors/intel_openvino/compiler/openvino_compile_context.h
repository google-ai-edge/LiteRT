// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_OPENVINO_COMPILE_CONTEXT_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_OPENVINO_COMPILE_CONTEXT_H_

#include <memory>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"

namespace litert {
namespace openvino {

// Holds all per-compilation OpenVINO state: target device, runtime
// properties, SoC platform settings, and device-specific model optimization
// passes.  This is the context object passed through a single Compile() call.
class OpenVinoCompileContext {
 public:
  // Creates an OpenVinoCompileContext from the given options.  If |opts| has
  // no value the returned context holds the default configuration (NPU
  // device, LATENCY mode); otherwise it is populated from the contained
  // IntelOpenVinoOptions.
  static ::litert::Expected<OpenVinoCompileContext> Create(
      const ::litert::Expected<
          ::litert::intel_openvino::IntelOpenVinoOptions>& opts);

  // If the target device is NPU, applies SoC-model-specific compilation
  // parameters (e.g. NPU_PLATFORM).  |soc_model| may be nullptr.
  LiteRtStatus ConfigureForSoc(const char* soc_model);

  // Runs NPU-specific optimization passes on the given OV model.
  void OptimizeModel(const std::shared_ptr<ov::Model>& model) const;

  const std::string& Device() const { return device_; }
  const ov::AnyMap& ConfigsMap() const { return configs_map_; }

 private:
  // Applies default configuration (NPU device, LATENCY mode).  Construction
  // must go through Create().
  OpenVinoCompileContext();

  std::string device_ = "NPU";
  ov::AnyMap configs_map_;
};

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_OPENVINO_COMPILE_CONTEXT_H_
