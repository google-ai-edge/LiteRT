// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_ARM_VULKAN_ML_DEVTOOL_H_
#define LITERT_ARM_VULKAN_ML_DEVTOOL_H_
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include <map>
#include <string>
#include <vector>
#include "litert/tools/imagenet_eval_data.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "tflite/schema/schema_generated.h"
namespace litert::arm_vulkan_ml::tool {
using imagenet::Json;
struct Options {
  std::string command;
  std::map<std::string, std::string> values;
  std::string Get(const std::string& key, const std::string& fallback = "") const;
  Expected<int> Integer(const std::string& key, int fallback) const;
  Expected<double> Number(const std::string& key, double fallback) const;
};
Expected<Options> ParseOptions(int argc, char** argv);
Expected<std::unique_ptr<tflite::ModelT>> LoadModel(const std::string& path);
Expected<std::vector<uint8_t>> PackModel(const tflite::ModelT& model);
Expected<void> Inspect(const Options& options, Json& report, bool compile);
Expected<void> Capabilities(const Options& options, Json& report);
Expected<void> Validate(const Options& options, Json& report);
Expected<void> Trace(const Options& options, Json& report);
Expected<void> Compare(const void* actual, const void* expected, size_t count,
                       bool floating, const Options& options, Json& result);
Expected<void> Run(const Options& options, Json& report);
int Main(int argc, char** argv);
}  // namespace litert::arm_vulkan_ml::tool
#endif
