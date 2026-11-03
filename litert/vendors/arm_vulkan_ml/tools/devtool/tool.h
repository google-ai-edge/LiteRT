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
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/vendors/arm_vulkan_ml/compiler/graph_ir.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include "litert/vendors/arm_vulkan_ml/tools/devtool/devtool.pb.h"
#include "google/protobuf/message.h"  // from @com_google_protobuf
#include "tflite/interpreter.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml::tool {
struct Options {
  std::string command;
  std::map<std::string, std::string> values;
  std::map<std::string, std::vector<std::string>> repeated;
  std::string Get(const std::string& key,
                  const std::string& fallback = "") const;
  Expected<SoftmaxLutType> SoftmaxLut() const;
  Expected<int> RescaleMultiplierBits() const;
  Expected<int> Integer(const std::string& key, int fallback) const;
  Expected<double> Number(const std::string& key, double fallback) const;
};
Expected<void> ReadTextProto(const std::string& path, google::protobuf::Message& message);
Expected<void> WriteTextProto(const std::string& path,
                              const google::protobuf::Message& message);
Expected<std::string> FormatTextProto(const google::protobuf::Message& message);
Expected<std::string> HashFile(const std::string& path);
std::string CompilerDirectory(const Options& options);
Expected<litert::Options> RuntimeOptions(const Options& options);
Expected<Options> ParseOptions(int argc, char** argv);
Expected<std::unique_ptr<tflite::ModelT>> LoadModel(const std::string& path);
Expected<std::vector<uint8_t>> PackModel(const tflite::ModelT& model);
Expected<void> Inspect(const Options& options, ToolReport& report,
                       bool compile);
Expected<void> Capabilities(const Options& options, ToolReport& report);
Expected<void> Validate(const Options& options, ToolReport& report);
Expected<std::vector<int8_t>> SoftmaxReference(const std::vector<int8_t>& input,
                                               size_t channels,
                                               double input_scale, double beta,
                                               const std::string& reference);
Expected<std::vector<int8_t>> QuantizedReference(
    const tflite::ModelT& model, const std::vector<int8_t>& input);
Expected<void> Trace(const Options& options, ToolReport& report);
Expected<void> Compare(const void* actual, const void* expected, size_t count,
                       bool floating, const Options& options,
                       OutputComparisonReport& result);
Expected<void> ExportVgf(const CompiledPartition&, const std::string& name,
                         const std::string& path);
Expected<void> Coverage(CoverageReport& report);
void PrintCoverage(const CoverageReport& report);
Expected<void> Run(const Options& options, ToolReport& report);
Expected<void> CollectValidationLogs(const std::string& directory,
                                     const std::string& mode,
                                     VulkanValidationReport& result);
std::string NumericalStatus(const ToolReport& report);
std::string TosaOperatorName(uint32_t opcode);
int Main(int argc, char** argv);
}  // namespace litert::arm_vulkan_ml::tool
#endif
