// Copyright (C) 2025 Intel Corporation
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

#include <cstddef>
#include <ios>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

constexpr char kPluginManufacturer[] = "IntelOpenVINO";

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflConv2d,
    kLiteRtOpCodeTflDepthwiseConv2d,
    kLiteRtOpCodeTflSplit,
    kLiteRtOpCodeTflFullyConnected,
    kLiteRtOpCodeTflAdd,
    kLiteRtOpCodeTflReshape,
    kLiteRtOpCodeTflMean,
    kLiteRtOpCodeTflResizeBilinear,
    kLiteRtOpCodeTflResizeNearestNeighbor,
    kLiteRtOpCodeTflConcatenation,
    kLiteRtOpCodeTflMaxPool2d,
    kLiteRtOpCodeTflAveragePool2d,
    kLiteRtOpCodeTflMul,
    kLiteRtOpCodeTflTransposeConv,
    kLiteRtOpCodeTflSoftmax,
    kLiteRtOpCodeTflMirrorPad,
    kLiteRtOpCodeTflStridedSlice,
    kLiteRtOpCodeTflDepthToSpace,
    kLiteRtOpCodeTflGather,
    kLiteRtOpCodeTflBatchMatmul,
    kLiteRtOpCodeTflLeakyRelu,
    kLiteRtOpCodeTflPack,
    kLiteRtOpCodeTflCast,
    kLiteRtOpCodeTflDiv,
    kLiteRtOpCodeTflCumsum,
    kLiteRtOpCodeTflSub,
    kLiteRtOpCodeTflGelu,
    kLiteRtOpCodeTflGatherNd,
    kLiteRtOpCodeTflSum,
    kLiteRtOpCodeTflReduceMax,
    kLiteRtOpCodeTflEmbeddingLookup,
    kLiteRtOpCodeTflConv3d,
    kLiteRtOpCodeTflArgMax,
    kLiteRtOpCodeTflOneHot,
    kLiteRtOpCodeTflUnpack,
    // These ops donot call get_attribute
    kLiteRtOpCodeTflDequantize,
    kLiteRtOpCodeTflLogistic,
    kLiteRtOpCodeTflRelu,
    kLiteRtOpCodeTflTanh,
    kLiteRtOpCodeTflPad,
    kLiteRtOpCodeTflTranspose,
    kLiteRtOpCodeTflSlice,
    kLiteRtOpCodeTflQuantize,
    kLiteRtOpCodeTflRange,
    kLiteRtOpCodeTflBroadcastTo,
    kLiteRtOpCodeTflPadv2,
    kLiteRtOpCodeTflEqual,
    kLiteRtOpCodeTflNotEqual,
    kLiteRtOpCodeTflExp,
    kLiteRtOpCodeTflReverseV2,
    kLiteRtOpCodeTflMaximum,
    kLiteRtOpCodeTflLogicalOr,
    kLiteRtOpCodeTflExpandDims,
    kLiteRtOpCodeTflLog,
    kLiteRtOpCodeTflSin,
    kLiteRtOpCodeTflPow,
    kLiteRtOpCodeTflFloorDiv,
    kLiteRtOpCodeTflCos,
    kLiteRtOpCodeTflMinimum,
    kLiteRtOpCodeTflSquaredDifference,
    kLiteRtOpCodeTflRsqrt,
    kLiteRtOpCodeTflAbs,
    kLiteRtOpCodeTflLess,
    kLiteRtOpCodeTflSelect,
    kLiteRtOpCodeTflSelectV2,
    kLiteRtOpCodeTflHardSwish,
    kLiteRtOpCodeTflPrelu,
    kLiteRtOpCodeTflSqrt,
    kLiteRtOpCodeTflGreaterEqual,
    kLiteRtOpCodeTflLessEqual,
    kLiteRtOpCodeTflLogicalAnd,
    kLiteRtOpCodeTflLogicalNot,
    kLiteRtOpCodeTflL2Normalization,
    kLiteRtOpCodeTflGreater,
    kLiteRtOpCodeTflRelu0To1,
    kLiteRtOpCodeTflSquare,
};
// clang format on

// When exporting a model via the OpenVINO NPU plugin, standard string streams
// might encounter a 32-bit std::streamsize limitation on specific platforms,
// which restricts model export capacity. This custom output stream buffer
// bypasses that limitation, enabling support for larger models.
class CustomOStreamBuf : public std::streambuf {
 public:
  CustomOStreamBuf() = default;
  std::string drain_str() { return std::move(target_); }

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    target_.append(s, n);
    return n;
  }
  int_type overflow(int_type ch) override {
    if (ch != traits_type::eof()) {
      target_.push_back(static_cast<char>(ch));
      return ch;
    }
    return traits_type::eof();
  }

 private:
  std::string target_;
};
}  // namespace

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = litert::openvino::GetNumSocModels();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr ||
      soc_model_idx >= litert::openvino::GetNumSocModels() ||
      soc_model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = litert::openvino::GetSocModelName(soc_model_idx);
  return kLiteRtStatusOk;
}

// Compiled Result Definition
/// \brief Define storage of compiled result object for OV compiler plugin
struct LiteRtCompiledResultT {
  std::vector<std::string> byte_code;
  std::vector<std::string> graph_names;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  const char* raw_data_ptr = compiled_result->byte_code[byte_code_idx].data();
  *byte_code = static_cast<void*>(const_cast<char*>(raw_data_ptr));
  *byte_code_size = compiled_result->byte_code[byte_code_idx].length();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  auto& graph_name = compiled_result->graph_names[call_idx];
  *call_info = graph_name.data();
  *call_info_size = graph_name.size();
  *byte_code_idx = call_idx;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

// Plugin Definition
/// \brief Define Compiler plugin APIs
struct LiteRtCompilerPluginT {
  using IntelOpenVinoOptions = ::litert::intel_openvino::IntelOpenVinoOptions;

  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env, LiteRtOptions options) {
    if (options == nullptr) return;
    auto cc_options = litert::Options(options, litert::OwnHandle::kNo);
    auto opaques_status = cc_options.GetOpaqueOptions();
    if (!opaques_status) return;

    auto target_opq_status = litert::FindOpaqueOptions(
        *opaques_status, LrtGetIntelOpenVinoOptionsIdentifier());
    if (target_opq_status) {
      auto payload_status = target_opq_status->GetData<const char>();
      if (payload_status) {
        LrtIntelOpenVinoOptions raw_options = nullptr;
        if (LrtCreateIntelOpenVinoOptionsFromToml(
                payload_status.Value(), &raw_options) == kLiteRtStatusOk) {
          intel_openvino_opts =
              IntelOpenVinoOptions::CreateFromOwnedHandle(raw_options);
        }
      }
    }
  }

  const ::litert::Expected<IntelOpenVinoOptions>& GetIntelOpenVinoOptions()
      const {
    return intel_openvino_opts;
  }

  const ::litert::Expected<litert::OpaqueOptions>& GetOpaqueOptions() const {
    return opq;
  }

 private:
  litert::Expected<litert::Options> compiler_opts =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::OpaqueOptions> opq =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
  litert::Expected<IntelOpenVinoOptions> intel_openvino_opts = litert::Error(
      kLiteRtStatusErrorInvalidArgument, "Null Intel OpenVINO options");
};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_INFO);
  auto* plugin = new LiteRtCompilerPluginT(env, options);
  *compiler_plugin = plugin;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

bool IsOpSupported(const ::litert::Op& op) {
  for (const auto& supportedOp : kSupportedOps) {
    if (op.Code() == supportedOp) return true;
  }
  return false;
}

#ifdef __cplusplus
extern "C" {
#endif
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);

  // Check if any subgraph input has dims.size() >= 6.
  auto subgraph_inputs = graph.Inputs();
  for (size_t i = 0; i < subgraph_inputs.size(); ++i) {
    auto ranked_type = subgraph_inputs[i].RankedTensorType();
    if (ranked_type.HasValue()) {
      auto dims = ranked_type.Value().Layout().Dimensions();
      if (dims.size() >= 6) {
        LITERT_LOG(LITERT_WARNING,
                   "Model not supported: subgraph input %zu has %zu dimensions "
                   "(>= 6), skipping partitioning.",
                   i, dims.size());
        return kLiteRtStatusErrorUnsupported;
      }
    }
  }

  // TODO(rjasuja): Enhance implementation for Partition() call
  for (const auto& op : graph.Ops()) {
    if (!IsOpSupported(op)) {
      LITERT_LOG(LITERT_INFO, "op type %d is not supported", op.Code());
      continue;
    }
    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }

  return kLiteRtStatusOk;
}
#ifdef __cplusplus
} /* end extern "C" */
#endif

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  try {
    auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
    const auto num_partitions = model.NumSubgraphs();

    // Configure device and OpenVINO settings from Intel OpenVINO options

    std::string device = "NPU";  // Default device
    ov::AnyMap configs_map;

    if (compiler_plugin->GetIntelOpenVinoOptions().HasValue()) {
      const auto& intel_opts =
          compiler_plugin->GetIntelOpenVinoOptions().Value();

      // Configure device type
      auto device_type = intel_opts.GetDeviceType();
      switch (device_type) {
        case kLiteRtIntelOpenVinoDeviceTypeCPU:
          device = "CPU";
          break;
        case kLiteRtIntelOpenVinoDeviceTypeGPU:
          device = "GPU";
          break;
        case kLiteRtIntelOpenVinoDeviceTypeNPU:
          device = "NPU";
          break;
        case kLiteRtIntelOpenVinoDeviceTypeAUTO:
          device = "AUTO";
          break;
      }

      LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s",
                 device.c_str());

      auto performance_mode = intel_opts.GetPerformanceMode();

      // Add custom configuration options
      int num_custom_options = intel_opts.GetNumConfigsMapOptions();
      for (int i = 0; i < num_custom_options; ++i) {
        auto [key, value] = intel_opts.GetConfigsMapOption(i);
        if (!key.empty()) {  // Valid config option
          configs_map[key] = value;
          LITERT_LOG(LITERT_INFO, "Custom config: %s = %s", key.c_str(),
                     value.c_str());
        }
      }

      // Configure performance mode (can be overridden by custom options)
      switch (performance_mode) {
        case kLiteRtIntelOpenVinoPerformanceModeLatency:
          if (configs_map.find(ov::hint::performance_mode.name()) ==
              configs_map.end()) {
            configs_map[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::LATENCY;
            LITERT_LOG(LITERT_INFO, "Performance mode: LATENCY");
          }
          break;
        case kLiteRtIntelOpenVinoPerformanceModeThroughput:
          if (configs_map.find(ov::hint::performance_mode.name()) ==
              configs_map.end()) {
            configs_map[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::THROUGHPUT;
            LITERT_LOG(LITERT_INFO, "Performance mode: THROUGHPUT");
          }
          break;
        case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
          if (configs_map.find(ov::hint::performance_mode.name()) ==
              configs_map.end()) {
            configs_map[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
            LITERT_LOG(LITERT_INFO, "Performance mode: CUMULATIVE_THROUGHPUT");
          }
          break;
      }
    } else {
      // Default configuration if no options provided
      configs_map[ov::hint::performance_mode.name()] =
          ov::hint::PerformanceMode::LATENCY;
      LITERT_LOG(LITERT_INFO, "Using default configuration (LATENCY mode)");
    }

    LITERT_RETURN_IF_ERROR(
        litert::openvino::ConfigureCompilationParams(soc_model, configs_map));

    auto result = std::make_unique<LiteRtCompiledResultT>();
    result->byte_code.resize(num_partitions);
    result->graph_names.resize(num_partitions);
    auto tflite_fe =
        std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();

    ov::Core core;
    for (int partition_idx = 0; partition_idx < num_partitions;
         ++partition_idx) {
      auto graph_name = absl::StrFormat("Partition_%d", partition_idx);
      litert::Expected<litert::Subgraph> expected_subgraph =
          model.Subgraph(partition_idx);
      if (expected_subgraph.HasValue()) {
        std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator>
            graph_delegate =
                std::make_shared<litert::openvino::GraphIteratorDelegate>(
                    &expected_subgraph.Value());
        auto input_model = tflite_fe->load(graph_delegate);
        LITERT_LOG(LITERT_INFO, "Model loaded");
        auto ov_model = tflite_fe->convert(input_model);

        // Use device and configs_map from Intel OpenVINO options
        auto compiled_model = core.compile_model(ov_model, device, configs_map);

        CustomOStreamBuf obuf;
        std::ostream oss(&obuf);
        compiled_model.export_model(oss);
        LITERT_LOG(LITERT_INFO, "Model export done");
        result->byte_code[partition_idx] = obuf.drain_str();

        result->graph_names.emplace_back(graph_name);
      } else {
        LITERT_LOG(LITERT_INFO, "Failed to retrieve Subgraph");
        return kLiteRtStatusErrorCompilation;
      }
    }
    *compiled_result = result.release();
    // TODO: Add support for caching
    return kLiteRtStatusOk;
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_ERROR, "Exception in compilation: %s", e.what());
    return kLiteRtStatusErrorCompilation;
  }
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  return kLiteRtStatusOk;
}
