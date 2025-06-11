// Copyright 2024 Google LLC.
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

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/options_helper.h"
#include "litert/vendors/mediatek/compiler/compile_model.h"
#include "litert/vendors/mediatek/compiler/create_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/common_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"
#include "litert/vendors/mediatek/schema/neuron_schema_generated.h"
#include "litert/vendors/mediatek/schema/schema_resolver.h"

//
// Configurations
//

using litert::Error;
using litert::Expected;
using litert::mediatek::NeuronAdapterApi;
using litert::mediatek::NeuronCompilationPtr;
using litert::mediatek::NeuronModelPtr;
using litert::mediatek::OperandMap;

namespace {

constexpr char kPluginManufacturer[] = "MediaTek";

// clang-format off
constexpr std::pair<const char*, const char*> kPluginSocModels[] = {
    {"mt6853", "mt6853"},
    {"mt6877", "mt6877"},
    {"mt6878", "mt6878"},
    {"mt6879", "mt6879"},
    {"mt6886", "mt6886"},
    {"mt6893", "mt6893"},
    {"mt6895", "mt6895"},
    {"mt6897", "mt6897"},
    {"mt6983", "mt6983"},
    {"mt6985", "mt6985"},
    {"mt6989", "mt6989"},
    {"mt6991", "mt6991"},
};

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflAdd,
    kLiteRtOpCodeTflMul,
    kLiteRtOpCodeTflBatchMatmul,
    kLiteRtOpCodeTflFullyConnected,
    kLiteRtOpCodeTflReshape,
    kLiteRtOpCodeTflTranspose,
    kLiteRtOpCodeTflRsqrt,
    kLiteRtOpCodeTflConcatenation,
    kLiteRtOpCodeTflQuantize,
    kLiteRtOpCodeTflSlice,
    kLiteRtOpCodeTflSub,
    kLiteRtOpCodeTflTanh,
    kLiteRtOpCodeTflSoftmax,
    kLiteRtOpCodeTflMean,
    kLiteRtOpCodeTflGelu,
    kLiteRtOpCodeTflPad,
    kLiteRtOpCodeTflLogistic,
    kLiteRtOpCodeTflSum,
    kLiteRtOpCodeTflConv2d,
    kLiteRtOpCodeTflDepthwiseConv2d,
    kLiteRtOpCodeTflSquaredDifference,
    kLiteRtOpCodeTflResizeBilinear,
    kLiteRtOpCodeTflResizeNearestNeighbor,
    kLiteRtOpCodeTflTransposeConv,
    kLiteRtOpCodeTflMaxPool2d,
    kLiteRtOpCodeTflDequantize,
    kLiteRtOpCodeTflPadv2,
    kLiteRtOpCodeTflHardSwish,
    kLiteRtOpCodeTflAveragePool2d
};
// clang-format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

std::optional<const char*> FindSocModel(absl::string_view soc_model_name) {
  std::optional<const char*> soc_model;
  for (auto i = 0; i < kNumPluginSocModels; ++i) {
    if (soc_model_name == kPluginSocModels[i].first) {
      soc_model = kPluginSocModels[i].second;
      break;
    }
  }
  return soc_model;
}

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
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = kNumPluginSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (soc_model_idx < 0 || soc_model_idx >= kNumPluginSocModels) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModels[soc_model_idx].first;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// TODO: Revisit this struct after we extend the compiler plugin API to return
// results with more than one single bytecode.
struct LiteRtCompiledResultT {
  std::vector<std::string> graph_names;
  neuron::BytecodeBuilder bytebuilder;
};

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // TODO(yunandrew) MTK should have one byte code per call. But now only one
  // bytecode is created for all partitions.
  *num_byte_code = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size ||
      (byte_code_idx >= compiled_result->graph_names.size())) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *byte_code = compiled_result->bytebuilder.GetBytecode().first;
  *byte_code_size = compiled_result->bytebuilder.GetBytecode().second;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  auto& graph_name = compiled_result->graph_names[call_idx];
  *call_info = graph_name.data();
  *call_info_size = graph_name.size();
  // TODO: MTK should have one byte code per call.
  // Only one bytecode is created for all partitions, so the byte code index is
  // always 0.
  *byte_code_idx = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
class LiteRtCompilerPluginT {
 public:
  using MediatekOptions = ::litert::mediatek::MediatekOptions;

  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env, LiteRtOptions options) {
    std::tie(env_, opts_, opq_, mediatek_opts_) =
        litert::ParseOptions<MediatekOptions>(env, options);
  }

  ::litert::Expected<MediatekOptions>& GetMediatekOptions() {
    return mediatek_opts_;
  }

  ::litert::Expected<litert::OpaqueOptions>& GetOpaqueOptions() { return opq_; }

 private:
  litert::Expected<litert::EnvironmentOptions> env_ = litert::Error(
      kLiteRtStatusErrorInvalidArgument, "Null environment options");
  litert::Expected<litert::Options> opts_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::OpaqueOptions> opq_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
  litert::Expected<litert::mediatek::MediatekOptions> mediatek_opts_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument,
                    "Null google tensor options");
};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  *compiler_plugin = new LiteRtCompilerPluginT(env, options);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

namespace {

// TODO update this function to match the new legalizations.
bool IsOpSupported(const litert::Op& op) {
  // NOTE: Currently we are demoing by just mapping simple f32 mul ops.  Use a
  // very loose guard for now -- only checking if op code is supported.
  for (auto supported_op : kSupportedOps) {
    if (op.Code() == supported_op &&
        litert::mediatek::VerifyCommonOp(op, op.Code())) {
      return true;
    }
  }
  return false;
}

}  // namespace

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }

  return kLiteRtStatusOk;
}

namespace {

Expected<std::vector<uint8_t>> CompilePartition(
    NeuronAdapterApi& neuron_adapter_api, const litert::Subgraph& partition,
    const std::string& graph_name, std::optional<std::string> soc_model,
    ::litert::Expected<litert::mediatek::MediatekOptions>& mediatek_opts) {
  auto model = neuron_adapter_api.CreateModel();
  if (!model) {
    return model.Error();
  }
  OperandMap operand_map(neuron_adapter_api, model->get());
  LITERT_RETURN_IF_ERROR(CreateModel(neuron_adapter_api, partition, graph_name,
                                     model->get(), &operand_map));

  auto compilation =
      CompileModel(neuron_adapter_api, model->get(), soc_model, mediatek_opts);
  if (!compilation) {
    return compilation.Error();
  }

  size_t bytecode_size;
  if (neuron_adapter_api.api().compilation_get_compiled_network_size(
          compilation->get(), &bytecode_size) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get compiled network size");
  }

  std::vector<uint8_t> bytecode(bytecode_size);
  if (neuron_adapter_api.api().compilation_store_compiled_network(
          compilation->get(), bytecode.data(), bytecode.size()) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get compiled network");
  }

  return bytecode;
}

}  // namespace

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
#if __ANDROID__
  char dla_directory_template[] =
      "/data/local/tmp/runfiles/tempdir_dla.XXXXXXX";
#else
  char dla_directory_template[] = "/tmp/tempdir_dla.XXXXXXX";
#endif

  char* dla_directory_name = mkdtemp(dla_directory_template);
  if (dla_directory_name == nullptr) {
    int error_code = errno;
    LITERT_LOG(LITERT_ERROR,
               "Failed to make DLA temporary directory, (errno=%d)",
               error_code);
    return kLiteRtStatusErrorFileIO;
  }

  // A null soc_model is passed when performing JIT compilation.
  if (soc_model) {
    setenv("MTKNN_ADAPTER_DLA_PLATFORM", soc_model, 1);
  }
  setenv("MTKNN_ADAPTER_DLA_DIR", dla_directory_name, 1);

  auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();

  LITERT_LOG(LITERT_INFO,
             "Starting MediaTek Compilation for %d subgraphs, soc_model=%s",
             num_partitions, soc_model);

  auto opt_soc_model = soc_model ? FindSocModel(soc_model) : std::nullopt;
  if (opt_soc_model) {
    LITERT_LOG(LITERT_ERROR, "Compiling for MediaTek architecture: %s",
               *opt_soc_model);
  } else if (soc_model) {
    LITERT_LOG(LITERT_ERROR, "Unexpected SoC model: %s", soc_model);
    rmdir(dla_directory_name);
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (!compiler_plugin->GetMediatekOptions()) {
    LITERT_ASSIGN_OR_RETURN(compiler_plugin->GetMediatekOptions(),
                            ::litert::mediatek::MediatekOptions::Create());
  }

  auto api = NeuronAdapterApi::Create(/*shared_library_dir=*/std::nullopt,
                                      compiler_plugin->GetMediatekOptions());
  if (!api) {
    rmdir(dla_directory_name);
    return api.Error().Status();
  }

  auto result = std::make_unique<LiteRtCompiledResultT>();

  for (auto i = 0; i < num_partitions; ++i) {
    auto graph_name = absl::StrFormat("Partition_%d", i);
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(i));
    auto bytecode = CompilePartition(**api, subgraph, graph_name, opt_soc_model,
                                     compiler_plugin->GetMediatekOptions());
    rmdir(dla_directory_name);
    if (!bytecode) {
      LITERT_LOG(LITERT_INFO, "%s", bytecode.Error().Message().c_str());
      return bytecode.Error().Status();
    }
    auto bufferIdx = result->bytebuilder.AddBuffer(
        graph_name, (int8_t*)bytecode->data(), bytecode->size());
    result->bytebuilder.AddCompiledNetwork(
        graph_name, NeuronSchema::CompiledType_AdapterCache, bufferIdx);
    result->graph_names.emplace_back(graph_name);
  }

  if (!result->bytebuilder.Finish()) {
    return kLiteRtStatusErrorCompilation;
  }
  *compiled_result = result.release();
  return kLiteRtStatusOk;
}
