// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include <openvino/frontend/tensorflow_lite/frontend.hpp>
#include <openvino/openvino.hpp>
#include "absl/strings/string_view.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"

#include "graph_iterator.h"

namespace {

constexpr char kPluginManufacturer[] = "IntelOpenVINO";

constexpr const char *kPluginSocModels[] = {
    "NPU2700",
};  // get the name for plugin soc model

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflAdd,
    kLiteRtOpCodeTflConv2d,
};
// clang format on

constexpr auto kNumPluginSocModels = sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

}  // namespace

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion *api_version) {
    if (api_version == nullptr) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    api_version->major = LITERT_API_VERSION_MAJOR;
    api_version->minor = LITERT_API_VERSION_MINOR;
    api_version->patch = LITERT_API_VERSION_PATCH;
    return kLiteRtStatusOk;
}

const char *LiteRtGetCompilerPluginSocManufacturer() { return kPluginManufacturer; }

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(LiteRtCompilerPlugin compiler_plugin,
                                                      LiteRtHwAccelerators *supported_hardware) {
    if (!compiler_plugin || !supported_hardware) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    *supported_hardware = kLiteRtHwAcceleratorNpu;
    return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex *num_supported_soc_models) {
    if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    *num_supported_soc_models = kNumPluginSocModels;
    return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(LiteRtCompilerPlugin compiler_plugin,
                                                      LiteRtParamIndex soc_model_idx,
                                                      const char **soc_model_name) {
    if (compiler_plugin == nullptr || soc_model_idx >= kNumPluginSocModels ||
        soc_model_name == nullptr) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    *soc_model_name = kPluginSocModels[soc_model_idx];
    return kLiteRtStatusOk;
}

// Compiled Result Definition
/// \brief Define storage of compiled result object for OV compiler plugin
struct LiteRtCompiledResultT {
    std::vector<std::stringstream> byte_code;
    // TODO: Revisit this to check if we need this or subgraph data
    std::vector<std::string> graph_names;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(LiteRtCompiledResult compiled_result,
                                             LiteRtParamIndex byte_code_idx, const void **byte_code,
                                             size_t *byte_code_size) {
    *byte_code = compiled_result->byte_code.data();
    *byte_code_size = compiled_result->byte_code.size();
    return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(LiteRtCompiledResult compiled_result,
                                             LiteRtParamIndex call_idx, const void **call_info,
                                             size_t *call_info_size,
                                             LiteRtParamIndex *byte_code_idx) {
    if (call_idx >= compiled_result->graph_names.size()) {
        return kLiteRtStatusErrorIndexOOB;
    }

    auto &graph_name = compiled_result->graph_names[call_idx];
    *call_info = graph_name.data();
    *call_info_size = graph_name.size();

    return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(LiteRtCompiledResult compiled_result,
                                             LiteRtParamIndex *num_calls) {
    *num_calls = compiled_result->graph_names.size();
    return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) { delete compiled_result; }

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(LiteRtCompiledResult compiled_result,
                                                    LiteRtParamIndex *num_byte_code) {
    if (!compiled_result || !num_byte_code) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    *num_byte_code = compiled_result->byte_code.size();
    return kLiteRtStatusOk;
}

// Plugin Definition
/// \brief Define Compiler plugin APIs
struct LiteRtCompilerPluginT {};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin *compiler_plugin) {
    *compiler_plugin = new LiteRtCompilerPluginT;
    return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) { delete compiler_plugin; }

bool IsOpSupported(const ::litert::Op &op) {
    for (const auto &supportedOp : kSupportedOps) {
        if (op.Code() == supportedOp) return true;
    }
    return false;
}

#ifdef __cplusplus
extern "C" {
#endif
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char *soc_model, LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
    ::litert::Subgraph graph(subgraph);

    //TODO(rjasuja): Enhance implementation for Partition() call 
    for (const auto &op : graph.Ops()) {
        if (!IsOpSupported(op)) {
            LITERT_LOG(LITERT_ERROR, "op type %d is not supported", op.Code());
            continue;
        }
        LITERT_LOG(LITERT_INFO, "op type %d is supported", op.Code());
        LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    }

    return kLiteRtStatusOk;
}
#ifdef __cplusplus
} /* end extern "C" */
#endif

LiteRtStatus LiteRtCompilerPluginCompile(LiteRtCompilerPlugin compiler_plugin,
                                         const char *soc_model, LiteRtModel partitions,
                                         LiteRtCompiledResult *compiled_result) {
    auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
    const auto num_partitions = model.NumSubgraphs();

    auto result = std::make_unique<LiteRtCompiledResultT>();
    result->byte_code.resize(num_partitions);
    result->graph_names.resize(num_partitions);
    auto tflite_fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
    // TODO: Update this hard coded path to an env option passed from LiteRT framework
    ov::Core core;
    for (int partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
        std::string subgraph_name = "Partition_" + partition_idx;
        litert::Expected<litert::Subgraph> expected_subgraph = model.Subgraph(partition_idx);
        if (expected_subgraph.HasValue()) {
            std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> graph_delegate =
                std::make_shared<litert::openvino::GraphIteratorDelegate>(
                    &expected_subgraph.Value());
            auto input_model = tflite_fe->load(graph_delegate);
            auto model = tflite_fe->convert(input_model);

            // TODO: pass the device string from env options
            std::string device = "NPU";
            auto compiled_model = core.compile_model(model, device);

            compiled_model.export_model(result->byte_code[partition_idx]);
            result->graph_names[partition_idx].append(subgraph_name.c_str());
        } else {
            LITERT_LOG(LITERT_INFO, "Failed to retrieve Subgraph");
            return kLiteRtStatusErrorCompilation;
        }
    }
    *compiled_result = result.release();
    // TODO: Add support for caching
    return kLiteRtStatusOk;
}
