#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>

#include "absl/strings/str_format.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"

namespace {

constexpr char kPluginManufacturer[] = "Hailo";

const std::vector<std::string>& GetSupportedSocModels() {
  static const std::vector<std::string>* const kSupportedSocModels =
      new std::vector<std::string>{"Hailo-8", "Hailo-8L", "Hailo-10", "Hailo-15"};
  return *kSupportedSocModels;
}

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  out->resize(size);
  if (!file.read(&out->front(), size)) {
    return false;
  }
  return true;
}

}  // namespace

struct LiteRtCompilerPluginT {
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx) : ctx_(ctx) {}
  const LiteRtCompilerContext* ctx() const { return ctx_; }

 private:
  const LiteRtCompilerContext* ctx_;
};

struct LiteRtCompiledResultT {
  std::vector<std::string> byte_code;
  std::vector<std::string> graph_names;
};

//
// Plugin API Implementation
//

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

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (compiler_context == nullptr || compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (compiler_plugin == nullptr || supported_hardware == nullptr) {
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
  *num_supported_soc_models = GetSupportedSocModels().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr || soc_model_name == nullptr ||
      soc_model_idx >= GetSupportedSocModels().size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = GetSupportedSocModels()[soc_model_idx].c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (compiler_plugin == nullptr || sdk_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version = "latest";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  if (compiler_plugin == nullptr || subgraph == nullptr || selected_ops == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::compiler::Subgraph graph(compiler_plugin->ctx(), subgraph);

  // Group all operations in this subgraph into a single partition for wrapping.
  for (const auto& op : graph.Ops()) {
    LITERT_RETURN_IF_ERROR(compiler_plugin->ctx()->push_op(selected_ops, op.Get(), 0));
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (compiler_plugin == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Retrieve the pre-compiled HEF file path from the environment variable.
  const char* hef_path_env = std::getenv("LITERT_HAILO_HEF_PATH");
  if (hef_path_env == nullptr || *hef_path_env == '\0') {
    LITERT_LOG(LITERT_ERROR, "Environment variable LITERT_HAILO_HEF_PATH is not set.");
    return kLiteRtStatusErrorCompilation;
  }

  std::string hef_data;
  if (!ReadFile(hef_path_env, &hef_data)) {
    LITERT_LOG(LITERT_ERROR, "Failed to read pre-compiled HEF file from: %s", hef_path_env);
    return kLiteRtStatusErrorCompilation;
  }

  litert::compiler::Model model(compiler_plugin->ctx(), partitions);
  const auto num_partitions = model.NumSubgraphs();

  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->byte_code.resize(num_partitions);
  result->graph_names.resize(num_partitions);

  for (int i = 0; i < num_partitions; ++i) {
    // Wrap the pre-compiled HEF bytecode directly into each partition.
    result->byte_code[i] = hef_data;
    result->graph_names[i] = absl::StrFormat("Partition_%d", i);
  }

  *compiled_result = result.release();
  LITERT_LOG(LITERT_INFO, "Successfully wrapped pre-compiled HEF of size %zu bytes.", hef_data.size());

  return kLiteRtStatusOk;
}

//
// Compiled Result API Implementation
//

void LiteRtDestroyCompiledResult(LiteRtCompiledResult result) {
  delete result;
}

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (compiled_result == nullptr || byte_code == nullptr || byte_code_size == nullptr ||
      byte_code_idx >= compiled_result->byte_code.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *byte_code = compiled_result->byte_code[byte_code_idx].data();
  *byte_code_size = compiled_result->byte_code[byte_code_idx].size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (compiled_result == nullptr || num_byte_code == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (compiled_result == nullptr || call_info == nullptr || call_info_size == nullptr ||
      byte_code_idx == nullptr || call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *call_info = compiled_result->graph_names[call_idx].data();
  *call_info_size = compiled_result->graph_names[call_idx].size();
  *byte_code_idx = call_idx;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (compiled_result == nullptr || num_calls == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (num_patterns == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  return kLiteRtStatusOk;
}
