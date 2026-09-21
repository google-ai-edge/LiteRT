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

#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/examples/compatibility_test/mock_plugin_common.h"

namespace {

using namespace ::litert::compatibility;  // NOLINT

const char* GetOlderSocManufacturer() { return "OlderSupportedManufacturer"; }

// Mock older V1.0 plugin compiled before get_compiler_plugin_sdk_version
// was added.
static const LiteRtCompilerPluginInterface_V1 kOlderSupportedInterface = {
    .abi_header =
        {
            .struct_size = offsetof(LiteRtCompilerPluginInterface_V1,
                                    get_compiler_plugin_sdk_version),
            .major_version = 1,
            .minor_version = 0,
            .reserved = 0,
        },
    .get_compiler_plugin_version = MockGetCompilerPluginVersion,
    .get_compiler_plugin_soc_manufacturer = GetOlderSocManufacturer,
    .create_compiler_plugin = MockCreateCompilerPlugin,
    .destroy_compiler_plugin = MockDestroyCompilerPlugin,
    .get_compiler_plugin_supported_hardware =
        MockGetCompilerPluginSupportedHardware,
    .get_num_compiler_plugin_supported_models =
        MockGetNumCompilerPluginSupportedSocModels,
    .get_compiler_plugin_supported_soc_model =
        MockGetCompilerPluginSupportedSocModel,
    .compiler_plugin_partition = MockCompilerPluginPartition,
    .compiler_plugin_compile = MockCompilerPluginCompile,
    .destroy_compiled_result = MockDestroyCompiledResult,
    .get_compiled_result_byte_code = MockGetCompiledResultByteCode,
    .get_compiled_result_num_byte_code = MockCompiledResultNumByteCodeModules,
    .get_compiled_result_call_info = MockGetCompiledResultCallInfo,
    .get_num_compiled_result_calls = MockGetNumCompiledResultCalls,
    .register_all_transformations =
        MockCompilerPluginRegisterAllTransformations,
    // Present in memory, but outside struct_size. LITERT_ABI_HAS_API must
    // return false.
    .get_compiler_plugin_sdk_version = MockGetCompilerPluginSDKVersion,
    .get_compiled_result_handle = nullptr,
    .check_compiler_compatibility = nullptr,
};

}  // namespace

extern "C" LITERT_CAPI_EXPORT LiteRtStatus
LiteRtCompilerPluginQueryInterface(LiteRtCompilerPluginInterfaceId interface_id,
                                   LiteRtApiVersion litert_runtime_version,
                                   LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major >= 1) {
    if (interface_id == kLiteRtCompilerPluginInterfaceBasic) {
      *out_interface = const_cast<LiteRtCompilerPluginInterface_V1*>(
          &kOlderSupportedInterface);
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorUnsupported;
}
