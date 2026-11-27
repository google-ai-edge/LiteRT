// Copyright 2026 Amlogic, Inc.
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

#include "aml_compiler_core_loader.h"

#include <dlfcn.h>

#include <mutex>

#include "litert/c/internal/litert_logging.h"

namespace aml_compiler_core {
namespace {

// dlsym every required symbol; logs and reports the first missing one.
bool ResolveRequiredSymbols(void* handle, FuncTable* funcs) {
  struct Sym {
    const char* name;
    void** slot;
  };
  const Sym required[] = {
      {"AmlContextCreate", reinterpret_cast<void**>(&funcs->AmlContextCreate)},
      {"AmlContextDestroy",
       reinterpret_cast<void**>(&funcs->AmlContextDestroy)},
      {"AmlGraphCreate", reinterpret_cast<void**>(&funcs->AmlGraphCreate)},
      {"AmlGraphSetTarget",
       reinterpret_cast<void**>(&funcs->AmlGraphSetTarget)},
      {"AmlGraphSetName", reinterpret_cast<void**>(&funcs->AmlGraphSetName)},
      {"AmlGraphAddInput",
       reinterpret_cast<void**>(&funcs->AmlGraphAddInput)},
      {"AmlGraphAddOutput",
       reinterpret_cast<void**>(&funcs->AmlGraphAddOutput)},
      {"AmlTensorCreate", reinterpret_cast<void**>(&funcs->AmlTensorCreate)},
      {"AmlTensorSetName", reinterpret_cast<void**>(&funcs->AmlTensorSetName)},
      {"AmlTensorSetDtype",
       reinterpret_cast<void**>(&funcs->AmlTensorSetDtype)},
      {"AmlTensorSetIndex",
       reinterpret_cast<void**>(&funcs->AmlTensorSetIndex)},
      {"AmlTensorSetDims", reinterpret_cast<void**>(&funcs->AmlTensorSetDims)},
      {"AmlTensorSetQuant",
       reinterpret_cast<void**>(&funcs->AmlTensorSetQuant)},
      {"AmlTensorSetConstData",
       reinterpret_cast<void**>(&funcs->AmlTensorSetConstData)},
      {"AmlNodeCreate", reinterpret_cast<void**>(&funcs->AmlNodeCreate)},
      {"AmlNodeAddInput", reinterpret_cast<void**>(&funcs->AmlNodeAddInput)},
      {"AmlNodeAddOutput",
       reinterpret_cast<void**>(&funcs->AmlNodeAddOutput)},
      {"AmlNodeSetParamI", reinterpret_cast<void**>(&funcs->AmlNodeSetParamI)},
      {"AmlNodeSetParamF", reinterpret_cast<void**>(&funcs->AmlNodeSetParamF)},
      {"AmlCompilerCoreOptionsInitDefaults",
       reinterpret_cast<void**>(&funcs->AmlCompilerCoreOptionsInitDefaults)},
      {"AmlCompilerExternalOptionsInitDefaults",
       reinterpret_cast<void**>(
           &funcs->AmlCompilerExternalOptionsInitDefaults)},
      {"AmlContextCompile", reinterpret_cast<void**>(&funcs->AmlContextCompile)},
      {"AmlCompilerCoreDestroyCompiledArtifact",
       reinterpret_cast<void**>(
           &funcs->AmlCompilerCoreDestroyCompiledArtifact)},
      {"AmlCompilerCoreGetCompiledArtifactAdlaBinary",
       reinterpret_cast<void**>(
           &funcs->AmlCompilerCoreGetCompiledArtifactAdlaBinary)},
      {"AmlCompilerCoreGetCompiledArtifactAdlaPath",
       reinterpret_cast<void**>(
           &funcs->AmlCompilerCoreGetCompiledArtifactAdlaPath)},
      {"AmlCompilerCoreGetCompiledArtifactFingerprint",
       reinterpret_cast<void**>(
           &funcs->AmlCompilerCoreGetCompiledArtifactFingerprint)},
      {"AmlCompilerCoreGetCompiledArtifactCacheHit",
       reinterpret_cast<void**>(
           &funcs->AmlCompilerCoreGetCompiledArtifactCacheHit)},
  };
  for (const Sym& sym : required) {
    dlerror();
    *sym.slot = dlsym(handle, sym.name);
    if (*sym.slot == nullptr) {
      const char* err = dlerror();
      LITERT_LOG(LITERT_ERROR,
                 "libaml_compiler_core.so missing required symbol %s (%s)",
                 sym.name, err != nullptr ? err : "unknown");
      return false;
    }
  }
  // Optional: SDK version query. LiteRtGetCompilerPluginSDKVersion falls
  // back to a static string when it is absent.
  dlerror();
  funcs->AmlCompilerCoreGetVersion =
      reinterpret_cast<decltype(funcs->AmlCompilerCoreGetVersion)>(
          dlsym(handle, "AmlCompilerCoreGetVersion"));
  return true;
}

}  // namespace

const FuncTable* Funcs(bool allow_dlopen) {
  static FuncTable funcs;
  static std::mutex mu;
  static bool ready = false;
  static bool load_failed = false;
  std::lock_guard<std::mutex> lock(mu);
  if (ready) {
    return &funcs;
  }
  if (load_failed) {
    return nullptr;
  }
  dlerror();
  void* handle =
      dlopen(kSoName, RTLD_LAZY | RTLD_LOCAL | (allow_dlopen ? 0 : RTLD_NOLOAD));
  if (handle == nullptr) {
    if (allow_dlopen) {
      const char* err = dlerror();
      LITERT_LOG(LITERT_ERROR, "open %s failed err=%s", kSoName,
                 err != nullptr ? err : "unknown");
      load_failed = true;
    }
    return nullptr;
  }
  if (!ResolveRequiredSymbols(handle, &funcs)) {
    // Keep the .so mapped (libcompiler_adla.so is not safe to unmap); only
    // the function table stays empty.
    load_failed = true;
    return nullptr;
  }
  ready = true;
  LITERT_LOG(LITERT_INFO, "libaml_compiler_core.so loaded via dlopen");
  return &funcs;
}

}  // namespace aml_compiler_core
