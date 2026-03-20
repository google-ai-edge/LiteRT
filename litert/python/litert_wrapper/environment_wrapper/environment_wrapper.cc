// Copyright 2026 Google LLC.
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

#include "litert/python/litert_wrapper/environment_wrapper/environment_wrapper.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"

namespace litert::environment_wrapper {

PyObject* ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

PyObject* CreateEnvironment(const char* runtime_path,
                            const char* compiler_plugin_path,
                            const char* dispatch_library_path) {
  std::vector<litert::EnvironmentOptions::Option> options;
  std::string runtime_path_str = runtime_path != nullptr ? runtime_path : "";
  if (!runtime_path_str.empty()) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, runtime_path_str});
  }
  if (compiler_plugin_path != nullptr && *compiler_plugin_path) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
        std::string(compiler_plugin_path)});
  }
  if (dispatch_library_path != nullptr && *dispatch_library_path) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        std::string(dispatch_library_path)});
  }

  auto env_or = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(options)));
  if (!env_or) {
    return ReportError(env_or.Error().Message());
  }

  auto* env = new litert::Environment(std::move(*env_or));
  PyObject* capsule = litert_wrapper_utils::MakeEnvironmentCapsule(env);
  if (capsule == nullptr) {
    delete env;
    return ReportError("Failed to create environment capsule");
  }
  return capsule;
}

}  // namespace litert::environment_wrapper
