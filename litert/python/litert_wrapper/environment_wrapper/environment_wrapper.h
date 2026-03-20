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

#ifndef LITERT_PYTHON_LITERT_WRAPPER_ENVIRONMENT_WRAPPER_ENVIRONMENT_WRAPPER_H_
#define LITERT_PYTHON_LITERT_WRAPPER_ENVIRONMENT_WRAPPER_ENVIRONMENT_WRAPPER_H_

#include <Python.h>

#include <string>

namespace litert::environment_wrapper {

// Creates a Python capsule that owns a LiteRT Environment.
PyObject* CreateEnvironment(const char* runtime_path,
                            const char* compiler_plugin_path,
                            const char* dispatch_library_path);

// Reports an error by setting a Python exception.
PyObject* ReportError(const std::string& msg);

}  // namespace litert::environment_wrapper

#endif  // LITERT_PYTHON_LITERT_WRAPPER_ENVIRONMENT_WRAPPER_ENVIRONMENT_WRAPPER_H_
