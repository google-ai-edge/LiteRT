// Copyright (C) 2026 Samsung Electronics Co. LTD.
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
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_AI_LITECORE_MANAGER_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_AI_LITECORE_MANAGER_H_

#include "graph_wrapper_api.h"
#include <optional>
namespace {
// Avoid enum name warning
#include "graphgen_c.h"
} // namespace
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include <functional>

namespace litert::samsung {

typedef void *backend_handler_t;
typedef GraphHandler graph_handler_t;

using GraphWrapperPtr = std::unique_ptr<
    std::remove_pointer<graph_handler_t>::type,
    std::function<void(std::remove_pointer<graph_handler_t>::type *)>>;
using BackendPtr = std::unique_ptr<
    std::remove_pointer<backend_handler_t>::type,
    std::function<void(std::remove_pointer<backend_handler_t>::type *)>>;

class AiLiteCoreManager {
public:
  using UniquePtr = std::unique_ptr<AiLiteCoreManager>;
  using Ptr = UniquePtr::pointer;
  struct PublicApi;

  AiLiteCoreManager(AiLiteCoreManager &) = delete;
  AiLiteCoreManager(AiLiteCoreManager &&) = delete;
  AiLiteCoreManager &operator=(const AiLiteCoreManager &) = delete;
  AiLiteCoreManager &operator=(AiLiteCoreManager &&) = delete;

  static Expected<UniquePtr>
  Create(std::optional<std::string> shared_library_dir);

  const PublicApi &Api() const { return *api_; }

  Expected<GraphWrapperPtr> CreateGraphHandler() const;

  Expected<BackendPtr> CreateBackendHandler() const;

private:
  AiLiteCoreManager();
  // Loads and resolve compiler related api
  LiteRtStatus LoadBackendLibrary(absl::string_view path);

  // Loads and resolve api for creating unified graph
  LiteRtStatus LoadGraphWrapperLibrary(absl::string_view path);
  // Handle to the npu compiler shared library
  SharedLibrary backend_lib_;

  // Handle to the graph wrapper shared library, for building unified-format
  // model
  SharedLibrary graph_wrapper_lib_;
  std::unique_ptr<PublicApi> api_;
};

struct AiLiteCoreManager::PublicApi {
  // graph
  decltype(&create_graph) CreateGraph = nullptr;
  decltype(&set_graph_input_tensors) SetGraphInputs = nullptr;
  decltype(&set_graph_output_tensors) SetGraphOutputs = nullptr;
  decltype(&finish_build_graph) FinishGraphBuild = nullptr;
  decltype(&serialize) Serialize = nullptr;
  decltype(&define_op_node) DefineOp = nullptr;
  decltype(&add_op_parameter) AddOpParam = nullptr;
  decltype(&define_tensor) DefineTensor = nullptr;
  decltype(&set_quantize_param_for_tensor) SetTensorQParam = nullptr;
  decltype(&set_data_for_constant_tensor) SetTensorData = nullptr;
  decltype(&release_graph) ReleaseGraph = nullptr;
  // backend compiler
  decltype(&graphgen_create) CreateBackend = nullptr;
  decltype(&graphgen_release) ReleaseBackend = nullptr;
  decltype(&graphgen_initialize_context) InitializeBackendContext = nullptr;
  decltype(&graphgen_generate) BackendCompile = nullptr;
  decltype(&graphgen_release_buffer) ReleaseBuffer = nullptr;
};

} // namespace litert::samsung
#endif // ODML_LITERT_LITERT_VENDORS_SAMSUNG_AI_LITECORE_MANAGER_H_
