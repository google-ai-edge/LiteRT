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
#include <cstdlib>

#include "litert/vendors/samsung/ai_litecore_manager.h"

static const char kEnvAiLiteCoreVar[] = "EXYNOS_AI_LITECORE_ROOT";
static const char kGraphWrapperLibName[] = "graph_wrapper";
static const char kBackendLibName[] = "graphgen_api";

#if defined(__linux__)
static constexpr absl::string_view kLibSuffix = ".so";
static constexpr absl::string_view kRelativePath = "/lib/x86_64-linux";
#else
// TODO: Add other operation system if enabled
#endif

#define LOAD_LIB_SYMBOL(LIB, SYM, M)                                           \
  if (auto symbol = LIB.LookupSymbol<void *>(#SYM); symbol.HasValue()) {       \
    M = reinterpret_cast<decltype(&SYM)>(std::move(symbol.Value()));           \
  } else {                                                                     \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #SYM,           \
               LIB.DlError());                                                 \
    return kLiteRtStatusErrorDynamicLoading;                                   \
  }

namespace litert::samsung {

AiLiteCoreManager::AiLiteCoreManager() : api_(new struct PublicApi) {}

Expected<AiLiteCoreManager::UniquePtr>
AiLiteCoreManager::Create(std::optional<std::string> shared_library_dir) {
  // Take current directory as default option
  std::string graph_wrapper_lib_path =
      absl::StrCat("lib", kGraphWrapperLibName, kLibSuffix);
  std::string backend_lib_path =
      absl::StrCat("lib", kBackendLibName, kLibSuffix);
  if (shared_library_dir.has_value()) {
    graph_wrapper_lib_path =
        absl::StrCat(*shared_library_dir, "/", graph_wrapper_lib_path);
    backend_lib_path = absl::StrCat(*shared_library_dir, "/", backend_lib_path);
  } else if (auto ai_lite_core_path = getenv(kEnvAiLiteCoreVar);
             ai_lite_core_path != NULL) {
    graph_wrapper_lib_path = absl::StrCat(ai_lite_core_path, kRelativePath, "/",
                                          graph_wrapper_lib_path);
    backend_lib_path =
        absl::StrCat(ai_lite_core_path, kRelativePath, "/", backend_lib_path);
  } else {
    LITERT_LOG(LITERT_INFO, "Loading %d  %s",
               (getenv(kEnvAiLiteCoreVar) != NULL), kEnvAiLiteCoreVar);
  }

  AiLiteCoreManager::UniquePtr ai_lite_core_mgr(new AiLiteCoreManager);
  if (auto status =
          ai_lite_core_mgr->LoadGraphWrapperLibrary(graph_wrapper_lib_path);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to load graph wrapper library");
  }
  if (auto status = ai_lite_core_mgr->LoadBackendLibrary(backend_lib_path);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to load backend compiler library");
  }

  return ai_lite_core_mgr;
}

Expected<GraphWrapperPtr> AiLiteCoreManager::CreateGraphHandler() const {
  graph_handler_t graph_handler = Api().CreateGraph("new");
  if (graph_handler == nullptr) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create samsung backend handler");
  }

  return GraphWrapperPtr(graph_handler, [this](graph_handler_t handler) {
    this->Api().ReleaseGraph(handler);
  });
}

Expected<BackendPtr> AiLiteCoreManager::CreateBackendHandler() const {
  backend_handler_t backend_handler = Api().CreateBackend();
  if (backend_handler == nullptr) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create samsung backend handler");
  }

  return BackendPtr(backend_handler, [this](backend_handler_t handler) {
    this->Api().ReleaseBackend(handler);
  });
}

/* private impl */
LiteRtStatus AiLiteCoreManager::LoadBackendLibrary(absl::string_view path) {
  auto loading_lib = SharedLibrary::Load(path, RtldFlags::Default());
  LITERT_LOG(LITERT_INFO, "Loading from: %s", path.data());
  if (!loading_lib.HasValue() || !loading_lib->Loaded()) {
    LITERT_LOG(LITERT_INFO, "Failed to load ");
    return kLiteRtStatusErrorDynamicLoading;
  }
  backend_lib_ = std::move(loading_lib.Value());

  LOAD_LIB_SYMBOL(backend_lib_, graphgen_create, api_->CreateBackend);
  LOAD_LIB_SYMBOL(backend_lib_, graphgen_release, api_->ReleaseBackend);
  LOAD_LIB_SYMBOL(backend_lib_, graphgen_initialize_context,
                  api_->InitializeBackendContext);
  LOAD_LIB_SYMBOL(backend_lib_, graphgen_generate, api_->BackendCompile);
  LOAD_LIB_SYMBOL(backend_lib_, graphgen_release_buffer, api_->ReleaseBuffer);

  return kLiteRtStatusOk;
}

LiteRtStatus
AiLiteCoreManager::LoadGraphWrapperLibrary(absl::string_view path) {
  auto loading_lib = SharedLibrary::Load(path, RtldFlags::Default());
  LITERT_LOG(LITERT_INFO, "Loading from: %s", path.data());
  if (!loading_lib.HasValue() || !loading_lib->Loaded()) {
    LITERT_LOG(LITERT_INFO, "Failed to load");
    return kLiteRtStatusErrorDynamicLoading;
  }

  graph_wrapper_lib_ = std::move(loading_lib.Value());
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, create_graph, api_->CreateGraph);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, set_graph_input_tensors,
                  api_->SetGraphInputs);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, set_graph_output_tensors,
                  api_->SetGraphOutputs);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, finish_build_graph,
                  api_->FinishGraphBuild);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, serialize, api_->Serialize);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, define_op_node, api_->DefineOp);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, add_op_parameter, api_->AddOpParam);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, define_tensor, api_->DefineTensor);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, set_quantize_param_for_tensor,
                  api_->SetTensorQParam);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, set_data_for_constant_tensor,
                  api_->SetTensorData);
  LOAD_LIB_SYMBOL(graph_wrapper_lib_, release_graph, api_->ReleaseGraph);

  return kLiteRtStatusOk;
}

} // namespace litert::samsung
