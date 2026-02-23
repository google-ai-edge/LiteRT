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
#include "litert/vendors/samsung/compiler/compile_model.h"

#include "litert/vendors/samsung/ai_litecore_manager.h"

#include <string.h>
#include <vector>

namespace litert::samsung {

Expected<std::vector<char>> Compile(AiLiteCoreManager::Ptr ai_lite_core,
                                    const std::vector<char> &g_buffer) {
  auto expected_backend_handler = ai_lite_core->CreateBackendHandler();
  if (!expected_backend_handler.HasValue()) {
    return expected_backend_handler.Error();
  }

  auto backend_handler = std::move(expected_backend_handler.Value());

  if (auto init_result = ai_lite_core->Api().InitializeBackendContext(
          backend_handler.get(), 9955);
      init_result != ::GraphGenResult::SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Samsung Backend initialize failed.");
  }

  NNCBuffer *nnc_buffer = nullptr;
  auto *buf_head =
      const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(g_buffer.data()));
  auto compile_result = ai_lite_core->Api().BackendCompile(
      backend_handler.get(), buf_head, g_buffer.size() * sizeof(char),
      &nnc_buffer);
  if (compile_result != ::GraphGenResult::SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Fail to compile graph with Samsung backend");
  }

  std::vector<char> compile_binary(nnc_buffer->size);
  memcpy(compile_binary.data(), nnc_buffer->addr, nnc_buffer->size);
  ai_lite_core->Api().ReleaseBuffer(backend_handler.get(), nnc_buffer);

  return compile_binary;
}

} // namespace litert::samsung
