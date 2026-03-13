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

#ifndef LITERT_VENDORS_SAMSUNG_ENN_MANAGER_H_
#define LITERT_VENDORS_SAMSUNG_ENN_MANAGER_H_

#include "litert/vendors/samsung/dispatch/enn_type.h"
#include <memory>

#include "litert/cc/litert_expected.h"

#include "litert/cc/internal/litert_shared_library.h"

namespace litert::samsung {

class EnnManager {
public:
  using UniquePtr = std::unique_ptr<EnnManager>;
  using Ptr = EnnManager *;
  struct PublicApi;

  EnnManager(EnnManager &) = delete;
  EnnManager(EnnManager &&) = delete;
  EnnManager &operator=(const EnnManager &) = delete;
  EnnManager &operator=(EnnManager &&) = delete;

  static Expected<EnnManager::UniquePtr> Create();

  const PublicApi &Api() const;
  ~EnnManager();

private:
  EnnManager();
// Loads and resolve compiler related api
  LiteRtStatus LoadEnnRuntimeLibrary(absl::string_view path);

  SharedLibrary enn_runtime_lib_;
  std::unique_ptr<PublicApi> api_;
};

struct EnnManager::PublicApi {
  EnnReturn (*EnnInitialize)(void);
  EnnReturn (*EnnOpenModelFromMemory)(const char *va, const uint32_t size,
                                      EnnModelId *model_id);
  EnnReturn (*EnnCreateBufferFromFdWithOffset)(const uint32_t fd,
                                               const uint32_t size,
                                               const uint32_t offset,
                                               EnnBufferPtr *out);
  EnnReturn (*EnnAllocateAllBuffers)(const EnnModelId model_id,
                                     EnnBufferPtr **out_buffers,
                                     NumberOfBuffersInfo *out_buffers_info);
  EnnReturn (*EnnBufferCommit)(const EnnModelId model_id);
  EnnReturn (*EnnGetBuffersInfo)(const EnnModelId model_id,
                                 NumberOfBuffersInfo *buffers_info);
  EnnReturn (*EnnSetBufferByIndex)(const EnnModelId model_id,
                                   const enn_buf_dir_e direction,
                                   const uint32_t index, EnnBufferPtr buf);
  EnnReturn (*EnnReleaseBuffer)(EnnBufferPtr buffer);
  EnnReturn (*EnnExecuteModel)(const EnnModelId model_id);
  EnnReturn (*EnnBufferUncommit)(const EnnModelId model_id);
  EnnReturn (*EnnUnsetBuffers)(const EnnModelId model_id);
  EnnReturn (*EnnCloseModel)(const EnnModelId model_id);
  EnnReturn (*EnnDeinitialize)(void);
};

} // namespace litert::samsung

#endif // LITERT_VENDORS_SAMSUNG_ENN_MANAGER_H_
