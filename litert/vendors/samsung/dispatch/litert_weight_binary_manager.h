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

#ifndef ODML_LITERT_VENDORS_SAMSUNG_DISPATCH_LITERT_WEIGHT_BINARY_MANAGER_H_
#define ODML_LITERT_VENDORS_SAMSUNG_DISPATCH_LITERT_WEIGHT_BINARY_MANAGER_H_

#pragma once

#include <atomic>
#include <list>
#include <mutex>
#include <string>

#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/dispatch/enn_manager.h"

namespace litert::samsung {

class WeightBinaryManager {
 public:
  static WeightBinaryManager& GetInstance(EnnManager* enn_manager);

  WeightBinaryManager(const WeightBinaryManager&) = delete;
  WeightBinaryManager& operator=(const WeightBinaryManager&) = delete;

  // Returns cached buffer if exists, otherwise creates new (from file-opened
  // fd)
  litert::Expected<EnnBufferPtr> Acquire(const std::string& signature, int fd,
                                         int64_t offset, size_t size);
  // Acquire from memory address without file descriptor
  litert::Expected<EnnBufferPtr> Acquire(const std::string& signature,
                                         const void* address, size_t size);

  // Decrements ref count
  litert::Expected<void> Release(const std::string& signature);

  // Clears all cached weights
  void ClearAll();

 private:
  explicit WeightBinaryManager(EnnManager* enn_manager);
  ~WeightBinaryManager();

  struct Entry {
    std::string signature;
    EnnBufferPtr buffer;
    int ref_count = 0;
  };

  EnnManager* enn_manager_;
  std::list<Entry> entries_;
  std::mutex mutex_;
};

}  // namespace litert::samsung

#endif  // ODML_LITERT_VENDORS_SAMSUNG_DISPATCH/LITERT_WEIGHT_BINARY_MANAGER_H_
