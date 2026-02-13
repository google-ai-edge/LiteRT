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
#include "litert/compiler/mlir/dialects/litert/callback_resource.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/RWMutex.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"

namespace litert {

CallbackResourceBase* CallbackResourceManager::Lookup(llvm::StringRef name) {
  llvm::sys::SmartScopedReader<true> reader(resources_lock_);

  auto it = resources_.find(name);
  return it != resources_.end() ? it->second.get() : nullptr;
}

CallbackResourceBase* CallbackResourceManager::Insert(
    llvm::StringRef name, std::unique_ptr<CallbackResourceBase> resource) {
  // Guard the map for writing
  llvm::sys::SmartScopedWriter<true> writer(resources_lock_);

  // Helper to attempt insertion and initialize the resource on success
  auto try_insertion =
      [&](llvm::StringRef name_to_try) -> CallbackResourceBase* {
    auto [it, inserted] =
        resources_.try_emplace(name_to_try, std::move(resource));
    if (inserted) {
      // The resource is now owned by the map. We set its internal key_
      // to point to the StringMap's stable key storage.
      CallbackResourceBase* ptr = it->second.get();
      ptr->SetKey(it->first());
      return ptr;
    }
    return nullptr;
  };

  // Try inserting with the exact name provided.
  if (CallbackResourceBase* entry = try_insertion(name)) {
    return entry;
  }

  // If an entry already exists for the user provided name, tweak the name and
  // re-attempt insertion until we find one that is unique.
  llvm::SmallString<32> name_storage(name);
  name_storage.push_back('_');
  size_t name_counter = 1;
  do {
    llvm::Twine(name_counter++).toVector(name_storage);

    // Try inserting with the new name.
    if (CallbackResourceBase* entry = try_insertion(name_storage)) {
      return entry;
    }
    name_storage.resize(name.size() + 1);
  } while (true);
}

CallbackResourceManagerDialectInterface::
    CallbackResourceManagerDialectInterface(mlir::Dialect* dialect)
    : Base(dialect) {}

}  // namespace litert
