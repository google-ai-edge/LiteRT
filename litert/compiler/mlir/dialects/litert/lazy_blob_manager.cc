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
#include "litert/compiler/mlir/dialects/litert/lazy_blob_manager.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/RWMutex.h"
#include "mlir/IR/Dialect.h"
#include "litert/compiler/mlir/dialects/litert/lazy_resource_blob.h"

namespace litert {

auto LazyBlobManager::Lookup(llvm::StringRef name) -> BlobEntry* {
  llvm::sys::SmartScopedReader<true> reader(blob_map_lock_);

  auto it = blob_map_.find(name);
  return it != blob_map_.end() ? &it->second : nullptr;
}

void LazyBlobManager::Update(llvm::StringRef name,
                             LazyResourceBlob&& new_blob) {
  BlobEntry* entry = Lookup(name);
  assert(entry && "`update` expects an existing entry for the provided name");
  entry->SetBlob(std::move(new_blob));
}

auto LazyBlobManager::Insert(llvm::StringRef name,
                             std::optional<LazyResourceBlob> blob)
    -> BlobEntry& {
  llvm::sys::SmartScopedWriter<true> writer(blob_map_lock_);

  // Functor used to attempt insertion with a given name.
  auto try_insertion = [&](llvm::StringRef name) -> BlobEntry* {
    auto it = blob_map_.try_emplace(name, BlobEntry());
    if (it.second) {
      it.first->second.initialize(it.first->getKey(), std::move(blob));
      return &it.first->second;
    }
    return nullptr;
  };

  // Try inserting with the name provided by the user.
  if (BlobEntry* entry = try_insertion(name)) return *entry;

  // If an entry already exists for the user provided name, tweak the name and
  // re-attempt insertion until we find one that is unique.
  llvm::SmallString<32> name_storage(name);
  name_storage.push_back('_');
  size_t name_counter = 1;
  do {
    llvm::Twine(name_counter++).toVector(name_storage);

    // Try inserting with the new name.
    if (BlobEntry* entry = try_insertion(name_storage)) return *entry;
    name_storage.resize(name.size() + 1);
  } while (true);
}

namespace {

LazyBlobManager& GetGlobalLazyBlobManager() {
  static absl::NoDestructor<LazyBlobManager> kLazyBlobManager;
  return *kLazyBlobManager;
}

}  // namespace

LazyBlobManagerDialectInterface::LazyBlobManagerDialectInterface(
    mlir::Dialect* dialect)
    : Base(dialect), blob_manager_(GetGlobalLazyBlobManager()) {}

}  // namespace litert
