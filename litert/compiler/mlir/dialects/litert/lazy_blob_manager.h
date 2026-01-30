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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_LAZY_BLOB_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_LAZY_BLOB_MANAGER_H_

#include <cassert>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"  // from @com_google_absl
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/RWMutex.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "litert/compiler/mlir/dialects/litert/lazy_resource_blob.h"

namespace litert {

class LazyBlobManager {
 public:
  class BlobEntry {
   public:
    llvm::StringRef GetKey() const { return key_; }

    const LazyResourceBlob* GetBlob() const {
      return blob_ ? &*blob_ : nullptr;
    }
    LazyResourceBlob* GetBlob() { return blob_ ? &*blob_ : nullptr; }

    void SetBlob(LazyResourceBlob&& new_blob) { blob_ = std::move(new_blob); }

   private:
    BlobEntry() = default;
    BlobEntry(BlobEntry&&) = default;
    BlobEntry& operator=(const BlobEntry&) = delete;
    BlobEntry& operator=(BlobEntry&&) = delete;

    void initialize(llvm::StringRef new_key,
                    std::optional<LazyResourceBlob> new_blob) {
      key_ = new_key;
      blob_ = std::move(new_blob);
    }

    llvm::StringRef key_;

    std::optional<LazyResourceBlob> blob_;

    /// Allow access to the constructors.
    friend LazyBlobManager;
    friend class llvm::StringMapEntryStorage<BlobEntry>;
  };

  ~LazyBlobManager() {
    // TODO(aarfaian): see if there's a more elegant way of handling cleanup
    // rather than in the destructor. Ultimately will be dictated by what's
    // provided by MLIR, since the framework handles the lifecycle of the blob
    // manager.
    for (auto& entry : blob_map_) {
      entry.getValue().GetBlob()->Cleanup();
    }
  }

  BlobEntry* Lookup(llvm::StringRef name);
  const BlobEntry* Lookup(llvm::StringRef name) const {
    return const_cast<LazyBlobManager*>(this)->Lookup(name);
  }

  void Update(llvm::StringRef name, LazyResourceBlob&& new_blob);

  BlobEntry& Insert(llvm::StringRef name,
                    std::optional<LazyResourceBlob> blob = {});

  template <typename HandleT>
  HandleT Insert(typename HandleT::Dialect* dialect, llvm::StringRef name,
                 std::optional<LazyResourceBlob> blob = {}) {
    BlobEntry& entry = Insert(name, std::move(blob));
    return HandleT(&entry, dialect);
  }

 private:
  mutable llvm::sys::SmartRWMutex<true> blob_map_lock_;
  llvm::StringMap<BlobEntry> blob_map_;
};

class LazyBlobManagerDialectInterface
    : public mlir::DialectInterface::Base<LazyBlobManagerDialectInterface> {
 public:
  explicit LazyBlobManagerDialectInterface(mlir::Dialect* dialect);

  LazyBlobManager& GetBlobManager() { return blob_manager_; }
  const LazyBlobManager& GetBlobManager() const { return blob_manager_; }

 private:
  LazyBlobManager& blob_manager_;
};

template <typename HandleT>
class LazyBlobManagerDialectInterfaceBase
    : public LazyBlobManagerDialectInterface {
 public:
  using LazyBlobManagerDialectInterface::LazyBlobManagerDialectInterface;

  /// Update the blob for the entry defined by the provided name. This method
  /// asserts that an entry for the given name exists in the manager.
  void Update(llvm::StringRef name, LazyResourceBlob&& new_blob) {
    GetBlobManager().Update(name, std::move(new_blob));
  }

  /// Insert a new resource blob entry with the provided name and optional blob
  /// data. The name may be modified during insertion if another entry already
  /// exists with that name. Returns a dialect specific handle to the inserted
  /// entry.
  HandleT Insert(llvm::StringRef name,
                 std::optional<LazyResourceBlob> blob = {}) {
    return GetBlobManager().template Insert<HandleT>(
        cast<typename HandleT::Dialect>(getDialect()), name, std::move(blob));
  }
};

template <typename DialectT>
struct LazyResourceBlobHandle : public mlir::AsmDialectResourceHandleBase<
                                    LazyResourceBlobHandle<DialectT>,
                                    LazyBlobManager::BlobEntry, DialectT> {
  using mlir::AsmDialectResourceHandleBase<
      LazyResourceBlobHandle<DialectT>, LazyBlobManager::BlobEntry,
      DialectT>::AsmDialectResourceHandleBase;
  using ManagerInterface =
      LazyBlobManagerDialectInterfaceBase<LazyResourceBlobHandle<DialectT>>;

  /// Return the human readable string key for this handle.
  llvm::StringRef GetKey() const { return this->getResource()->GetKey(); }

  /// Return the blob referenced by this handle if the underlying resource has
  /// been initialized. Returns nullptr otherwise.
  LazyResourceBlob* GetBlob() { return this->getResource()->GetBlob(); }
  const LazyResourceBlob* GetBlob() const {
    return this->getResource()->GetBlob();
  }

  /// Get the interface for the dialect that owns handles of this type. Asserts
  /// that the dialect is registered.
  static ManagerInterface& getManagerInterface(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<DialectT>();
    assert(dialect && "dialect not registered");

    auto* iface = dialect->template getRegisteredInterface<ManagerInterface>();
    assert(iface && "dialect doesn't provide the blob manager interface?");
    return *iface;
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_LAZY_BLOB_MANAGER_H_
