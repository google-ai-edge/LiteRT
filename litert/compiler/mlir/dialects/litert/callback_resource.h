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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_CALLBACK_RESOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_CALLBACK_RESOURCE_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/RWMutex.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpImplementation.h"

namespace litert {

class CallbackResourceManager;

class CallbackResourceBase {
 public:
  CallbackResourceBase() = default;
  virtual ~CallbackResourceBase() = default;

  llvm::StringRef GetKey() const { return key_; }

  // Applies the bytes to the given applier function. The applier function can
  // be called multiple times for chunked data.
  virtual llvm::Error ApplyBytes(
      llvm::function_ref<llvm::Error(llvm::StringRef)> applier) = 0;

  virtual void Cleanup() = 0;

  // Returns the bytes of the resource as a vector of uint8_t.
  llvm::Expected<std::vector<uint8_t>> GetBytes() {
    std::vector<uint8_t> bytes;
    auto applier = [&bytes](llvm::StringRef chunk) {
      bytes.insert(bytes.end(), chunk.begin(), chunk.end());
      return llvm::Error::success();
    };
    auto error = ApplyBytes(applier);
    if (error) {
      return error;
    }
    return bytes;
  };

 private:
  CallbackResourceBase(CallbackResourceBase&&) = default;
  CallbackResourceBase& operator=(const CallbackResourceBase&) = delete;
  CallbackResourceBase& operator=(CallbackResourceBase&&) = delete;

  void SetKey(llvm::StringRef key) { key_ = key; }

  llvm::StringRef key_;
  friend CallbackResourceManager;
  friend class llvm::StringMapEntryStorage<CallbackResourceBase>;
};

class CallbackResourceManager {
 public:
  CallbackResourceBase* Lookup(llvm::StringRef name);
  CallbackResourceBase* Insert(llvm::StringRef name,
                               std::unique_ptr<CallbackResourceBase> resource);

  template <typename HandleT>
  HandleT Insert(typename HandleT::Dialect* dialect, llvm::StringRef name,
                 std::unique_ptr<CallbackResourceBase> resource) {
    CallbackResourceBase* entry = Insert(name, std::move(resource));
    return HandleT(entry, dialect);
  }

 private:
  mutable llvm::sys::SmartRWMutex<true> resources_lock_;
  llvm::StringMap<std::unique_ptr<CallbackResourceBase>> resources_;
};

class CallbackResourceManagerDialectInterface
    : public mlir::DialectInterface::Base<
          CallbackResourceManagerDialectInterface> {
 public:
  explicit CallbackResourceManagerDialectInterface(mlir::Dialect* dialect);

  CallbackResourceManager& GetCallbackResourceManager() { return manager_; }
  const CallbackResourceManager& GetCallbackResourceManager() const {
    return manager_;
  }

 private:
  CallbackResourceManager& manager_;
};

template <typename HandleT>
class CallbackResourceManagerDialectInterfaceBase
    : public CallbackResourceManagerDialectInterface {
 public:
  using CallbackResourceManagerDialectInterface::
      CallbackResourceManagerDialectInterface;
};

template <typename DialectT>
struct CallbackResourceHandle
    : public mlir::AsmDialectResourceHandleBase<
          CallbackResourceHandle<DialectT>, CallbackResourceBase, DialectT> {
  using mlir::AsmDialectResourceHandleBase<
      CallbackResourceHandle<DialectT>, CallbackResourceBase,
      DialectT>::AsmDialectResourceHandleBase;
  using ManagerInterface = CallbackResourceManagerDialectInterfaceBase<
      CallbackResourceHandle<DialectT>>;

  llvm::StringRef GetKey() const { return this->getResource()->GetKey(); }

  /// Get the interface for the dialect that owns handles of this type.
  /// Asserts that the dialect is registered.
  static ManagerInterface& getManagerInterface(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<DialectT>();
    assert(dialect && "dialect not registered");

    auto* iface = dialect->template getRegisteredInterface<ManagerInterface>();
    assert(iface && "dialect doesn't provide the blob manager interface?");
    return *iface;
  }
};
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_CALLBACK_RESOURCE_H_
