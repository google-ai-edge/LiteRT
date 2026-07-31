// Copyright (C) 2025 Intel Corporation
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_D3D12_SHARED_BUFFER_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_D3D12_SHARED_BUFFER_H_

#include "litert/c/litert_common.h"

// Cross-device shared allocation is implemented with Direct3D 12 on Windows,
// where D3D12 is the neutral allocator able to produce an OS-shareable NT
// handle that both the OpenVINO NPU (Level Zero) and GPU (OpenCL) plugins can
// import.  The whole facility is therefore Windows-only.
#if defined(LITERT_WINDOWS_OS)

#include <cstddef>
#include <memory>

#include "litert/cc/litert_expected.h"

namespace litert {
namespace openvino {

// Owns a cross-device shared D3D12 allocation that is both CPU-mappable and
// OS-shareable, so no host<->device staging copy is required.
//
// Hardware reality (measured on Intel UMA): a single committed D3D12 resource
// cannot be both OS-shareable and CPU-mappable -- `D3D12_HEAP_FLAG_SHARED` is
// rejected for any CPU-visible heap. The way around this (the same technique
// Chromium's D3D shared-image backing uses for its proven GPU/NPU WebNN/ORT
// interop path) is `ID3D12Device3::OpenExistingHeapFromAddress`:
//   1. Reserve system memory with VirtualAlloc.
//   2. Wrap it as a D3D12 heap via OpenExistingHeapFromAddress -- on UMA this
//      is the only way to get a heap that is simultaneously CPU-visible and
//      SHARED.
//   3. Export the heap's NT handle. The heap owns the backing memory and is the
//      shareable object (CreateSharedHandle only accepts heaps, committed
//      resources and fences); external-memory importers consume the raw heap
//      allocation, so no D3D12 resource is needed on our side.
// The VirtualAlloc'd pointer serves the LiteRT Lock/Unlock host contract
// directly, while the heap's NT handle is imported (zero-copy) into the NPU
// (Level Zero) and GPU (OpenCL) remote contexts. Host and device therefore see
// one allocation; on cache-coherent UMA there is nothing to copy or flush.
//
// COM types are hidden behind a PIMPL so consumers need not include <d3d12.h>.
class D3D12SharedBuffer {
 public:
  // Allocates a shared buffer of at least `size` bytes (rounded up internally
  // to the 4 KiB page granularity required for external-memory import).
  // Returns an error if any D3D12 object creation, handle creation or mapping
  // fails; callers should fall back to the host-tensor path in that case.
  static litert::Expected<std::unique_ptr<D3D12SharedBuffer>> Create(
      size_t size);

  ~D3D12SharedBuffer();

  D3D12SharedBuffer(const D3D12SharedBuffer&) = delete;
  D3D12SharedBuffer& operator=(const D3D12SharedBuffer&) = delete;

  // Host-visible pointer to the shared allocation (the VirtualAlloc'd memory
  // backing the D3D12 heap). Directly reflects device-side writes on
  // cache-coherent UMA. Never null on a created buffer.
  void* cpu_ptr() const;

  // Win32 NT handle (as void*) of the shared heap, for the OpenVINO remote-
  // context create_tensor import overloads. Owned by this object.
  void* nt_handle() const;

  // Allocation size in bytes (rounded up to the 64 KiB D3D12 buffer
  // granularity).
  size_t size() const;

  // No-op on cache-coherent UMA: the CPU pointer and the device buffer alias
  // the same memory. Retained to satisfy the LiteRT Lock/Unlock contract and
  // as a hook for an explicit flush on non-coherent parts.
  litert::Expected<void> SyncToHost();

  // No-op on cache-coherent UMA (see SyncToHost).
  litert::Expected<void> SyncFromHost();

 private:
  D3D12SharedBuffer();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_WINDOWS_OS
#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_D3D12_SHARED_BUFFER_H_
