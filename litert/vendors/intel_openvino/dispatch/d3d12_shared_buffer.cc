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

#include "litert/vendors/intel_openvino/dispatch/d3d12_shared_buffer.h"

#include "litert/c/litert_common.h"

#if defined(LITERT_WINDOWS_OS)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

// clang-format off
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
// clang-format on

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {
namespace openvino {

using Microsoft::WRL::ComPtr;

struct D3D12SharedBuffer::Impl {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12Heap> heap;
  HANDLE nt_handle = nullptr;
  void* cpu_ptr = nullptr;  // VirtualAlloc'd system memory that backs `heap`
  size_t size = 0;

  ~Impl() {
    // Release the heap before the backing pages are handed back to the OS.
    heap.Reset();
    if (cpu_ptr) ::VirtualFree(cpu_ptr, 0, MEM_RELEASE);
    if (nt_handle) CloseHandle(nt_handle);
  }
};

namespace {

// D3D12 allocates buffer resources in multiples of 64 KiB; the shared heap must
// match so the whole allocation is importable by the external-memory APIs.
constexpr uint64_t kHeapAlignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

uint64_t RoundUpToHeapAlignment(uint64_t size) {
  return (size + kHeapAlignment - 1) & ~(kHeapAlignment - 1);
}

// Selects the first hardware (non-software/WARP) DXGI adapter. On a UMA system
// with a single Intel adapter this is the iGPU that both the NPU and GPU
// OpenVINO plugins resolve to; picking the hardware adapter guarantees the
// shared handle is importable by the GPU OpenCL device.
litert::Expected<ComPtr<IDXGIAdapter1>> PickHardwareAdapter() {
  ComPtr<IDXGIFactory6> factory;
  if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create DXGI factory");
  }
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0;
       factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      return adapter;
    }
    adapter.Reset();
  }
  return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                            "No hardware DXGI adapter found");
}

}  // namespace

// static
litert::Expected<std::unique_ptr<D3D12SharedBuffer>> D3D12SharedBuffer::Create(
    size_t size) {
  // D3D12 rounds buffer allocations up to 64 KiB; size the heap to match.
  const uint64_t alloc_size = RoundUpToHeapAlignment(size);

  LITERT_ASSIGN_OR_RETURN(ComPtr<IDXGIAdapter1> adapter, PickHardwareAdapter());

  auto impl = std::make_unique<Impl>();
  impl->size = static_cast<size_t>(alloc_size);

  if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                               IID_PPV_ARGS(&impl->device)))) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create D3D12 device");
  }

  // ID3D12Device3::OpenExistingHeapFromAddress is the only way on a UMA part to
  // obtain a heap that is simultaneously CPU-visible and OS-shareable: a plain
  // DEFAULT heap is shareable but not CPU-mappable, and a CPU-visible CUSTOM
  // heap rejects the SHARED flag. Chromium's D3D shared-image backing uses this
  // exact technique for its GPU/NPU (WebNN/ORT) interop path, which is proven
  // to import into both the OpenVINO GPU (OpenCL) and NPU (Level Zero) plugins,
  // So we reference its implementation here:
  // https://source.chromium.org/chromium/chromium/src/+/main:gpu/command_buffer/service/shared_image/d3d_image_backing_factory.cc;drc=4863ce7e2ffd140f7765530e45d67903fd964a69
  ComPtr<ID3D12Device3> device3;
  if (FAILED(impl->device.As(&device3))) {
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "ID3D12Device3 unavailable (OpenExistingHeapFromAddress required)");
  }

  // A UMA part that is not cache-coherent benefits from write-combined CPU
  // pages for faster device-visible writes.
  D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
  arch.NodeIndex = 0;
  impl->device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch,
                                    sizeof(arch));
  DWORD page_protection = PAGE_READWRITE;
  if (!arch.UMA || !arch.CacheCoherentUMA) {
    page_protection |= PAGE_WRITECOMBINE;
  }

  // Reserve the system memory that backs the shared heap. This same pointer is
  // handed to the CPU (Lock/Unlock) -- host and device observe one allocation.
  impl->cpu_ptr = ::VirtualAlloc(nullptr, static_cast<SIZE_T>(alloc_size),
                                 MEM_RESERVE | MEM_COMMIT, page_protection);
  if (impl->cpu_ptr == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to reserve system memory for D3D12 heap");
  }

  HRESULT heap_hr = device3->OpenExistingHeapFromAddress(
      impl->cpu_ptr, IID_PPV_ARGS(&impl->heap));
  if (FAILED(heap_hr)) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "Failed to open D3D12 heap from system-memory address: 0x%08lX",
             static_cast<unsigned long>(heap_hr));
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, msg);
  }

  // Export the heap's NT handle for import into the NPU and GPU remote
  // contexts. The heap -- not a placed resource -- is the shareable object:
  // CreateSharedHandle only accepts heaps, committed resources and fences, and
  // a CPU-visible heap rejects a UAV placed resource anyway (E_INVALIDARG).
  // External-memory importers consume the raw heap allocation, so no D3D12
  // resource is needed on our side.
  HRESULT handle_hr = impl->device->CreateSharedHandle(
      impl->heap.Get(), nullptr, GENERIC_ALL, nullptr, &impl->nt_handle);
  if (FAILED(handle_hr) || impl->nt_handle == nullptr) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "Failed to create D3D12 shared NT handle: 0x%08lX",
             static_cast<unsigned long>(handle_hr));
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, msg);
  }

  auto buffer = std::unique_ptr<D3D12SharedBuffer>(new D3D12SharedBuffer());
  buffer->impl_ = std::move(impl);
  return buffer;
}

D3D12SharedBuffer::D3D12SharedBuffer() = default;
D3D12SharedBuffer::~D3D12SharedBuffer() = default;

void* D3D12SharedBuffer::cpu_ptr() const { return impl_->cpu_ptr; }
void* D3D12SharedBuffer::nt_handle() const { return impl_->nt_handle; }
size_t D3D12SharedBuffer::size() const { return impl_->size; }

// The shared heap *is* the CPU allocation (VirtualAlloc'd system memory opened
// via OpenExistingHeapFromAddress), so host and device observe the same bytes.
// On cache-coherent UMA no copy or flush is required; these remain to satisfy
// the LiteRT Lock/Unlock contract.
litert::Expected<void> D3D12SharedBuffer::SyncToHost() { return {}; }

litert::Expected<void> D3D12SharedBuffer::SyncFromHost() { return {}; }

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_WINDOWS_OS
