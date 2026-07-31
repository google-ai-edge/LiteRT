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

#include <gtest/gtest.h>

#include "litert/c/litert_common.h"

// The shared allocator is Windows-only. On other platforms this translation
// unit contributes no tests (gtest_main still provides main()).
#if defined(LITERT_WINDOWS_OS)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

// clang-format off
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
// clang-format on

namespace litert {
namespace openvino {
namespace {

// Diagnostic: probe which (heap-type, shared) combinations D3D12 accepts on
// this machine, whether the resource is CPU-mappable, and whether a shared NT
// handle can be created. Prints results; never fails. Guides the final design.
TEST(D3D12Probe, ConfigurationMatrix) {
  IDXGIFactory6* factory = nullptr;
  ASSERT_EQ(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), S_OK);
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0;
       factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    adapter->GetDesc1(&d);
    if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
    adapter->Release();
    adapter = nullptr;
  }
  ASSERT_NE(adapter, nullptr);

  ID3D12Device* device = nullptr;
  ASSERT_EQ(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0,
                              IID_PPV_ARGS(&device)),
            S_OK);
  adapter->Release();
  factory->Release();

  D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
  arch.NodeIndex = 0;
  device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch));
  std::printf("[probe] UMA=%d CacheCoherentUMA=%d\n", arch.UMA,
              arch.CacheCoherentUMA);

  struct Case {
    const char* name;
    D3D12_HEAP_TYPE type;
    D3D12_CPU_PAGE_PROPERTY page;
    D3D12_MEMORY_POOL pool;
    D3D12_HEAP_FLAGS flags;
  };
  const Case cases[] = {
      {"DEFAULT+SHARED", D3D12_HEAP_TYPE_DEFAULT,
       D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN,
       D3D12_HEAP_FLAG_SHARED},
      {"CUSTOM/WB/L0+SHARED", D3D12_HEAP_TYPE_CUSTOM,
       D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0,
       D3D12_HEAP_FLAG_SHARED},
      {"CUSTOM/WB/L0+NONE", D3D12_HEAP_TYPE_CUSTOM,
       D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0,
       D3D12_HEAP_FLAG_NONE},
      {"CUSTOM/WC/L0+SHARED", D3D12_HEAP_TYPE_CUSTOM,
       D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0,
       D3D12_HEAP_FLAG_SHARED},
  };

  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = 4096;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  for (const auto& c : cases) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = c.type;
    hp.CPUPageProperty = c.page;
    hp.MemoryPoolPreference = c.pool;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;
    ID3D12Resource* res = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        &hp, c.flags, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&res));
    bool mappable = false, shareable = false;
    if (SUCCEEDED(hr)) {
      void* p = nullptr;
      mappable = SUCCEEDED(res->Map(0, nullptr, &p)) && p != nullptr;
      if (mappable) res->Unmap(0, nullptr);
      HANDLE h = nullptr;
      shareable = SUCCEEDED(device->CreateSharedHandle(res, nullptr,
                                                       GENERIC_ALL, nullptr,
                                                       &h)) &&
                  h != nullptr;
      if (h) CloseHandle(h);
      res->Release();
    }
    std::printf("[probe] %-22s create=0x%08lX mappable=%d shareable=%d\n",
                c.name, static_cast<unsigned long>(hr), mappable, shareable);
  }
  device->Release();
}

TEST(D3D12SharedBuffer, CreateExposesHandleAndMappedPointer) {
  constexpr size_t kSize = 4096;
  auto buffer = D3D12SharedBuffer::Create(kSize);
  ASSERT_TRUE(buffer) << buffer.Error().Message();

  const auto& buf = buffer.Value();
  ASSERT_NE(buf->cpu_ptr(), nullptr);
  ASSERT_NE(buf->nt_handle(), nullptr);
  EXPECT_GE(buf->size(), kSize);
}

TEST(D3D12SharedBuffer, CpuPointerRoundTrips) {
  constexpr size_t kSize = 2048;
  auto buffer = D3D12SharedBuffer::Create(kSize);
  ASSERT_TRUE(buffer) << buffer.Error().Message();

  auto* data = static_cast<uint8_t*>(buffer.Value()->cpu_ptr());
  ASSERT_NE(data, nullptr);
  for (size_t i = 0; i < kSize; ++i) {
    data[i] = static_cast<uint8_t>(i & 0xFF);
  }
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(data[i], static_cast<uint8_t>(i & 0xFF));
  }
}

TEST(D3D12SharedBuffer, SizeIsPageRounded) {
  // A sub-page request is rounded up to the 4 KiB page granularity required
  // for external-memory import.
  auto buffer = D3D12SharedBuffer::Create(1);
  ASSERT_TRUE(buffer) << buffer.Error().Message();
  EXPECT_EQ(buffer.Value()->size() % 4096, 0u);
  EXPECT_GE(buffer.Value()->size(), 4096u);
}

// Exercises the copy-on-lock machinery: write to the CPU staging resource,
// push it to the device-local shared resource (SyncFromHost), wipe staging,
// then pull it back (SyncToHost). Data must survive the round trip through the
// shared resource, proving the D3D12 copy queue + fence path works.
TEST(D3D12SharedBuffer, SyncRoundTripThroughSharedResource) {
  constexpr size_t kSize = 4096;
  auto buffer = D3D12SharedBuffer::Create(kSize);
  ASSERT_TRUE(buffer) << buffer.Error().Message();
  const auto& buf = buffer.Value();

  auto* data = static_cast<uint8_t*>(buf->cpu_ptr());
  ASSERT_NE(data, nullptr);
  for (size_t i = 0; i < kSize; ++i) {
    data[i] = static_cast<uint8_t>((i * 7) & 0xFF);
  }

  ASSERT_TRUE(buf->SyncFromHost());       // staging -> shared
  std::memset(data, 0, kSize);            // clobber staging
  ASSERT_TRUE(buf->SyncToHost());         // shared -> staging

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(data[i], static_cast<uint8_t>((i * 7) & 0xFF));
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert

#endif  // LITERT_WINDOWS_OS
