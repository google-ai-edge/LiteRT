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

// Minimal fake Dawn shared library exporting the core required WebGPU symbols
// for unit testing dynamic runtime loading without a physical GPU.

#include <atomic>

#include "dawn/webgpu.h"  // from @dawn

namespace {
std::atomic<int> g_create_instance_calls{0};
}  // namespace

#if defined(_WIN32)
#define FAKE_DAWN_EXPORT __declspec(dllexport)
#else
#define FAKE_DAWN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

FAKE_DAWN_EXPORT int FakeDawnGetCreateInstanceCalls() {
  return g_create_instance_calls.load(std::memory_order_relaxed);
}

FAKE_DAWN_EXPORT WGPUInstance
wgpuCreateInstance(WGPUInstanceDescriptor const* /*descriptor*/) {
  g_create_instance_calls.fetch_add(1, std::memory_order_relaxed);
  return nullptr;
}

FAKE_DAWN_EXPORT WGPUFuture wgpuInstanceRequestAdapter(
    WGPUInstance /*instance*/, WGPURequestAdapterOptions const* /*options*/,
    WGPURequestAdapterCallbackInfo /*callback_info*/) {
  return {0};
}

FAKE_DAWN_EXPORT WGPUFuture wgpuAdapterRequestDevice(
    WGPUAdapter /*adapter*/, WGPUDeviceDescriptor const* /*descriptor*/,
    WGPURequestDeviceCallbackInfo /*callback_info*/) {
  return {0};
}

FAKE_DAWN_EXPORT WGPUShaderModule wgpuDeviceCreateShaderModule(
    WGPUDevice /*device*/, WGPUShaderModuleDescriptor const* /*descriptor*/) {
  return nullptr;
}

FAKE_DAWN_EXPORT WGPUComputePipeline wgpuDeviceCreateComputePipeline(
    WGPUDevice /*device*/,
    WGPUComputePipelineDescriptor const* /*descriptor*/) {
  return nullptr;
}

FAKE_DAWN_EXPORT WGPUBuffer wgpuDeviceCreateBuffer(
    WGPUDevice /*device*/, WGPUBufferDescriptor const* /*descriptor*/) {
  return nullptr;
}

FAKE_DAWN_EXPORT WGPUQueue wgpuDeviceGetQueue(WGPUDevice /*device*/) {
  return nullptr;
}

}  // extern "C"
