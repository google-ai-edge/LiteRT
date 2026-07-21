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

#include "litert/vendors/intel_openvino/dispatch/openvino_tensor_buffer.h"

#include <gtest/gtest.h>

#include "litert/c/litert_common.h"

// The cross-device shared allocation path is Windows-only. On other platforms
// this translation unit contributes no tests (gtest_main still provides main).
#if defined(LITERT_WINDOWS_OS)

#include <cstdint>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"

namespace {

LiteRtRankedTensorType MakeFloat32Vector(int32_t n) {
  LiteRtRankedTensorType type = {};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = 1;
  type.layout.has_strides = 0;
  type.layout.dimensions[0] = n;
  return type;
}

// Forces the "model spans both NPU and GPU" condition so Alloc() takes the
// D3D12 shared-allocation path. State is process-global on the shared-core
// singleton; every test here sets it, so the tests are order-independent.
void ForceCrossDevice() {
  OpenVINOSharedCore::GetInstance()->NoteDeviceRequested("NPU");
  OpenVINOSharedCore::GetInstance()->NoteDeviceRequested("GPU");
}

TEST(OpenVinoTensorBufferShared, AllocUsesSharedPathAndExposesHostPtr) {
  ForceCrossDevice();
  constexpr int32_t kN = 16;
  OpenVinoTensorBuffer buffer;
  auto alloc = buffer.Alloc(MakeFloat32Vector(kN), kN * sizeof(float));
  ASSERT_TRUE(alloc) << alloc.Error().Message();
  EXPECT_TRUE(buffer.is_shared());

  auto data = buffer.GetTensorData();
  ASSERT_TRUE(data) << data.Error().Message();
  EXPECT_NE(data.Value(), nullptr);
}

TEST(OpenVinoTensorBufferShared, DoubleAllocFails) {
  ForceCrossDevice();
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Alloc(MakeFloat32Vector(4), 4 * sizeof(float)));
  EXPECT_FALSE(buffer.Alloc(MakeFloat32Vector(4), 4 * sizeof(float)));
}

// Write through a write-lock (Unlock copies host->shared), then read through a
// read-lock (Lock copies shared->host). The bytes must survive the round trip
// through the device-local shared resource, proving the copy-on-lock direction
// logic and the D3D12 sync are wired correctly end to end.
TEST(OpenVinoTensorBufferShared, LockUnlockRoundTripThroughSharedResource) {
  ForceCrossDevice();
  constexpr int32_t kN = 16;
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Alloc(MakeFloat32Vector(kN), kN * sizeof(float)));

  auto w = buffer.Lock(kLiteRtTensorBufferLockModeWrite);
  ASSERT_TRUE(w) << w.Error().Message();
  auto* wptr = static_cast<float*>(w.Value());
  for (int32_t i = 0; i < kN; ++i) {
    wptr[i] = static_cast<float>(i) + 0.5f;
  }
  ASSERT_TRUE(buffer.Unlock());

  auto r = buffer.Lock(kLiteRtTensorBufferLockModeRead);
  ASSERT_TRUE(r) << r.Error().Message();
  auto* rptr = static_cast<float*>(r.Value());
  for (int32_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(rptr[i], static_cast<float>(i) + 0.5f);
  }
  ASSERT_TRUE(buffer.Unlock());
}

// A CPU-targeted partition binds the shared allocation directly through its
// host mapping, so the returned tensor must alias the Lock/GetTensorData
// pointer (no import, no copy).
TEST(OpenVinoTensorBufferShared, GetOVTensorCpuAliasesHostPtr) {
  ForceCrossDevice();
  constexpr int32_t kN = 8;
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Alloc(MakeFloat32Vector(kN), kN * sizeof(float)));

  auto host = buffer.GetTensorData();
  ASSERT_TRUE(host) << host.Error().Message();
  auto tensor = buffer.GetOVTensor("CPU");
  ASSERT_TRUE(tensor) << tensor.Error().Message();
  EXPECT_EQ(tensor.Value().data(), host.Value());
}

// ---------------------------------------------------------------------------
// The tests below require BOTH an Intel NPU and GPU to be present, since they
// import the shared D3D12 NT handle into each device's OpenVINO remote context
// (the driver-gated create_tensor path). Run manually on such hardware:
//   openvino_tensor_buffer_test.exe --gtest_filter=*CrossDevice*
// (with the OpenVINO runtime on PATH, e.g. via setupvars.bat).
// ---------------------------------------------------------------------------

// Importing the same shared NT handle must succeed into both the NPU (Level
// Zero SHARED_BUF) and GPU (BUFFER_FROM_HANDLE) contexts, yielding tensors of
// the expected byte size. This is the driver-capability gate.
TEST(OpenVinoTensorBufferCrossDevice, ImportsIntoNpuAndGpu) {
  ForceCrossDevice();
  constexpr int32_t kN = 16;
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Alloc(MakeFloat32Vector(kN), kN * sizeof(float)));

  auto npu = buffer.GetOVTensor("NPU");
  ASSERT_TRUE(npu) << npu.Error().Message();
  EXPECT_EQ(npu.Value().get_byte_size(), kN * sizeof(float));

  auto gpu = buffer.GetOVTensor("GPU");
  ASSERT_TRUE(gpu) << gpu.Error().Message();
  EXPECT_EQ(gpu.Value().get_byte_size(), kN * sizeof(float));
}

// Proof of zero-copy sharing: a pattern written by the CPU into the device
// -local shared resource is read back through BOTH the NPU and GPU imports.
// They observe the same physical allocation (the NT handle aliases it).
TEST(OpenVinoTensorBufferCrossDevice, NpuAndGpuAliasTheSharedAllocation) {
  ForceCrossDevice();
  constexpr int32_t kN = 16;
  const ov::Shape shape{static_cast<size_t>(kN)};
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Alloc(MakeFloat32Vector(kN), kN * sizeof(float)));

  // CPU write -> device-local shared resource (Unlock runs SyncFromHost).
  auto w = buffer.Lock(kLiteRtTensorBufferLockModeWrite);
  ASSERT_TRUE(w) << w.Error().Message();
  auto* wptr = static_cast<float*>(w.Value());
  for (int32_t i = 0; i < kN; ++i) {
    wptr[i] = static_cast<float>(i) * 2.0f + 1.0f;
  }
  ASSERT_TRUE(buffer.Unlock());

  auto npu = buffer.GetOVTensor("NPU");
  ASSERT_TRUE(npu) << npu.Error().Message();
  ov::Tensor npu_host(ov::element::f32, shape);
  npu.Value().as<ov::RemoteTensor>().copy_to(npu_host);
  const auto* np = static_cast<const float*>(npu_host.data());
  for (int32_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(np[i], static_cast<float>(i) * 2.0f + 1.0f);
  }

  auto gpu = buffer.GetOVTensor("GPU");
  ASSERT_TRUE(gpu) << gpu.Error().Message();
  ov::Tensor gpu_host(ov::element::f32, shape);
  gpu.Value().as<ov::RemoteTensor>().copy_to(gpu_host);
  const auto* gp = static_cast<const float*>(gpu_host.data());
  for (int32_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(gp[i], static_cast<float>(i) * 2.0f + 1.0f);
  }
}

// The full boundary direction: a write issued through the GPU import must be
// visible to the CPU read path (and thus to the NPU) -- i.e. GPU-produced
// output is consumed downstream without a copy. Mirrors an NPU<->GPU boundary.
TEST(OpenVinoTensorBufferCrossDevice, GpuWriteVisibleToCpu) {
  ForceCrossDevice();
  constexpr int32_t kN = 16;
  const ov::Shape shape{static_cast<size_t>(kN)};
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Alloc(MakeFloat32Vector(kN), kN * sizeof(float)));

  auto gpu = buffer.GetOVTensor("GPU");
  ASSERT_TRUE(gpu) << gpu.Error().Message();
  ov::Tensor src(ov::element::f32, shape);
  auto* sp = static_cast<float*>(src.data());
  for (int32_t i = 0; i < kN; ++i) {
    sp[i] = 100.0f + static_cast<float>(i);
  }
  // Host -> device-local shared resource, via the GPU import. (copy_from is
  // non-const, so bind the imported tensor to a mutable RemoteTensor first.)
  ov::RemoteTensor gpu_remote = gpu.Value().as<ov::RemoteTensor>();
  gpu_remote.copy_from(src);

  // CPU read path: Lock(Read) runs SyncToHost (shared -> staging).
  auto r = buffer.Lock(kLiteRtTensorBufferLockModeRead);
  ASSERT_TRUE(r) << r.Error().Message();
  const auto* rp = static_cast<const float*>(r.Value());
  for (int32_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(rp[i], 100.0f + static_cast<float>(i));
  }
  ASSERT_TRUE(buffer.Unlock());
}

}  // namespace

#endif  // LITERT_WINDOWS_OS
