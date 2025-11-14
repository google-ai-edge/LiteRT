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

#include "litert/vendors/intel_openvino/dispatch/device_context.h"

#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include <string.h>

#include <cstdint>
#include <vector>
#include <regex>
#        include <wrl.h>
#        include <initguid.h>  // it has to be placed before dxcore
#include <iostream>

#if LITERT_HAS_AHWB_SUPPORT
#include <sys/socket.h>
#include <unistd.h>
#endif  // LITERT_HAS_AHWB_SUPPORT

#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/remote_context.hpp>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/vendors/intel_openvino/utils.h"
#include "litert/vendors/intel_openvino/dispatch/platform_functions.h"
#include "litert/vendors/intel_openvino/dispatch/dxcore.h"
#include "litert/vendors/intel_openvino/dispatch/dxcore_interface.h"
#include "litert/vendors/intel_openvino/dispatch/d3dx12_core.h"

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create() {
  return Ptr(new LiteRtDispatchDeviceContextT());
}

#if LITERT_HAS_AHWB_SUPPORT
litert::Expected<int> GetFdFromUnixHandle(AHardwareBuffer *ahwb) {
  int socks[2];
  if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, socks) != 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create socket pair");
  }

  auto socket_cleaup = absl::Cleanup([&socks] {
    close(socks[0]);
    close(socks[1]);
  });

  if (AHardwareBuffer_sendHandleToUnixSocket(ahwb, socks[0]) != 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to send handle to unix socket");
  }
  // Receives a fd(an int) over the unix socket, sets up control buffer to
  // receive an int
  char payload_byte;
  struct iovec io = {.iov_base = &payload_byte,
                     .iov_len = sizeof(payload_byte)};

  // Buffer for receiving fd
  char control_buf[CMSG_SPACE(sizeof(int))];

  struct msghdr msg = {.msg_iov = &io,
                       .msg_iovlen = 1,
                       .msg_control = control_buf,
                       .msg_controllen = sizeof(control_buf)};

  if (recvmsg(socks[1], &msg, 0) < 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to receive socket message");
  }
  int fd = -1;
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  }

  return fd;
}
#endif  // LITERT_HAS_AHWB_SUPPORT

// #ifdef _WIN32
// #        include <initguid.h>  // it has to be placed before dxcore
// #endif

// #ifdef _WIN32
// #        ifndef NOMINMAX
// #            define NOMINMAX
// #            define NOMINMAX_DEFINED_CTX_UT
// #        endif

// #        include <combaseapi.h>
// #        include <d3d12.h>
// #        include <d3dcommon.h>
// #        include <dxcore.h>
// #        include <dxcore_interface.h>
// #        include <wrl.h>
// #include <dxgi1_4.h>
// #        include <wrl/client.h>

// // #        include "d3dx12_core.h"

// #        ifdef NOMINMAX_DEFINED_CTX_UT
// #            undef NOMINMAX
// #            undef NOMINMAX_DEFINED_CTX_UT
// #        endif

#define RETURN_IF_FAILED(res, error_message)                                 \
  do {                                                             \
    if (res != S_OK) {                                              \
      LITERT_LOG(LITERT_ERROR,    error_message);                          \
      return;                                                   \
    }                                                              \
  } while (0)

// namespace ml {
class DX12RemoteRun {
public:
    // std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    // ov::AnyMap configuration;

    Microsoft::WRL::ComPtr<IDXCoreAdapter> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device9> device;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed_resources = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> comitted_resource;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback_resource;

public:

    HANDLE shared_mem = nullptr;

    void createAdapter() {
       PlatformFunctions* platform_functions = PlatformFunctions::GetInstance();
        if (!platform_functions || !platform_functions->IsDXCoreSupported()) {
          LITERT_LOG(LITERT_ERROR,    "DXCore is not supported on this platform.");
        }

        PlatformFunctions::DXCoreCreateAdapterFactoryProc
        dxcore_create_adapter_factory_proc =
            platform_functions->dxcore_create_adapter_factory_proc();
        if (!dxcore_create_adapter_factory_proc) {
          LITERT_LOG(LITERT_ERROR,    "Failed to get DXCoreCreateAdapterFactory function.");
        }

        Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> dxcore_factory;
        HRESULT hr =
            dxcore_create_adapter_factory_proc(IID_PPV_ARGS(&dxcore_factory));
        if (FAILED(hr)) {
          LITERT_LOG(LITERT_ERROR,    "Failed to create adapter factory.");
        }

        // const auto regex = std::regex(L"^\\bIntel\\b.*?\\bGraphics\\b.*?");
        const GUID guids[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE};

        // create the adapter list
        Microsoft::WRL::ComPtr<IDXCoreAdapterList> adapter_list;
        hr = dxcore_factory->CreateAdapterList(ARRAYSIZE(guids), guids, IID_PPV_ARGS(adapter_list.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(hr, "CreateAdapterList failed.");

      LITERT_LOG(LITERT_ERROR,    "======CreateAdapterList %d.", adapter_list->GetAdapterCount());
        // find our adapter
        for (uint32_t iter = 0; iter < adapter_list->GetAdapterCount(); iter++) {
            Microsoft::WRL::ComPtr<IDXCoreAdapter> local_adapter;
            hr = adapter_list->GetAdapter(iter, IID_PPV_ARGS(local_adapter.ReleaseAndGetAddressOf()));
            RETURN_IF_FAILED(hr, "GetAdapter failed.");

            // size_t driver_desc_size = 0;
            // hr = local_adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &driver_desc_size);
            // RETURN_IF_FAILED(hr, "GetPropertySize failed.");

            // LITERT_LOG(LITERT_ERROR,    "======driver_desc_size %d.", driver_desc_size);

            // // std::vector<char> driver_desc(driver_desc_size);
            // std::vector<wchar_t> driver_desc(driver_desc_size / sizeof(wchar_t));
            // hr =
            //     local_adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, driver_desc_size,driver_desc.data());
            // RETURN_IF_FAILED(hr, "GetProperty failed.");
            //  std::wcout << L"Adapter "  << L": " << driver_desc.data() << L"\n";

            // if (std::regex_match(std::string(driver_desc.data()), regex)) {
                adapter = local_adapter;
            //     break;
            // }
            break;
        }

        LITERT_LOG(LITERT_ERROR,    "==========.");

        auto check_adapter = adapter->IsValid();
        if (!check_adapter) {
            LITERT_LOG(LITERT_ERROR,    "======GPU adapter is not valid.");
        }
        LITERT_LOG(LITERT_ERROR,    "======GPU adapter is valid.");

        
      auto d3d12_create_device_proc =
      platform_functions->d3d12_create_device_proc();
        auto res =
            d3d12_create_device_proc(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "D3D12CreateDevice failed.");
        LITERT_LOG(LITERT_ERROR,    "======D3D12CreateDevice  success.");
    }


    void createHeap(const size_t byte_size) {
        const size_t size = (byte_size + (static_cast<size_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1)) &
                            ~(static_cast<size_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1);

        D3D12_HEAP_DESC desc_heap{};
        desc_heap.SizeInBytes = size;
        desc_heap.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
        desc_heap.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
        desc_heap.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        desc_heap.Properties.CreationNodeMask = 1;
        desc_heap.Properties.VisibleNodeMask = 1;
        desc_heap.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc_heap.Flags = D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER | D3D12_HEAP_FLAG_SHARED;
        auto res = device->CreateHeap(&desc_heap, IID_PPV_ARGS(heap.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateHeap failed.");

        res = device->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_mem);
        RETURN_IF_FAILED(res, "CreateSharedHandle failed.");
        LITERT_LOG(LITERT_ERROR,    "======createHeap  success.");
    }

    void createPlacedResources(const size_t byte_size) {
        D3D12_RESOURCE_DESC desc_resource = {};
        desc_resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc_resource.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc_resource.Width = byte_size;
        desc_resource.Height = 1;
        desc_resource.DepthOrArraySize = 1;
        desc_resource.MipLevels = 1;
        desc_resource.Format = DXGI_FORMAT_UNKNOWN;
        desc_resource.SampleDesc.Count = 1;
        desc_resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc_resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        auto res = device->CreatePlacedResource(heap.Get(),
                                                0,
                                                &desc_resource,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                nullptr,
                                                IID_PPV_ARGS(placed_resources.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreatePlacedResource failed.");
    }

    void createComittedResources(const size_t byte_size) {
       CD3DX12_HEAP_PROPERTIES dx12_heap_properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
       CD3DX12_RESOURCE_DESC dx12_resource_desc = CD3DX12_RESOURCE_DESC::Buffer(byte_size);
        auto res = device->CreateCommittedResource(&dx12_heap_properties,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &dx12_resource_desc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr,
                                                   IID_PPV_ARGS(comitted_resource.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommittedResource failed.");
    }

    void createResources(const size_t byte_size) {
        createHeap(byte_size);
        createPlacedResources(byte_size);
        createComittedResources(byte_size);
    }

    void copyResources(const size_t byte_size) {
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> command_list;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        uint32_t fence_value = 0;

        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 0;
        auto res = device->CreateCommandQueue(&desc, IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommandQueue failed.");

        res = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateFence failed.");

        res = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommandAllocator failed.");

        res = device->CreateCommandList(0,
                                        D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        command_allocator.Get(),
                                        nullptr,
                                        IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommandList failed.");

        
        // D3D12_RESOURCE_BARRIER barrier = {};
        // barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        // barrier.Transition.pResource = placed_resources.Get();
        // barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        // barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        // command_list->ResourceBarrier(1, &barrier);

        command_list->CopyBufferRegion(placed_resources.Get(), 0, comitted_resource.Get(), 0, byte_size);
        // command_list->CopyBufferRegion(readback_resource.Get(), 0, placed_resources.Get(), 0, byte_size);
        res = command_list->Close();
        RETURN_IF_FAILED(res, "Close command list failed.");

        ID3D12CommandList* command_lists[] = {command_list.Get()};
        command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);
        res = command_queue->Signal(fence.Get(), ++fence_value);
        RETURN_IF_FAILED(res, "Signal command queue failed.");

        volatile auto event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        res = fence->SetEventOnCompletion(fence_value, event);
        RETURN_IF_FAILED(res, "SetEventOnCompletion failed.");
        WaitForSingleObject(event, INFINITE);
        LITERT_LOG(LITERT_ERROR,    "======copy resources  success.");

        // ==============================================
      // Step 6: Read Back Data from Readback Buffer
      // ==============================================
      // Map readback buffer to host memory (read-only)
      // void* readbackHostPtr = nullptr;
      // res = readback_resource->Map(0, nullptr, &readbackHostPtr);
      // RETURN_IF_FAILED(res, "Failed to map Readback Buffer");

      // // Copy data from readback buffer to host vector
      // std::vector<float> hostReadbackData(byte_size / sizeof(float));
      // memcpy(hostReadbackData.data(), readbackHostPtr, byte_size);
      // for (int i = 0; i < 3; ++i) {
      //   LITERT_LOG(LITERT_ERROR,    "======copy resources  success %f.", hostReadbackData[i]);
      // }

      // // Unmap readback buffer (critical to avoid memory leaks)
      // readback_resource->Unmap(0, nullptr);
    }

    void readbackResources(const size_t byte_size) {
      D3D12_RESOURCE_DESC gpuBufferDesc = {};
      gpuBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      gpuBufferDesc.Width = byte_size;  // Buffer size in bytes
      gpuBufferDesc.Height = 1;
      gpuBufferDesc.DepthOrArraySize = 1;
      gpuBufferDesc.MipLevels = 1;
      gpuBufferDesc.Format = DXGI_FORMAT_UNKNOWN;  // Buffers have no format
      gpuBufferDesc.SampleDesc.Count = 1;
      gpuBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      
      D3D12_HEAP_PROPERTIES readbackHeapProps = {};
      readbackHeapProps.Type = D3D12_HEAP_TYPE_READBACK;  // Readback memory

      auto res = device->CreateCommittedResource(
          &readbackHeapProps,
          D3D12_HEAP_FLAG_NONE,
          &gpuBufferDesc,  // Same size as GPU buffer
          D3D12_RESOURCE_STATE_COPY_DEST,  // Initial state: ready to receive copy
          nullptr,
          IID_PPV_ARGS(&readback_resource)
      );
      RETURN_IF_FAILED(res, "Failed to create Readback Buffer.");

        Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> command_list;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        uint32_t fence_value = 0;

        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 0;
        res = device->CreateCommandQueue(&desc, IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommandQueue failed.");

        res = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateFence failed.");

        res = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommandAllocator failed.");

        res = device->CreateCommandList(0,
                                        D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        command_allocator.Get(),
                                        nullptr,
                                        IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf()));
        RETURN_IF_FAILED(res, "CreateCommandList failed.");


        command_list->CopyBufferRegion(readback_resource.Get(), 0, placed_resources.Get(), 0, byte_size);
        res = command_list->Close();
        RETURN_IF_FAILED(res, "Close command list failed.");

        ID3D12CommandList* command_lists[] = {command_list.Get()};
        command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);
        res = command_queue->Signal(fence.Get(), ++fence_value);
        RETURN_IF_FAILED(res, "Signal command queue failed.");

        volatile auto event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        res = fence->SetEventOnCompletion(fence_value, event);
        RETURN_IF_FAILED(res, "SetEventOnCompletion failed.");
        WaitForSingleObject(event, INFINITE);
        LITERT_LOG(LITERT_ERROR,    "======copy resources  success.");

        // ==============================================
      // Step 6: Read Back Data from Readback Buffer
      // ==============================================
      // Map readback buffer to host memory (read-only)
      void* readbackHostPtr = nullptr;
      res = readback_resource->Map(0, nullptr, &readbackHostPtr);
      RETURN_IF_FAILED(res, "Failed to map Readback Buffer");

      // Copy data from readback buffer to host vector
      std::vector<float> hostReadbackData(byte_size / sizeof(float));
      memcpy(hostReadbackData.data(), readbackHostPtr, byte_size);
      for (int i = 0; i < 3; ++i) {
        LITERT_LOG(LITERT_ERROR,    "======copy resources  success %f.", hostReadbackData[i]);
      }

      // Unmap readback buffer (critical to avoid memory leaks)
      readback_resource->Unmap(0, nullptr);
    }
};
// }
// #endif

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;

  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer type"));

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer size"));

  size_t tensor_buffer_offset;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer offset"));

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer's type"));
  LITERT_RETURN_IF_ERROR(
      !tensor_type.layout.has_strides,
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Tensor strides are not supported"));
  static DX12RemoteRun* dx12_tensor_buffer;
  if (!dx12_tensor_buffer) {
    dx12_tensor_buffer = new DX12RemoteRun();
    dx12_tensor_buffer->createAdapter();
  }
  dx12_tensor_buffer->createHeap(tensor_buffer_size);
  dx12_tensor_buffer->createResources(tensor_buffer_size);
  void* mem;
  dx12_tensor_buffer->comitted_resource.Get()->Map(0, nullptr, &mem);
  std::vector<float> hostInputData(tensor_buffer_size / sizeof(float), 7.0);  // Dummy input data
  // memset(mem, 15, tensor_buffer_size);
  memcpy(mem, hostInputData.data(), tensor_buffer_size);
  for (int i = 0; i < 3; ++i) {
    LITERT_LOG(LITERT_ERROR,    "======memset %f.", ((float*)mem)[i]);
  }
  dx12_tensor_buffer->comitted_resource.Get()->Unmap(0, nullptr);
  dx12_tensor_buffer->copyResources(tensor_buffer_size);
  dx12_tensor_buffer->readbackResources(tensor_buffer_size);

  ov::element::Type ov_element_type =
      litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeOpenClBuffer: {
      LITERT_LOG(LITERT_ERROR, "======111==kLiteRtTensorBufferTypeOpenClBuffer ");
      cl_mem cl_mem_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferOpenClMemory(tensor_buffer, &cl_mem_addr),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get cl_mem buffer"));

      auto context = core_->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
      for (int i = 0; i < ov_shape_vec.size(); i++)
        ov_shape_vec[i] = tensor_type.layout.dimensions[i];

      LITERT_LOG(LITERT_ERROR, "======111==create_tensor%p ", cl_mem_addr);
      auto remote_tensor = context.create_tensor(
          ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()}, dx12_tensor_buffer->shared_mem);
      // memcpy(remote_tensor.get(), buffer_host_addr, tensor_buffer_size);
      tensor_handle_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 remote_tensor);
      tensor_handle_buffer_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 tensor_buffer);

      LITERT_LOG(LITERT_ERROR, "====222====create_tensor%p ", cl_mem_addr);
      return next_handle_++;
    }
    case kLiteRtTensorBufferTypeDmaBuf: {
#if LITERT_HAS_DMABUF_SUPPORT
      int buffer_fd;
      void *buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferDmaBufBuffer(tensor_buffer, &buffer_host_addr,
                                            &buffer_fd),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get DMA-BUF buffer"));

      auto mmap_handle = mmap(NULL, tensor_buffer_size, PROT_WRITE | PROT_READ,
                              MAP_SHARED, buffer_fd, tensor_buffer_offset);

      if (mmap_handle == MAP_FAILED)
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "MMAP failed for tensor buffer");

      auto context = core_->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
      for (int i = 0; i < ov_shape_vec.size(); i++)
        ov_shape_vec[i] = tensor_type.layout.dimensions[i];

      // TODO: change f32 to ov_element_type fetched from TensorType
      auto remote_tensor = context.create_tensor(
          ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
          buffer_fd);
      tensor_handle_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 remote_tensor);
      return next_handle_++;

#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "DmaBuf support is missing on this platform");
#endif  // LRT_HAS_DMABUF_SUPPORT
      break;
    }

    case kLiteRtTensorBufferTypeAhwb: {
#if LITERT_HAS_AHWB_SUPPORT
      AHardwareBuffer *ahwb;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get LiteRT Tensor Buffer for AHWB"));

      auto fd_exp = GetFdFromUnixHandle(ahwb);
      int fd = fd_exp.Value();
      LITERT_RETURN_IF_ERROR(
          fd != -1, litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                       "Failed to get FD from unix handle"));

      std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
      for (int i = 0; i < ov_shape_vec.size(); i++)
        ov_shape_vec[i] = tensor_type.layout.dimensions[i];
      auto context = core_->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      void* buffer = mmap(nullptr, tensor_buffer_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, tensor_buffer_offset);
      ov::Tensor ov_tensor(ov_element_type,
                           ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
                           buffer);
      tensor_handle_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 ov_tensor);
      return next_handle_++;

#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "AHWB support is missing on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      break;
    }

    default:
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Unsupported tensor buffer type");
  }
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto it = tensor_handle_map_.find(tensor_buffer_handle);
  if (it != tensor_handle_map_.end()) {
    tensor_handle_map_.erase(tensor_buffer_handle);
  } else {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to Unregister Tensor Buffer");
  }

  auto tensor_buffer_it = tensor_handle_buffer_map_.find(tensor_buffer_handle);
  if (tensor_buffer_it != tensor_handle_buffer_map_.end()) {
    tensor_handle_buffer_map_.erase(tensor_buffer_handle);
  } else {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to Unregister Tensor Buffer");
  }

  return {};
}
