// Copyright 2024 Google LLC.
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

#include "litert/runtime/d3d12_memory.h"

#include <stdlib.h>

#include <cstddef>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/runtime/gpu_environment.h"


namespace litert::internal {

#define RETURN_IF_FAILED(res, error_message)                                 \
do {                                                             \
  if (res != S_OK) {                                              \
    LITERT_LOG(LITERT_ERROR,    error_message);                          \
    return;                                                   \
  }                                                              \
} while (0)

void D3D12Eenvironment::CreateAdapter() {
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

TFLiteGPUD3D12Buffer::TFLiteGPUD3D12Buffer(Microsoft::WRL::ComPtr<ID3D12Device9> other_device) {
  device = other_device;
}

void TFLiteGPUD3D12Buffer::CreateHeap(const size_t byte_size) {
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

void TFLiteGPUD3D12Buffer::CreatePlacedResources(const size_t byte_size) {
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

void TFLiteGPUD3D12Buffer::CreateComittedResources(const size_t byte_size) {
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

void TFLiteGPUD3D12Buffer::CreateResources(const size_t byte_size) {
    CreateHeap(byte_size);
    CreatePlacedResources(byte_size);
    CreateComittedResources(byte_size);
}

void TFLiteGPUD3D12Buffer::CopyResources(const size_t byte_size) {
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

void TFLiteGPUD3D12Buffer::ReadbackResources(const size_t byte_size, void* data) {
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
  // std::vector<float> hostReadbackData(byte_size / sizeof(float));
  memcpy(data, readbackHostPtr, byte_size);
  for (int i = 0; i < 3; ++i) {
    LITERT_LOG(LITERT_ERROR,    "======downaload resources  success %f.", ((float*)data)[i]);
  }

  // Unmap readback buffer (critical to avoid memory leaks)
  readback_resource->Unmap(0, nullptr);
}

template Expected<float*> D3D12Memory::Lock<float>(
    LiteRtTensorBufferLockMode mode);
template Expected<char*> D3D12Memory::Lock<char>(
    LiteRtTensorBufferLockMode mode);
template Expected<void> D3D12Memory::Unlock<float>();
template Expected<void> D3D12Memory::Unlock<char>();

template <typename T>
Expected<T*> D3D12Memory::Lock(LiteRtTensorBufferLockMode mode) {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "D3D12 is not supported"));
  LITERT_RETURN_IF_ERROR(lock_state_ == LockState::kUnlocked,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "The D3D12 memory is already locked."));
  bool lock_success = false;
  LockState lock_state = ToLockState(mode);
  absl::Cleanup lock_set = [this, &lock_success, &lock_state] {
    if (lock_success) {
      lock_state_ = lock_state;
    }
  };

  if (data_ == nullptr) {
    // The current Lock() always provides a packed buffer regardless of the
    // underlying H/W buffer type. If the underlying H/W buffer has a stride,
    // the data will be converted to the packed buffer by
    // LiteRtGpuMemoryDownload().
    // TODO b/413449050 - Update behavior to return raw H/W buffer and its size.
    LITERT_ASSIGN_OR_RETURN(cpu_buffer_size_,
                            litert::internal::GetNumPackedBytes(tensor_type_));
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                 cpu_buffer_size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
  }
  if (lock_state == LockState::kReadLocked ||
      lock_state == LockState::kReadWriteLocked) {
    // if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    //   LITERT_RETURN_IF_ERROR(gpu_env_->GetCommandQueue()
    //                              ->EnqueueReadBuffer(GetMemoryPtr(),
    //                                                  cpu_buffer_size_, data_,
    //                                                  /*async=*/false)
    //                              .ok());
    // } else {
    //   // Use the GPU Delegate API to download the data from the OpenCL buffer
    //   // to the aligned memory.
    //   LITERT_RETURN_IF_ERROR(
    //       LiteRtGpuMemoryDownload(gpu_env_, &tensor_type_, buffer_type_,
    //                               cpu_buffer_size_, GetMemoryPtr(), data_));
    // }

    // TODO::Download buffer with D3D12
    buffer_.ReadbackResources(cpu_buffer_size_, data_);
  }
  lock_success = true;
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> D3D12Memory::Unlock() {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "D3D12 is not supported"));
  LITERT_RETURN_IF_ERROR(lock_state_ != LockState::kUnlocked,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "The D3D12 memory is already unlocked."));
  absl::Cleanup unlock = [this] { lock_state_ = LockState::kUnlocked; };
  if (lock_state_ == LockState::kWriteLocked ||
      lock_state_ == LockState::kReadWriteLocked) {
    // if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    //   LITERT_RETURN_IF_ERROR(gpu_env_->GetCommandQueue()
    //                              ->EnqueueWriteBuffer(GetMemoryPtr(),
    //                                                   cpu_buffer_size_, data_,
    //                                                   /*async=*/true)
    //                              .ok());
    // } else {
    //   // The current Unlock() translates the packed buffer (data_) if the
    //   // underlying H/W buffer has a stride. This conversion is done by
    //   // LiteRtGpuMemoryUpload().
    //   // TODO b/413449050 - Update behavior to upload raw H/W buffer as it is.
    //   LITERT_RETURN_IF_ERROR(
    //       LiteRtGpuMemoryUpload(gpu_env_, &tensor_type_, buffer_type_,
    //                             cpu_buffer_size_, data_, GetMemoryPtr()));
    // }
    //TODO:: upload buffer with D3D12
    
    void* mem;
    buffer_.comitted_resource.Get()->Map(0, nullptr, &mem);
    // std::vector<float> hostInputData(bytes_size / sizeof(float), 7.0);  // Dummy input data
    memcpy(mem, data_, cpu_buffer_size_);
    for (int i = 0; i < 3; ++i) {
      LITERT_LOG(LITERT_ERROR,    "======memset %f.", ((float*)mem)[i]);
    }
    buffer_.comitted_resource.Get()->Unmap(0, nullptr);
    buffer_.CopyResources(cpu_buffer_size_);
  }
  return Expected<void>();
}

bool D3D12Memory::IsSupported() {
  // static bool is_supported = ::tflite::gpu::cl::LoadOpenCL().ok();
  // return is_supported;
  return true;
}

Expected<D3D12Memory> D3D12Memory::Alloc(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes_size) {
  static D3D12Eenvironment* d3d12_env;
  if (d3d12_env == nullptr) {
    d3d12_env = new D3D12Eenvironment();
    d3d12_env->CreateAdapter();
  }

  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "D3D12 is not supported"));
  // if (gpu_env == nullptr) {
  //   return Unexpected(kLiteRtStatusErrorRuntimeFailure,
  //                     "D3D12 is not supported");
  // }

  // if (buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
  //   tflite::gpu::cl::Buffer buffer;
  //   LITERT_RETURN_IF_ERROR(tflite::gpu::cl::CreateReadWriteBuffer(
  //                              bytes_size, gpu_env->GetContext(), &buffer)
  //                              .ok());
  //   return Expected<OpenClMemory>(gpu_env, tensor_type, buffer_type,
  //                                 std::move(buffer));
  // }

  // cl_mem cl_memory;
  // LITERT_RETURN_IF_ERROR(LiteRtGpuMemoryCreate(
  //     gpu_env, &tensor_type, buffer_type, bytes_size, &cl_memory));

  // TFLiteGPUD3D12Buffer buffer(cl_memory, bytes_size);
  TFLiteGPUD3D12Buffer buffer(d3d12_env->device);
  buffer.CreateHeap(bytes_size);
  buffer.CreateResources(bytes_size);

  return Expected<D3D12Memory>(gpu_env, tensor_type, buffer_type,
                                std::move(buffer));
}

}  // namespace litert::internal

