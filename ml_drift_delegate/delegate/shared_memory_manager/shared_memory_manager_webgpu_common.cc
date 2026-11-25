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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu_common.h"

#include <memory>

#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"

namespace ml_drift {
namespace webgpu_internal {
namespace {

#ifndef __EMSCRIPTEN__
#ifdef _WIN32
// Dawn requires larger alignment on Windows than other platforms.
constexpr size_t kAlignment = 65536;
#else
constexpr size_t kAlignment = 4096;
#endif

// Copies `data` into a wgpu::Buffer using the BufferHostMappedPointer extension
// which can create a buffer directly from a memory region. This allows for fast
// uploading of data on a thread because it does not require acquiring the
// internal WebGPU lock.
//
// If `row_size` is not zero, `data` must start with `num_rows` tightly packed
// rows of `row_size` bytes, and row i is written at byte i * `dst_row_pitch` of
// the buffer, see CopyRowsWithPitch().
absl::StatusOr<wgpu::Buffer> CreateOwnedDataBuffer(
    const wgpu::Device& device, absl::Span<const uint8_t> data, size_t min_size,
    size_t row_size = 0, size_t dst_row_pitch = 0, size_t num_rows = 0) {
  size_t size = std::max(data.size(), min_size);
  // BufferHostMappedPointer requires aligned buffers.
  size_t aligned_size = AlignByN(size, kAlignment);
#if defined(_WIN32)
  uint8_t* data_copy = static_cast<uint8_t*>(VirtualAlloc(
      nullptr, aligned_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
  void* data_copy_void_ptr;
  ABSL_CHECK_EQ(posix_memalign(  // NOLINT: posix_memalign is in stdlib
                    &data_copy_void_ptr, kAlignment, aligned_size),
                0);
  uint8_t* data_copy = static_cast<uint8_t*>(data_copy_void_ptr);
#endif
  auto free_data_copy = [](void* userdata) {
#if defined(_WIN32)
    VirtualFree(userdata, 0, MEM_RELEASE);
#else
    free(userdata);
#endif
  };
  if (row_size != 0) {
    absl::Status status =
        CopyRowsWithPitch(data, row_size, num_rows, dst_row_pitch,
                          absl::MakeSpan(data_copy, aligned_size));
    if (!status.ok()) {
      free_data_copy(data_copy);
      return status;
    }
  } else {
    memcpy(data_copy, data.data(), data.size());
  }
  wgpu::BufferHostMappedPointer host_mapped_desc;
  host_mapped_desc.pointer = data_copy;
  host_mapped_desc.disposeCallback = free_data_copy;
  host_mapped_desc.userdata = data_copy;
  wgpu::BufferDescriptor buffer_desc = {
      .nextInChain = &host_mapped_desc,
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = aligned_size,
  };
  return device.CreateBuffer(&buffer_desc);
}

// This is a simplified version of the logic from
// LlmWebGpu::CreateModelTensor().
absl::Status UploadPrepackedTensor(const webgpu::ExecutionEnvironment& env,
                                   TensorDescriptor& desc, Executor* executor,
                                   bool has_prepacked_tflite_tensors,
                                   size_t page_adjusted_offset,
                                   webgpu::NotifiedTensor* tensor) {
  (void)page_adjusted_offset;
  wgpu::Device device = env.device();
  // Tick the device to allow freeing any staging buffers.
  device.Tick();
  TensorDescriptor desc_no_data;
  desc.CopyWithoutData(&desc_no_data);
  ABSL_RETURN_IF_ERROR(
      tensor->CreateFromDescriptor(env.device(), desc_no_data));

  absl::Span<const uint8_t> weights_data = desc.GetData();
  wgpu::Texture dst_texture = tensor->GetTextureHandle();
  wgpu::Buffer dst_buffer = tensor->GetBufferHandle();
  wgpu::Queue queue = env.queue();
  wgpu::CommandEncoder encoder = env.device().CreateCommandEncoder();
  uint32_t offset = 0;

  // CopyBufferToTexture reads the rows at a 256-byte aligned bytesPerRow, so
  // tightly packed rows are repacked at that pitch in the staging buffer, see
  // GetTextureStagingLayout().
  const size_t pixel_size = SizeOf(desc.GetDataType()) * 4;
  const int3 region = desc.GetFullTensorRegion();
  TextureStagingLayout layout;
  if (dst_texture) {
    ABSL_ASSIGN_OR_RETURN(layout, GetTextureStagingLayout(region, pixel_size,
                                                          weights_data.size()));
  }
  const size_t min_owned_buffer_size = layout.min_buffer_size;
  const size_t row_size = layout.repack_rows ? layout.bytes_per_row : 0;
  const uint32_t bytes_per_row_aligned = layout.bytes_per_row_aligned;
  const size_t num_rows = layout.num_rows;
  const uint32_t rows_per_image = region.y;

  wgpu::Buffer buffer;
  // If the buffer is prepacked, it is likely mmapped and it will still be
  // available for reading when the thread is run. If the buffer was serialized,
  // it is not mmapped and will not survive until the upload thread is run so we
  // need to copy it to a buffer on the main thread.
  bool copy_buffer_on_main_thread = !has_prepacked_tflite_tensors;
  if (copy_buffer_on_main_thread) {
    ABSL_ASSIGN_OR_RETURN(
        buffer,
        CreateOwnedDataBuffer(device, weights_data, min_owned_buffer_size,
                              row_size, bytes_per_row_aligned, num_rows));
  }

  auto upload_notification = std::make_shared<absl::Notification>();
  tensor->set_upload_notification(upload_notification);

  auto fn = [=] {
    wgpu::Buffer owned_buffer;
    if (copy_buffer_on_main_thread) {
      owned_buffer = std::move(buffer);
    } else {
      // The layout was validated above, so this only fails if the arguments
      // are inconsistent.
      absl::StatusOr<wgpu::Buffer> created_buffer =
          CreateOwnedDataBuffer(device, weights_data, min_owned_buffer_size,
                                row_size, bytes_per_row_aligned, num_rows);
      if (!created_buffer.ok()) {
        ABSL_LOG(ERROR) << "Failed to stage weights for upload: "
                        << created_buffer.status();
        upload_notification->Notify();
        return;
      }
      owned_buffer = *std::move(created_buffer);
      ::ml_drift::MadviseData(const_cast<uint8_t*>(weights_data.data()),
                              weights_data.size());
    }
    if (dst_texture) {
      wgpu::Extent3D copy_size = {
          .width = static_cast<uint32_t>(region.x),
          .height = static_cast<uint32_t>(region.y),
          .depthOrArrayLayers = static_cast<uint32_t>(region.z),
      };
      wgpu::TexelCopyTextureInfo copy_texture = {
          .texture = dst_texture,
      };
      wgpu::TexelCopyBufferInfo copy_buffer = {
          .layout =
              {
                  .offset = offset,
                  .bytesPerRow = bytes_per_row_aligned,
                  .rowsPerImage = rows_per_image,
              },
          .buffer = owned_buffer,
      };

      encoder.CopyBufferToTexture(&copy_buffer, &copy_texture, &copy_size);
    } else {
      encoder.CopyBufferToBuffer(owned_buffer, 0, dst_buffer, 0,
                                 weights_data.size());
    }

    auto cb = encoder.Finish();
    queue.Submit(1, &cb);
    upload_notification->Notify();
    device.Tick();
  };
  if (executor) {
    executor->Schedule(std::move(fn));
  } else {
    fn();
  }
  device.Tick();
  return absl::OkStatus();
}
#endif  // !__EMSCRIPTEN__

}  // namespace

absl::Status CopyRowsWithPitch(absl::Span<const uint8_t> src, size_t row_size,
                               size_t num_rows, size_t dst_row_pitch,
                               absl::Span<uint8_t> dst) {
  if (row_size != 0 &&
      num_rows > (std::numeric_limits<size_t>::max)() / row_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        num_rows, " rows of ", row_size, " bytes overflow size_t."));
  }
  if (src.size() < num_rows * row_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected at least ", num_rows, " rows of ", row_size,
                     " bytes, got ", src.size(), " bytes."));
  }
  if (dst_row_pitch < row_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Row pitch ", dst_row_pitch,
                     " is smaller than the row size ", row_size, "."));
  }
  if (num_rows == 0 || row_size == 0) {
    return absl::OkStatus();
  }
  // Here dst_row_pitch >= row_size > 0. Written without multiplying so that it
  // cannot overflow.
  if (dst.size() < row_size ||
      (dst.size() - row_size) / dst_row_pitch < num_rows - 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Destination of ", dst.size(), " bytes is too small for ",
                     num_rows, " rows with a pitch of ", dst_row_pitch, "."));
  }
  if (dst_row_pitch == row_size) {
    std::copy_n(src.data(), num_rows * row_size, dst.data());
    return absl::OkStatus();
  }
  for (size_t row = 0; row < num_rows; ++row) {
    memcpy(dst.data() + row * dst_row_pitch, src.data() + row * row_size,
           row_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<TextureStagingLayout> GetTextureStagingLayout(const int3& region,
                                                             size_t texel_size,
                                                             size_t data_size) {
  if (region.x <= 0 || region.y <= 0 || region.z <= 0 || texel_size == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid texture region ", region.x, "x", region.y, "x",
                     region.z, " with texels of ", texel_size, " bytes."));
  }
  // Keep bytes_per_row_aligned representable as uint32_t.
  constexpr uint64_t kMaxBytesPerRow =
      (std::numeric_limits<uint32_t>::max)() - 255;
  const uint64_t bytes_per_row = static_cast<uint64_t>(texel_size) * region.x;
  if (texel_size > kMaxBytesPerRow || bytes_per_row > kMaxBytesPerRow) {
    return absl::InvalidArgumentError(absl::StrCat("Texture row of ", region.x,
                                                   " texels of ", texel_size,
                                                   " bytes is too large."));
  }
  TextureStagingLayout layout;
  layout.bytes_per_row = static_cast<uint32_t>(bytes_per_row);
  // From the spec: "texelCopyBufferInfo.bytesPerRow must be a multiple of
  // 256."
  layout.bytes_per_row_aligned = AlignByN(layout.bytes_per_row, 256);
  const uint64_t num_rows = static_cast<uint64_t>(region.y) * region.z;
  // min_buffer_size = (num_rows - 1) * bytes_per_row_aligned + bytes_per_row,
  // which must fit in size_t.
  if ((num_rows - 1) >
      ((std::numeric_limits<size_t>::max)() - layout.bytes_per_row) /
          layout.bytes_per_row_aligned) {
    return absl::InvalidArgumentError(
        absl::StrCat("Texture of ", num_rows, " rows of ",
                     layout.bytes_per_row_aligned, " bytes is too large."));
  }
  layout.num_rows = static_cast<size_t>(num_rows);
  layout.min_buffer_size =
      (layout.num_rows - 1) * layout.bytes_per_row_aligned +
      layout.bytes_per_row;
  if (layout.bytes_per_row == layout.bytes_per_row_aligned) {
    // The rows are already at the aligned pitch: copied as is.
    return layout;
  }
  // Data that is already padded to the aligned pitch, with or without the
  // padding of the last row, is copied as is. Repacking it would scramble it.
  const size_t last_row_padding =
      layout.bytes_per_row_aligned - layout.bytes_per_row;
  const bool is_padded =
      data_size == layout.min_buffer_size ||
      (data_size >= last_row_padding &&
       data_size - last_row_padding == layout.min_buffer_size);
  if (is_padded) {
    return layout;
  }
  // Otherwise the rows are tightly packed, possibly followed by bytes that are
  // not uploaded. num_rows * bytes_per_row <= min_buffer_size, so this does not
  // overflow.
  if (data_size < layout.num_rows * layout.bytes_per_row) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Texture data is too small: expected at least ", layout.num_rows, " x ",
        layout.bytes_per_row, " bytes, got ", data_size, "."));
  }
  layout.repack_rows = true;
  return layout;
}

#if !defined(__EMSCRIPTEN__)
// Copies `desc` into a wgpu::Buffer using the BufferHostMappedPointer extension
// which can create a buffer directly from a memory region. This allows for fast
// uploading of data because it does not require creating extra copies of the
// data.
absl::Status CopyBufferToBuffer(
    const webgpu::ExecutionEnvironment* env, const TensorDescriptor& desc,
    size_t page_adjusted_offset,
    ::litert::ml_drift::ReleaseDataCallback release_data_callback,
    webgpu::SpatialTensor* tensor) {
  // data points to the start of the buffer data.
  const uint8_t* data = desc.GetData().data();
  // size is the size of the buffer data.
  size_t size = desc.GetData().size();
  // offset is the offset of the buffer data within the mmap region.
  const size_t offset = page_adjusted_offset;
  // raw_data points to the start of the mmap region.
  const uint8_t* raw_data = data - offset;
  // raw_size is the size of the entire mmap region not just the buffer data.
  // BufferHostMappedPointer requires the wrapped size to be aligned, so round
  // up. This is safe because the data is mmap'd, and mmap maps whole pages.
  size_t raw_size = AlignByN(size + offset, kAlignment);

  // If an integrated GPU is being used, the host mapped buffer can be
  // directly used instead of copying to individual GPU buffers.
  const bool is_integrated_gpu = env->GetInfo().webgpu_info.is_integrated_gpu;
  wgpu::BufferHostMappedPointer host_mapped_desc;
  host_mapped_desc.pointer = const_cast<uint8_t*>(raw_data);

  // The disposeCallback is called after the buffer is no longer in use. Pass it
  // the release_data_callback which will release the mmap'd memory.
  host_mapped_desc.disposeCallback = [](void* user_data) {
    // The user_data is the real release_data_callback. This "trampoline"
    // approach is beneficial because it allows the original std::function
    // to capture several variables (data ptr, size, offset, etc) which are not
    // possible with a simple function callback.
    auto real_callback = static_cast<std::function<void()>*>(user_data);
    if (!real_callback) {
      return;
    }
    (*real_callback)();
    delete real_callback;
  };
  host_mapped_desc.userdata = release_data_callback.release();

  wgpu::BufferDescriptor buffer_desc = {
      .nextInChain = &host_mapped_desc,
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = raw_size,
  };
  if (is_integrated_gpu) {
    buffer_desc.usage = wgpu::BufferUsage::CopyDst |
                        wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage;
  }

  wgpu::Buffer owned_data_buffer = env->device().CreateBuffer(&buffer_desc);

  TensorDescriptor desc_no_data;
  desc.CopyWithoutData(&desc_no_data);
  if (owned_data_buffer.GetUsage() & wgpu::BufferUsage::Storage) {
    ABSL_RETURN_IF_ERROR(
        tensor->CreateFromBuffer(owned_data_buffer, offset, desc_no_data));
  } else {
    ABSL_RETURN_IF_ERROR(
        tensor->CreateFromDescriptor(env->device(), desc_no_data));
    wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
    encoder.CopyBufferToBuffer(owned_data_buffer, offset,
                               tensor->GetBufferHandle(), 0,
                               desc.GetData().size());
    wgpu::CommandBuffer cb = encoder.Finish();
    env->queue().Submit(1, &cb);
  }
  return absl::OkStatus();
}
#endif  // !__EMSCRIPTEN__

namespace {

// Implements CreateSharedWebGpuTensor(). `tensor` must be null. May leave a
// partially created tensor in `tensor` on failure.
absl::Status CreateSharedWebGpuTensorImpl(
    const webgpu::ExecutionEnvironment& env, TensorDescriptor& tensor_desc,
    size_t page_adjusted_offset,
    ::litert::ml_drift::ReleaseDataCallback release_data_callback,
    bool has_prepacked_tflite_tensors, Executor* upload_executor,
    UploadScheduling upload_scheduling,
    std::unique_ptr<GpuSpatialTensor>& tensor) {
#if defined(__EMSCRIPTEN__)
  tensor = std::make_unique<webgpu::SpatialTensor>();
#else  // defined(__EMSCRIPTEN__)
  tensor = std::make_unique<webgpu::NotifiedTensor>();

  // TODO: b/423950292 - Support other data types.
  bool is_supported_texture =
      tensor_desc.GetStorageType() == TensorStorageType::kTexture2D &&
      tensor_desc.GetDataType() == DataType::kUint16;
  bool is_quantized_buffer =
      tensor_desc.GetStorageType() == TensorStorageType::kBuffer &&
      (tensor_desc.GetDataType() == DataType::kUint2 ||
       tensor_desc.GetDataType() == DataType::kUint4 ||
       tensor_desc.GetDataType() == DataType::kUint8);
  const bool has_host_mapped_pointer =
      env.device().HasFeature(wgpu::FeatureName::HostMappedPointer);

#if defined(__APPLE__)
  // The Metal API imports host pointers from mmap'd data quickly, so no
  // need to copy the data. Vulkan and D3D12 both import memory much
  // more quickly from non-mmap'd memory, so in those APIs it's faster
  // to copy all data before importing.
  //
  // Avoid the copy on Apple if the buffer type is BUFFER and there is a
  // valid release_data_callback that can be used to release the mmap'd
  // memory. Besides the quantized weights, this covers float buffers such as
  // the moe_experts block scales restored from the weight cache. In the
  // future, we should add support for no-copy textures as well.
  const bool is_float_buffer =
      tensor_desc.GetStorageType() == TensorStorageType::kBuffer &&
      (tensor_desc.GetDataType() == DataType::kFloat16 ||
       tensor_desc.GetDataType() == DataType::kFloat32);
  if ((is_quantized_buffer || is_float_buffer) && has_host_mapped_pointer &&
      release_data_callback) {
    return ::ml_drift::webgpu_internal::CopyBufferToBuffer(
        &env, tensor_desc, page_adjusted_offset,
        std::move(release_data_callback),
        static_cast<webgpu::SpatialTensor*>(tensor.get()));
  }
#endif  // defined(__APPLE__)

  if ((is_quantized_buffer || is_supported_texture) &&
      has_host_mapped_pointer) {
    if (release_data_callback) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Release data callback is not currently supported on "
          "non-Apple devices or on Apple devices with storage type: ",
          ToString(tensor_desc.GetStorageType())));
    }
    const bool allow_inline =
        upload_scheduling == UploadScheduling::kAllowInline;
    if (upload_executor || allow_inline) {
      return UploadPrepackedTensor(
          env, tensor_desc, upload_executor, has_prepacked_tflite_tensors,
          page_adjusted_offset,
          static_cast<webgpu::NotifiedTensor*>(tensor.get()));
    }
  }
#endif  // defined(__EMSCRIPTEN__)
  auto status = static_cast<webgpu::SpatialTensor*>(tensor.get())
                    ->CreateFromDescriptor(env.device(), tensor_desc);
#if !defined(__EMSCRIPTEN__)
  env.device().Tick();
#endif  // !defined(__EMSCRIPTEN__)
  return status;
}

}  // namespace

absl::Status CreateSharedWebGpuTensor(
    const webgpu::ExecutionEnvironment& env, TensorDescriptor& tensor_desc,
    size_t page_adjusted_offset,
    ::litert::ml_drift::ReleaseDataCallback release_data_callback,
    bool has_prepacked_tflite_tensors, Executor* upload_executor,
    UploadScheduling upload_scheduling,
    std::unique_ptr<GpuSpatialTensor>& tensor) {
  if (tensor) {
    return absl::InternalError("Tensor is already initialized.");
  }
  absl::Status status = CreateSharedWebGpuTensorImpl(
      env, tensor_desc, page_adjusted_offset, std::move(release_data_callback),
      has_prepacked_tflite_tensors, upload_executor, upload_scheduling, tensor);
  if (!status.ok()) {
    // Release the partially created tensor so that the caller can create it
    // again, e.g. SharedMemoryManager falls back to converting the original
    // weights when the weight cache data is invalid.
    tensor.reset();
  }
  return status;
}

}  // namespace webgpu_internal
}  // namespace ml_drift
