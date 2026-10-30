/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/ml_drift/ml_drift_buffer.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_memory.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/cl/util.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace litert::tensor {

namespace {

template <ml_drift::DataType DT>
absl::Status ReadAndCopyTensor(ml_drift::cl::InferenceContext* context,
                               ml_drift::ValueId value_id,
                               ml_drift::cl::CLCommandQueue* queue,
                               size_t bytes_size, std::byte* dest) {
  auto* gpu_tensor = context->GetTensor(value_id);
  if (!gpu_tensor) {
    return absl::NotFoundError("Tensor not found in context");
  }

  ml_drift::Tensor<ml_drift::BHWC, DT> output_tensor;
  const auto dst_shape =
      ml_drift::BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                     gpu_tensor->Width(), gpu_tensor->Channels());
  output_tensor.id = value_id;
  output_tensor.shape = dst_shape;

  ml_drift::TensorDescriptor desc;
  auto status = gpu_tensor->ToDescriptor(&desc, queue);
  if (!status.ok()) {
    return status;
  }
  using T = typename ml_drift::Tensor<ml_drift::BHWC, DT>::ValueType;
  output_tensor.data.resize(output_tensor.shape.DimensionsProduct(), T(0.0f));
  desc.DownloadData(output_tensor.data.data());

  assert(output_tensor.size() * sizeof(T) <= bytes_size);
  const void* src_ptr = nullptr;
  if constexpr (DT == ml_drift::DataType::kBool) {
    src_ptr = !output_tensor.data.empty() ? output_tensor.data.data()
                                          : output_tensor.spanned_data.data();
  } else {
    src_ptr = output_tensor.Data();
  }
  std::memcpy(dest, src_ptr, output_tensor.size() * sizeof(T));
  return absl::OkStatus();
}

template <ml_drift::DataType DT>
absl::Status CopyAndSetTensor(ml_drift::cl::InferenceContext* context,
                              ml_drift::ValueId value_id,
                              ml_drift::cl::CLCommandQueue* queue,
                              ml_drift::BHWC shape, size_t bytes_size,
                              const std::byte* src) {
  auto* gpu_tensor = context->GetTensor(value_id);
  if (!gpu_tensor) {
    return absl::NotFoundError("Tensor not found in context");
  }

  ml_drift::Tensor<ml_drift::BHWC, DT> t;
  t.id = value_id;
  t.shape = shape;
  using T = typename ml_drift::Tensor<ml_drift::BHWC, DT>::ValueType;
  t.data.resize(bytes_size / sizeof(T));
  assert(shape.DimensionsProduct() * sizeof(T) <= bytes_size);
  std::memcpy(t.data.data(), src, t.data.size() * sizeof(T));

  ml_drift::TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(t);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, queue);
}

}  // namespace

absl::StatusOr<std::unique_ptr<MlDriftBuffer>> MlDriftBuffer::Create(
    ml_drift::ValueId value_id, ml_drift::cl::InferenceContext& context,
    ml_drift::cl::CLCommandQueue& queue, size_t size_bytes) {
  return std::make_unique<MlDriftBuffer>(value_id, context, queue, size_bytes);
}

absl::StatusOr<std::unique_ptr<MlDriftBuffer>> MlDriftBuffer::Create(
    ml_drift::cl::CLContext& context, ml_drift::cl::CLCommandQueue& queue,
    size_t size_bytes) {
  ml_drift::TensorDescriptor desc;
  desc.SetDataType(ml_drift::DataType::kUint8);
  desc.SetStorageType(ml_drift::TensorStorageType::kBuffer);
  desc.SetLayout(ml_drift::Layout::kLinear);
  desc.SetBHWCShape(ml_drift::BHWC(1, 1, 1, size_bytes));

  ml_drift::cl::CLMemory cl_memory;
  if (auto status =
          ml_drift::cl::AllocateTensorMemory(context, desc, &cl_memory);
      !status.ok()) {
    return status;
  }
  return std::make_unique<MlDriftBuffer>(std::move(cl_memory), queue,
                                         size_bytes);
}

MlDriftBuffer::MlDriftBuffer(ml_drift::ValueId value_id,
                             ml_drift::cl::InferenceContext& context,
                             ml_drift::cl::CLCommandQueue& queue,
                             size_t size_bytes)
    : value_id_(value_id),
      context_(&context),
      queue_(&queue),
      size_bytes_(size_bytes) {}

MlDriftBuffer::MlDriftBuffer(ml_drift::cl::CLMemory memory,
                             ml_drift::cl::CLCommandQueue& queue,
                             size_t size_bytes)
    : queue_(&queue), size_bytes_(size_bytes) {
  cl_memory_ = std::move(memory);
}

cl_mem MlDriftBuffer::GetMemory() const {
  if (cl_memory_.is_valid()) {
    return cl_memory_.memory();
  }
  if (context_) {
    // If attached to a context, get the memory from the tensor.
    // Note: InferenceContext::GetTensor returns a pointer to internal tensor
    // structure.
    auto* tensor = context_->GetTensor(value_id_);
    if (tensor) {
      return tensor->GetMemoryPtr();
    }
  }
  return nullptr;
}

cl_mem MlDriftBuffer::GetMemoryForWriting() const {
  if (cl_memory_.is_valid()) {
    return cl_memory_.memory();
  }
  if (context_) {
    auto* tensor = context_->GetTensor(value_id_);
    if (tensor) {
      return tensor->GetMemoryPtrForWriting();
    }
  }
  return nullptr;
}

LockedBufferSpan<const std::byte> MlDriftBuffer::Lock() {
  if (!host_buffer_) {
    host_buffer_ = OwningCpuBuffer::Allocate<Type::kU8>(size_bytes_);

    LockedBufferSpan<std::byte> host_lock = host_buffer_->LockMutable();
    std::byte* host_data = host_lock.data();
    if (cl_memory_.is_valid()) {
      // Read from standalone memory.
      auto status = queue_->EnqueueReadBuffer(cl_memory_.memory(), size_bytes_,
                                              host_data);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to read data from MlDriftBuffer: " << status;
        return LockedBufferSpan<const std::byte>::Empty();
      }
    } else if (context_) {
      auto* tensor = context_->GetTensor(value_id_);
      if (tensor) {
        ml_drift::TensorDescriptor desc = tensor->GetDescriptor();
        absl::Status status;
        switch (desc.GetDataType()) {
          case ml_drift::DataType::kFloat32:
            status = ReadAndCopyTensor<ml_drift::DataType::kFloat32>(
                context_, value_id_, queue_, size_bytes_, host_data);
            break;
          case ml_drift::DataType::kFloat16:
            status = ReadAndCopyTensor<ml_drift::DataType::kFloat16>(
                context_, value_id_, queue_, size_bytes_, host_data);
            break;
          case ml_drift::DataType::kInt32:
            status = ReadAndCopyTensor<ml_drift::DataType::kInt32>(
                context_, value_id_, queue_, size_bytes_, host_data);
            break;
          case ml_drift::DataType::kBool:
            status = ReadAndCopyTensor<ml_drift::DataType::kBool>(
                context_, value_id_, queue_, size_bytes_, host_data);
            break;
          default:
            // Unsupported data type for reading.
            status = absl::UnimplementedError(
                "Unsupported data type in MlDriftBuffer::Lock");
            break;
        }
        if (!status.ok()) {
          ABSL_LOG(ERROR) << "Failed to read data from MlDriftBuffer: "
                          << status;
          return LockedBufferSpan<const std::byte>::Empty();
        }
      }
    }
  }
  return host_buffer_->Lock();
}

LockedBufferSpan<std::byte> MlDriftBuffer::LockMutable() {
  if (!host_buffer_) {
    host_buffer_ = OwningCpuBuffer::Allocate<Type::kU8>(size_bytes_);
  }

  LockedBufferSpan<std::byte> host_lock = host_buffer_->LockMutable();
  return {host_lock.data(),
          [this](std::byte* data) {
            absl::Status last_upload_status = absl::OkStatus();
            if (cl_memory_.is_valid()) {
              // Upload back to GPU
              last_upload_status = queue_->EnqueueWriteBuffer(
                  cl_memory_.memory(), size_bytes_, data);
            } else if (context_) {
              auto* tensor = context_->GetTensor(value_id_);
              if (tensor) {
                ml_drift::TensorDescriptor desc = tensor->GetDescriptor();
                switch (desc.GetDataType()) {
                  case ml_drift::DataType::kFloat32:
                    last_upload_status =
                        CopyAndSetTensor<ml_drift::DataType::kFloat32>(
                            context_, value_id_, queue_, desc.GetBHWCShape(),
                            size_bytes_, data);
                    break;
                  case ml_drift::DataType::kFloat16:
                    last_upload_status =
                        CopyAndSetTensor<ml_drift::DataType::kFloat16>(
                            context_, value_id_, queue_, desc.GetBHWCShape(),
                            size_bytes_, data);
                    break;
                  case ml_drift::DataType::kInt32:
                    last_upload_status =
                        CopyAndSetTensor<ml_drift::DataType::kInt32>(
                            context_, value_id_, queue_, desc.GetBHWCShape(),
                            size_bytes_, data);
                    break;
                  case ml_drift::DataType::kBool:
                    last_upload_status =
                        CopyAndSetTensor<ml_drift::DataType::kBool>(
                            context_, value_id_, queue_, desc.GetBHWCShape(),
                            size_bytes_, data);
                    break;
                  default:
                    last_upload_status = absl::UnimplementedError(
                        "Unsupported data type for writing in MlDriftBuffer");
                    break;
                }
              }
            }
            if (!last_upload_status.ok()) {
              ABSL_LOG(ERROR) << "Failed to upload data to MlDriftBuffer: "
                              << last_upload_status;
            }
          },
          size_bytes_};
}

absl::Status MlDriftBuffer::WriteDataAsync(const void* ptr, size_t offset,
                                           size_t size_bytes) {
  cl_mem mem = GetMemoryForWriting();
  if (!mem) {
    return absl::InternalError("Buffer memory is null");
  }
  if (!queue_) {
    return absl::InternalError("Queue is null");
  }
  if (offset + size_bytes > size_bytes_) {
    return absl::InvalidArgumentError("Write out of bounds");
  }
  cl_int error_code = ml_drift::cl::clEnqueueWriteBuffer(
      queue_->queue(), mem, /*blocking_write=*/CL_TRUE, offset, size_bytes, ptr,
      0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    std::string error_msg = "Failed to write buffer async - ";
    absl::StrAppend(&error_msg, ml_drift::cl::CLErrorCodeToString(error_code));
    return absl::UnknownError(error_msg);
  }
  Invalidate();
  return absl::OkStatus();
}

absl::Status MlDriftBuffer::CopyFromAsync(const MlDriftBuffer& src,
                                          size_t src_offset, size_t dst_offset,
                                          size_t size_bytes) {
  cl_mem dst_mem = GetMemoryForWriting();
  cl_mem src_mem = src.GetMemory();
  if (!dst_mem || !src_mem) {
    return absl::InternalError("Source or destination memory is null");
  }
  if (!queue_) {
    return absl::InternalError("Queue is null");
  }

  ml_drift::TensorStorageType src_storage =
      ml_drift::TensorStorageType::kBuffer;
  ml_drift::int3 src_region;
  if (src.context()) {
    auto* tensor = src.context()->GetTensor(src.value_id());
    if (tensor) {
      src_storage = tensor->GetStorageType();
      src_region = tensor->GetDescriptor().GetFullTensorRegion();
    }
  }

  ml_drift::TensorStorageType dst_storage =
      ml_drift::TensorStorageType::kBuffer;
  ml_drift::int3 dst_region;
  if (context_) {
    auto* tensor = context_->GetTensor(value_id_);
    if (tensor) {
      dst_storage = tensor->GetStorageType();
      dst_region = tensor->GetDescriptor().GetFullTensorRegion();
    }
  }

  if (src_storage == ml_drift::TensorStorageType::kBuffer &&
      dst_storage == ml_drift::TensorStorageType::kBuffer) {
    if (dst_offset + size_bytes > size_bytes_) {
      std::string error_msg = "Copy out of bounds: dst_offset=";
      absl::StrAppend(&error_msg, dst_offset, " size_bytes=", size_bytes,
                      " size_bytes_=", size_bytes_);
      return absl::InvalidArgumentError(error_msg);
    }
    if (src_offset + size_bytes > src.GetSizeBytes()) {
      std::string error_msg = "Copy out of bounds: src_offset=";
      absl::StrAppend(&error_msg, src_offset, " size_bytes=", size_bytes,
                      " src.GetSizeBytes()=", src.GetSizeBytes());
      return absl::InvalidArgumentError(error_msg);
    }
    cl_int error_code = ml_drift::cl::clEnqueueCopyBuffer(
        queue_->queue(), src_mem, dst_mem, src_offset, dst_offset, size_bytes,
        0, nullptr, nullptr);
    if (error_code != CL_SUCCESS) {
      std::string error_msg = "Failed to copy buffer async - ";
      absl::StrAppend(&error_msg,
                      ml_drift::cl::CLErrorCodeToString(error_code));
      return absl::UnknownError(error_msg);
    }
  } else if (src_storage != ml_drift::TensorStorageType::kBuffer &&
             dst_storage == ml_drift::TensorStorageType::kBuffer) {
    if (src_offset != 0) {
      return absl::UnimplementedError(
          "Non-zero src_offset for image-to-buffer copy is not supported");
    }
    size_t element_size = 0;
    cl_int err = ml_drift::cl::clGetImageInfo(src_mem, CL_IMAGE_ELEMENT_SIZE,
                                              sizeof(element_size),
                                              &element_size, nullptr);
    if (err != CL_SUCCESS) {
      std::string error_msg = "Failed to get image element size - ";
      absl::StrAppend(&error_msg, ml_drift::cl::CLErrorCodeToString(err));
      return absl::InternalError(error_msg);
    }

    size_t src_origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(src_region.x),
                        static_cast<size_t>(src_region.y),
                        static_cast<size_t>(src_region.z)};

    // Check for potential overflow
    size_t copy_pixels = region[0] * region[1] * region[2];
    size_t copy_bytes = copy_pixels * (element_size ? element_size : 4);
    if (dst_offset + copy_bytes > size_bytes_) {
      ABSL_LOG(WARNING)
          << "CopyFromAsync: Potential overflow! dst_offset="
          << dst_offset << " copy_bytes=" << copy_bytes
          << " dst size_bytes_=" << size_bytes_;
    }

    cl_int error_code = ml_drift::cl::clEnqueueCopyImageToBuffer(
        queue_->queue(), src_mem, dst_mem, src_origin, region, dst_offset, 0,
        nullptr, nullptr);
    if (error_code != CL_SUCCESS) {
      std::string error_msg = "Failed to copy image to buffer async - ";
      absl::StrAppend(&error_msg,
                      ml_drift::cl::CLErrorCodeToString(error_code));
      return absl::UnknownError(error_msg);
    }
  } else {
    std::string error_msg = "Copy from storage type ";
    absl::StrAppend(&error_msg, static_cast<int>(src_storage), " to ",
                    static_cast<int>(dst_storage), " is not supported");
    return absl::UnimplementedError(error_msg);
  }

  Invalidate();
  return absl::OkStatus();
}

absl::Status MlDriftBuffer::ZeroInit() {
  cl_mem mem = GetMemoryForWriting();
  if (!mem) {
    return absl::InternalError("Buffer memory is null");
  }
  if (!queue_) {
    return absl::InternalError("Queue is null");
  }

  ml_drift::TensorStorageType storage_type =
      ml_drift::TensorStorageType::kBuffer;
  ml_drift::int3 desc_region;
  if (context_) {
    auto* tensor = context_->GetTensor(value_id_);
    if (tensor) {
      storage_type = tensor->GetStorageType();
      desc_region = tensor->GetDescriptor().GetFullTensorRegion();
    }
  }

  if (storage_type == ml_drift::TensorStorageType::kBuffer ||
      storage_type == ml_drift::TensorStorageType::kImageBuffer) {
    float zero = 0.0f;
    cl_int error_code = ml_drift::cl::clEnqueueFillBuffer(
        queue_->queue(), mem, &zero, sizeof(float), 0, size_bytes_, 0, nullptr,
        nullptr);
    if (error_code != CL_SUCCESS) {
      std::string error_msg = "Failed to fill buffer with zero - ";
      absl::StrAppend(&error_msg,
                      ml_drift::cl::CLErrorCodeToString(error_code));
      return absl::UnknownError(error_msg);
    }
  } else if (storage_type == ml_drift::TensorStorageType::kTexture2D ||
             storage_type == ml_drift::TensorStorageType::kTexture3D ||
             storage_type == ml_drift::TensorStorageType::kTextureArray ||
             storage_type == ml_drift::TensorStorageType::kSingleTexture2D) {
    float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(desc_region.x),
                        static_cast<size_t>(desc_region.y),
                        static_cast<size_t>(desc_region.z)};
    cl_int error_code = ml_drift::cl::clEnqueueFillImage(
        queue_->queue(), mem, zero, origin, region, 0, nullptr, nullptr);
    if (error_code != CL_SUCCESS) {
      std::string error_msg = "Failed to fill image with zero - ";
      absl::StrAppend(&error_msg,
                      ml_drift::cl::CLErrorCodeToString(error_code));
      return absl::UnknownError(error_msg);
    }
  } else {
    std::string error_msg = "Unsupported storage type for ZeroInit: ";
    absl::StrAppend(&error_msg, static_cast<int>(storage_type));
    return absl::UnimplementedError(error_msg);
  }
  Invalidate();
  return absl::OkStatus();
}

}  // namespace litert::tensor
