#include "litert/runtime/tensor_buffer_registry.h"

#include <cstddef>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {

LiteRtStatus MyCustomTensorBufferCreate(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void** cl_memory) {
  return kLiteRtStatusOk;
}

LiteRtStatus MyCustomTensorBufferUpload(
    void* cl_memory, size_t bytes, const void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) {
  return kLiteRtStatusOk;
}

LiteRtStatus MyCustomTensorBufferDownload(
    void* cl_memory, size_t bytes, void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) {
  return kLiteRtStatusOk;
}

TEST(TensorBufferRegistryTest, Basic) {
  auto& registry = litert::internal::TensorBufferRegistry::GetInstance();
  EXPECT_EQ(registry.RegisterAccessors(
                kLiteRtTensorBufferTypeOpenClBuffer, MyCustomTensorBufferCreate,
                MyCustomTensorBufferUpload, MyCustomTensorBufferDownload),
            kLiteRtStatusOk);
  void* hw_memory = nullptr;
  LiteRtRankedTensorType* tensor_type = nullptr;
  constexpr int kBufferSize = 10;
  char temp_buffer[kBufferSize];
  EXPECT_EQ(registry.CreateCustomTensorBuffer(
                tensor_type, kLiteRtTensorBufferTypeOpenClBuffer, kBufferSize,
                &hw_memory),
            kLiteRtStatusOk);
  EXPECT_EQ(registry.UploadCustomTensorBuffer(
                hw_memory, kBufferSize, temp_buffer, tensor_type,
                kLiteRtTensorBufferTypeOpenClBuffer),
            kLiteRtStatusOk);
  EXPECT_EQ(registry.DownloadCustomTensorBuffer(
                hw_memory, kBufferSize, temp_buffer, tensor_type,
                kLiteRtTensorBufferTypeOpenClBuffer),
            kLiteRtStatusOk);
}

}  // namespace
