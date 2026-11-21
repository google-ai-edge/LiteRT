// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/dma_buf_context_callback.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "QnnContext.h"  // from @qairt
#include <gtest/gtest.h>

namespace litert::qnn {
namespace {

bool HasDmaHeap() {
  int fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
  if (fd < 0) return false;
  close(fd);
  return true;
}

// Creates an unlinked temp file containing `data` and returns its fd. The
// caller owns the fd.
int MakeTempFileWithContent(const std::string& data) {
  int fd = memfd_create("dma_buf_context_callback_test", 0);
  if (fd < 0) return -1;
  if (write(fd, data.data(), data.size()) !=
      static_cast<ssize_t>(data.size())) {
    close(fd);
    return -1;
  }
  return fd;
}

class DmaBufContextCallbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!HasDmaHeap()) {
      GTEST_SKIP() << "No access to /dev/dma_heap/system on this host.";
    }
  }
};

TEST_F(DmaBufContextCallbackTest, ProvideReturnsRequestedBytes) {
  const std::string content(8192, '\0');
  std::string data = content;
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>(i % 251);
  }
  int fd = MakeTempFileWithContent(data);
  ASSERT_GE(fd, 0);

  constexpr uint64_t kFdOffset = 100;
  constexpr uint64_t kSize = 2048;
  auto callback = DmaBufContextCallback::Create(fd, kFdOffset, kSize);
  Qnn_ContextBinaryCallback_t qnn_callback = callback->AsQnnCallback();

  Qnn_ContextBinaryDataRequest_t req{};
  req.offset = 10;
  req.size = 64;
  Qnn_ContextBinaryDmaDataResponse_t resp{};
  Qnn_ErrorHandle_t status = qnn_callback.dmaBufferCallback.v1.dataProvide(
      req, &resp, qnn_callback.dmaBufferCallback.v1.notifyParam);

  ASSERT_EQ(status, QNN_SUCCESS);
  EXPECT_GE(resp.alignedSize, req.size);
  EXPECT_EQ(resp.dataStartOffset, 0);
  ASSERT_NE(resp.dmaBuffer.data, nullptr);
  EXPECT_EQ(
      std::memcmp(resp.dmaBuffer.data, data.data() + kFdOffset + req.offset,
                  req.size),
      0);

  Qnn_ContextBinaryDmaDataMem_t mem{};
  mem.dmaBuffer = resp.dmaBuffer;
  mem.memSize = resp.alignedSize;
  EXPECT_EQ(qnn_callback.dmaBufferCallback.v1.dataRelease(
                mem, qnn_callback.dmaBufferCallback.v1.notifyParam),
            QNN_SUCCESS);

  close(fd);
}

TEST_F(DmaBufContextCallbackTest, ProvideReturnsDistinctFdsPerRequest) {
  const std::string data(4096, 'x');
  int fd = MakeTempFileWithContent(data);
  ASSERT_GE(fd, 0);

  auto callback = DmaBufContextCallback::Create(fd, /*fd_offset=*/0,
                                                /*size=*/data.size());
  Qnn_ContextBinaryCallback_t qnn_callback = callback->AsQnnCallback();

  Qnn_ContextBinaryDataRequest_t req{};
  req.offset = 0;
  req.size = 64;
  Qnn_ContextBinaryDmaDataResponse_t resp1{};
  Qnn_ContextBinaryDmaDataResponse_t resp2{};
  ASSERT_EQ(qnn_callback.dmaBufferCallback.v1.dataProvide(
                req, &resp1, qnn_callback.dmaBufferCallback.v1.notifyParam),
            QNN_SUCCESS);
  ASSERT_EQ(qnn_callback.dmaBufferCallback.v1.dataProvide(
                req, &resp2, qnn_callback.dmaBufferCallback.v1.notifyParam),
            QNN_SUCCESS);

  EXPECT_NE(resp1.dmaBuffer.fd, resp2.dmaBuffer.fd);

  for (auto* resp : {&resp1, &resp2}) {
    Qnn_ContextBinaryDmaDataMem_t mem{};
    mem.dmaBuffer = resp->dmaBuffer;
    mem.memSize = resp->alignedSize;
    qnn_callback.dmaBufferCallback.v1.dataRelease(
        mem, qnn_callback.dmaBufferCallback.v1.notifyParam);
  }
  close(fd);
}

TEST_F(DmaBufContextCallbackTest, ProvideRejectsOutOfRangeRequest) {
  const std::string data(256, 'y');
  int fd = MakeTempFileWithContent(data);
  ASSERT_GE(fd, 0);

  auto callback = DmaBufContextCallback::Create(fd, /*fd_offset=*/0,
                                                /*size=*/data.size());
  Qnn_ContextBinaryCallback_t qnn_callback = callback->AsQnnCallback();

  Qnn_ContextBinaryDataRequest_t req{};
  req.offset = data.size() - 1;
  req.size = 2;  // offset + size exceeds the registered `size`.
  Qnn_ContextBinaryDmaDataResponse_t resp{};
  EXPECT_EQ(qnn_callback.dmaBufferCallback.v1.dataProvide(
                req, &resp, qnn_callback.dmaBufferCallback.v1.notifyParam),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT);

  Qnn_ContextBinaryDataRequest_t zero_req{};
  zero_req.offset = 0;
  zero_req.size = 0;
  Qnn_ContextBinaryDmaDataResponse_t zero_resp{};
  EXPECT_EQ(
      qnn_callback.dmaBufferCallback.v1.dataProvide(
          zero_req, &zero_resp, qnn_callback.dmaBufferCallback.v1.notifyParam),
      QNN_CONTEXT_ERROR_INVALID_ARGUMENT);

  close(fd);
}

TEST_F(DmaBufContextCallbackTest, ReleaseOfUnknownBufferIsNoop) {
  const std::string data(256, 'z');
  int fd = MakeTempFileWithContent(data);
  ASSERT_GE(fd, 0);

  auto callback = DmaBufContextCallback::Create(fd, /*fd_offset=*/0,
                                                /*size=*/data.size());
  Qnn_ContextBinaryCallback_t qnn_callback = callback->AsQnnCallback();

  Qnn_ContextBinaryDmaDataMem_t mem{};
  mem.dmaBuffer.fd = -1;
  mem.dmaBuffer.data = nullptr;
  mem.memSize = 0;
  EXPECT_EQ(qnn_callback.dmaBufferCallback.v1.dataRelease(
                mem, qnn_callback.dmaBufferCallback.v1.notifyParam),
            QNN_SUCCESS);

  close(fd);
}

TEST_F(DmaBufContextCallbackTest, DestructorReleasesOutstandingBuffers) {
  const std::string data(256, 'w');
  int fd = MakeTempFileWithContent(data);
  ASSERT_GE(fd, 0);

  {
    auto callback = DmaBufContextCallback::Create(fd, /*fd_offset=*/0,
                                                  /*size=*/data.size());
    Qnn_ContextBinaryCallback_t qnn_callback = callback->AsQnnCallback();
    Qnn_ContextBinaryDataRequest_t req{};
    req.offset = 0;
    req.size = 64;
    Qnn_ContextBinaryDmaDataResponse_t resp{};
    ASSERT_EQ(qnn_callback.dmaBufferCallback.v1.dataProvide(
                  req, &resp, qnn_callback.dmaBufferCallback.v1.notifyParam),
              QNN_SUCCESS);
    // `callback` goes out of scope here without an explicit Release() call;
    // the destructor must munmap/close the outstanding buffer rather than
    // leaking it or crashing.
  }

  close(fd);
}

}  // namespace
}  // namespace litert::qnn
