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

#include "litert/runtime/ahwb_buffer.h"

#include <cstddef>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/runtime/event.h"

#if LITERT_HAS_SYNC_FENCE_SUPPORT
#include <fcntl.h>
#include <unistd.h>
#endif  // LITERT_HAS_SYNC_FENCE_SUPPORT

namespace litert::internal {
namespace {

constexpr size_t kBufferSize = 64;

// AHardwareBuffer_lock() takes ownership of its acquire fence fd and closes
// it. AhwbBuffer::Lock() must not hand over the fd that `event` owns, or
// ~LiteRtEventT closes the fd a second time (b/577170510).
TEST(AhwbBufferTest, LockKeepsEventFenceFdOpen) {
#if !LITERT_HAS_SYNC_FENCE_SUPPORT
  GTEST_SKIP() << "Sync fences are not supported on this platform";
#else
  if (!AhwbBuffer::IsSupported()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform";
  }
  auto buffer = AhwbBuffer::Alloc(kBufferSize);
  ASSERT_TRUE(buffer.HasValue()) << buffer.Error().Message();

  // Use the read end of a pipe whose write end is closed as the event fd.
  // poll() reports it as ready at once, so a fence wait does not block.
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  ASSERT_EQ(close(pipe_fds[1]), 0);

  LiteRtEventT event;
  event.type = LiteRtEventTypeSyncFenceFd;
  event.fd = pipe_fds[0];
  event.owns_fd = true;

  // The pipe fd is not a real sync fence, so the lock can fail. The fence
  // passed to AHardwareBuffer_lock() is consumed in both cases.
  auto host_addr = AhwbBuffer::Lock(buffer->ahwb, &event);
  if (host_addr.HasValue()) {
    EXPECT_TRUE(AhwbBuffer::Unlock(buffer->ahwb).HasValue());
  }

  EXPECT_NE(fcntl(event.fd, F_GETFD), -1)
      << "AhwbBuffer::Lock() let the event fence fd get closed";

  AhwbBuffer::Free(buffer->ahwb);
#endif  // !LITERT_HAS_SYNC_FENCE_SUPPORT
}

}  // namespace
}  // namespace litert::internal
