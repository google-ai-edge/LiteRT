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

#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/core/dispatch_op_schema.h"

namespace litert::internal {
namespace {

TEST(DispatchDelegateKernelTest, BuildExecutableBytecodeBufferWithoutFd) {
  const DispatchOpOptions dispatch_options = {
      .bytecode_size = 64,
      .bytecode_offset = 128,
      .name = "main",
  };

  auto bytecode_buffer =
      BuildExecutableBytecodeBuffer(dispatch_options, /*alloc_base=*/nullptr,
                                    /*alloc_base_fd=*/-1,
                                    /*alloc_base_file_offset=*/0,
                                    /*alloc_base_size=*/0,
                                    /*has_alloc_base_file_region=*/false);
  ASSERT_TRUE(bytecode_buffer);
  EXPECT_EQ(bytecode_buffer->fd, -1);
  EXPECT_EQ(bytecode_buffer->offset, dispatch_options.bytecode_offset);
  EXPECT_EQ(bytecode_buffer->size, dispatch_options.bytecode_size);
  EXPECT_EQ(bytecode_buffer->alloc_base_file_offset, 0);
  // Allocation type defaults to unknown (the conservative value) when the
  // caller does not provide one.
  EXPECT_EQ(bytecode_buffer->allocation_type,
            kLiteRtModelAllocationTypeUnknown);
}

TEST(DispatchDelegateKernelTest,
     BuildExecutableBytecodeBufferPropagatesAllocationType) {
  const DispatchOpOptions dispatch_options = {
      .bytecode_size = 64,
      .bytecode_offset = 128,
      .name = "main",
  };

  // Host memory (e.g. a model created with LiteRtCreateModelFromBuffer) has no
  // fd but still needs its allocation type forwarded to the vendor.
  auto host_buffer =
      BuildExecutableBytecodeBuffer(dispatch_options, /*alloc_base=*/nullptr,
                                    /*alloc_base_fd=*/-1,
                                    /*alloc_base_file_offset=*/0,
                                    /*alloc_base_size=*/0,
                                    /*has_alloc_base_file_region=*/false,
                                    kLiteRtModelAllocationTypeHostMemory);
  ASSERT_TRUE(host_buffer);
  EXPECT_EQ(host_buffer->fd, -1);
  EXPECT_EQ(host_buffer->allocation_type, kLiteRtModelAllocationTypeHostMemory);

  // File-backed mmap (e.g. LiteRtCreateModelFromFile / FromFd).
  auto mmap_buffer =
      BuildExecutableBytecodeBuffer(dispatch_options, /*alloc_base=*/nullptr,
                                    /*alloc_base_fd=*/17,
                                    /*alloc_base_file_offset=*/4096,
                                    /*alloc_base_size=*/8192,
                                    /*has_alloc_base_file_region=*/true,
                                    kLiteRtModelAllocationTypeFileBackedMmap);
  ASSERT_TRUE(mmap_buffer);
  EXPECT_EQ(mmap_buffer->fd, 17);
  EXPECT_EQ(mmap_buffer->allocation_type,
            kLiteRtModelAllocationTypeFileBackedMmap);
}

TEST(DispatchDelegateKernelTest, BuildExecutableBytecodeBufferWithFd) {
  const DispatchOpOptions dispatch_options = {
      .bytecode_size = 64,
      .bytecode_offset = 128,
      .name = "main",
  };

  auto bytecode_buffer =
      BuildExecutableBytecodeBuffer(dispatch_options, /*alloc_base=*/nullptr,
                                    /*alloc_base_fd=*/17,
                                    /*alloc_base_file_offset=*/4096,
                                    /*alloc_base_size=*/8192,
                                    /*has_alloc_base_file_region=*/true);
  ASSERT_TRUE(bytecode_buffer);
  EXPECT_EQ(bytecode_buffer->fd, 17);
  EXPECT_EQ(bytecode_buffer->offset, dispatch_options.bytecode_offset);
  EXPECT_EQ(bytecode_buffer->alloc_base_file_offset, 4096u);
  EXPECT_EQ(bytecode_buffer->alloc_base_file_offset + bytecode_buffer->offset,
            4096u + dispatch_options.bytecode_offset);
  EXPECT_EQ(bytecode_buffer->size, dispatch_options.bytecode_size);
}

TEST(DispatchDelegateKernelTest,
     BuildExecutableBytecodeBufferRejectsOutOfBoundsBytecode) {
  const DispatchOpOptions dispatch_options = {
      .bytecode_size = 64,
      .bytecode_offset = 128,
      .name = "main",
  };

  auto bytecode_buffer =
      BuildExecutableBytecodeBuffer(dispatch_options, /*alloc_base=*/nullptr,
                                    /*alloc_base_fd=*/17,
                                    /*alloc_base_file_offset=*/4096,
                                    /*alloc_base_size=*/191,
                                    /*has_alloc_base_file_region=*/true);
  ASSERT_FALSE(bytecode_buffer);
}

TEST(DispatchDelegateKernelTest, BuildExecutableBytecodeBufferRejectsMissingFd) {
  const DispatchOpOptions dispatch_options = {
      .bytecode_size = 64,
      .bytecode_offset = 128,
      .name = "main",
  };

  auto bytecode_buffer =
      BuildExecutableBytecodeBuffer(dispatch_options, /*alloc_base=*/nullptr,
                                    /*alloc_base_fd=*/-1,
                                    /*alloc_base_file_offset=*/4096,
                                    /*alloc_base_size=*/8192,
                                    /*has_alloc_base_file_region=*/true);
  ASSERT_FALSE(bytecode_buffer);
}

}  // namespace
}  // namespace litert::internal
