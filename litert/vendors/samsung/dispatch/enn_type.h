// Copyright (C) 2026 Samsung Electronics Co. LTD. 
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
#pragma once

#include <stdint.h>

typedef uint64_t EnnModelId; // higher 32-bits are initialized with zero

typedef enum _EnnReturn {
  ENN_RET_SUCCESS = 0,
  ENN_RET_FAILED,
  ENN_RET_IO,
  ENN_RET_INVAL,
  ENN_RET_FILTERED,
  ENN_RET_CANCELED,
  ENN_RET_MEM_ERR,
  ENN_RET_SIZE,
  ENN_RET_FAILED_TIMEOUT_ENN = 10,
  ENN_RET_FAILED_TIMEOUT_DD,
  ENN_RET_FAILED_TIMEOUT_FW,
  ENN_RET_FAILED_TIMEOUT_HW_NOTRECOVERED,
  ENN_RET_FAILED_TIMEOUT_HW_RECOVERED,
  ENN_RET_FAILED_SERVICE_NULL,
  ENN_RET_FAILED_RESOURCE_BUSY,
  ENN_RET_NOT_SUPPORTED = 0xFF,
} EnnReturn;

/* NOTE: should be sync with types.hal */
typedef enum _enn_buf_dir_e {
  ENN_DIR_IN,
  ENN_DIR_OUT,
  ENN_DIR_EXT,
  ENN_DIR_NONE,
  ENN_DIR_SIZE
} enn_buf_dir_e;

// data structure for user buffer
typedef struct _ennBuffer {
  void *va;
  uint32_t size; // requested size
  uint32_t offset;
} EnnBuffer;

typedef EnnBuffer *EnnBufferPtr;

typedef struct _NumberOfBuffersInfo {
  uint32_t n_in_buf;
  uint32_t n_out_buf;
} NumberOfBuffersInfo;
