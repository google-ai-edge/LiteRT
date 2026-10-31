/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

#include "tflite/kernels/internal/portable_tensor_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "tflite/core/c/builtin_op_data.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

namespace {

template <int BitWidth>
int8_t SignExtend(uint8_t value) {
  static_assert(BitWidth > 0 && BitWidth < 8);
  constexpr uint8_t kMask = (uint8_t{1} << BitWidth) - 1;
  constexpr uint8_t kSignBit = uint8_t{1} << (BitWidth - 1);
  value &= kMask;
  return static_cast<int8_t>((value ^ kSignBit) - kSignBit);
}

}  // namespace

// Apply Rectified Linear to elements of a vector.
void ApplyReluToVector(const float* __restrict__ vector, int v_size,
                       float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, vector[v]);
  }
}

// Apply Rectified Linear 1 (cap to [-1;1]) to elements of a vector
void ApplyRelu1ToVector(const float* __restrict__ vector, int v_size,
                        float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(-1.0f, std::min(vector[v], 1.0f));
  }
}

// Apply Rectified Linear 6 (cap to [0;6]) to elements of a vector
void ApplyRelu6ToVector(const float* __restrict__ vector, int v_size,
                        float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, std::min(vector[v], 6.0f));
  }
}

// Apply signbit to elements of a vector
void ApplySignbitToVector(const float* __restrict__ vector, int v_size,
                          float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::signbit(vector[v]);
  }
}

void UnpackDenseInt4IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer) {
  // num_elements means the number of elements regardless of packed or unpacked.
  // For example, 3 elements means both
  //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
  //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
  //   2) Unpacked: 3 int8's = 3 bytes.
  //.     stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j = 0..2)
  for (int i = 0; i < num_elements / 2; i++) {
    const uint8_t byte = static_cast<uint8_t>(src_buffer[i]);
    const int8_t lower = SignExtend<4>(byte);
    const int8_t higher = SignExtend<4>(byte >> 4);
    dst_buffer[2 * i] = lower;
    dst_buffer[2 * i + 1] = higher;
  }

  // If the buffer size is odd, extract the final lower nibble.
  if (num_elements % 2 != 0) {
    dst_buffer[num_elements - 1] = SignExtend<4>(
        static_cast<uint8_t>(src_buffer[num_elements / 2]));
  }
}

void UnpackPackedIntToInt8(const int8_t* src_buffer, int num_elements,
                           int bit_width, int8_t* dst_buffer,
                           bool unpack_unsigned) {
  assert(bit_width == 2 || bit_width == 4);
  if (bit_width == 4) {
    // num_elements means the number of elements regardless of packed or
    // unpacked. For example, 3 elements means both
    //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
    //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
    //   2) Unpacked: 3 int8's = 3 bytes.
    //.     stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j = 0..2)
    for (int i = 0; i < num_elements / 2; i++) {
      const uint8_t byte = static_cast<uint8_t>(src_buffer[i]);
      int8_t lower, higher;
      if (unpack_unsigned) {
        lower = byte & 0x0F;
        higher = (byte >> 4) & 0x0F;
      } else {
        lower = SignExtend<4>(byte);
        higher = SignExtend<4>(byte >> 4);
      }
      dst_buffer[2 * i] = lower;
      dst_buffer[2 * i + 1] = higher;
    }

    // If the buffer size is odd, extract the final lower nibble.
    if (num_elements % 2 != 0) {
      const uint8_t byte =
          static_cast<uint8_t>(src_buffer[num_elements / 2]);
      dst_buffer[num_elements - 1] =
          unpack_unsigned ? (byte & 0x0F) : SignExtend<4>(byte);
    }
  } else if (bit_width == 2) {
    for (int i = 0; i < num_elements / 4; i++) {
      const uint8_t byte = static_cast<uint8_t>(src_buffer[i]);
      if (unpack_unsigned) {
        dst_buffer[4 * i] = byte & 0x03;
        dst_buffer[4 * i + 1] = (byte >> 2) & 0x03;
        dst_buffer[4 * i + 2] = (byte >> 4) & 0x03;
        dst_buffer[4 * i + 3] = (byte >> 6) & 0x03;
      } else {
        const int8_t val1 = SignExtend<2>(byte);
        const int8_t val2 = SignExtend<2>(byte >> 2);
        const int8_t val3 = SignExtend<2>(byte >> 4);
        const int8_t val4 = SignExtend<2>(byte >> 6);
        dst_buffer[4 * i] = val1;
        dst_buffer[4 * i + 1] = val2;
        dst_buffer[4 * i + 2] = val3;
        dst_buffer[4 * i + 3] = val4;
      }
    }

    // Handle the remaining elements.
    int remaining_elements = num_elements % 4;
    if (remaining_elements > 0) {
      const uint8_t byte =
          static_cast<uint8_t>(src_buffer[num_elements / 4]);
      for (int i = 0; i < remaining_elements; i++) {
        if (unpack_unsigned) {
          dst_buffer[num_elements - remaining_elements + i] =
              (byte >> (2 * i)) & 0x03;
        } else {
          dst_buffer[num_elements - remaining_elements + i] =
              SignExtend<2>(byte >> (2 * i));
        }
      }
    }
  }
}

void PackInt8IntoDenseInt(const int8_t* src_buffer, int num_elements,
                          int bit_width, int8_t* dst_buffer) {
  assert(bit_width == 2 || bit_width == 4);
  if (bit_width == 4) {
    // num_elements means the number of elements regardless of packed or
    // unpacked. For example, 3 elements means both
    //   1) Unpacked: 3 int8's = 3 bytes.
    //      stored in src_buffer[0], src_buffer[1] and src_buffer[2] (j = 0..2)
    //   2) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
    //      stored in dst_buffer[0] and dst_buffer[1] (i = 0..1)
    for (int i = 0; i < num_elements / 2; ++i) {
      const uint8_t low = static_cast<uint8_t>(src_buffer[2 * i]) & 0x0F;
      const uint8_t high =
          (static_cast<uint8_t>(src_buffer[2 * i + 1]) & 0x0F) << 4;
      dst_buffer[i] = static_cast<int8_t>(low | high);
    }
    // If the buffer size is odd, pack the final nibble.
    if (num_elements % 2 != 0) {
      dst_buffer[num_elements / 2] = static_cast<int8_t>(
          static_cast<uint8_t>(src_buffer[num_elements - 1]) & 0x0F);
    }
  } else if (bit_width == 2) {
    for (int i = 0; i < num_elements / 4; ++i) {
      const uint8_t packed =
          (static_cast<uint8_t>(src_buffer[4 * i]) & 0x03) |
          ((static_cast<uint8_t>(src_buffer[4 * i + 1]) & 0x03) << 2) |
          ((static_cast<uint8_t>(src_buffer[4 * i + 2]) & 0x03) << 4) |
          ((static_cast<uint8_t>(src_buffer[4 * i + 3]) & 0x03) << 6);
      dst_buffer[i] = static_cast<int8_t>(packed);
    }
    // Handle the remaining elements.
    int remaining_elements = num_elements % 4;
    if (remaining_elements > 0) {
      uint8_t packed_val = 0;
      for (int i = 0; i < remaining_elements; ++i) {
        packed_val |=
            (static_cast<uint8_t>(src_buffer[num_elements - remaining_elements + i]) &
             0x03)
            << (i * 2);
      }
      dst_buffer[num_elements / 4] = static_cast<int8_t>(packed_val);
    }
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
