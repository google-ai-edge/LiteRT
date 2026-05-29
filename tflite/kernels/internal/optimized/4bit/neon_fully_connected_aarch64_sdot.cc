/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#if defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include <stdint.h>

#include <algorithm>
#include <vector>

#include "tflite/kernels/internal/cppmath.h"
#include "tflite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h"

#define DOTPROD_ATTRIBUTE __attribute__((target("dotprod")))

namespace tflite {
namespace optimized_4bit {

template <int RowsLeft, int RowsRight, int Cols>
void NeonRunKernelSDot(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                       int lhs_layout_rows, int lhs_layout_cols,
                       int rhs_layout_rows, int rhs_layout_cols,
                       int dst_layout_rows, int dst_layout_cols);

template <>
DOTPROD_ATTRIBUTE void NeonRunKernelSDot<4, 1, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 1;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  const int run_depth = depth;
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    __builtin_prefetch(lhs_ptr_data, 0, 3);
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      __builtin_prefetch(rhs_ptr, 0, 3);
      // General register allocation:
      // v0-v3: int32 accumulators for the dot products
      // v16-v19: high int4 values of LHS (after USHR)
      // v20-v23: low int4 values of LHS (after AND)
      // v24-v25: int8 values of RHS
      // v31: mask for AND
      asm volatile(
          R"asm(
          movi v31.16b, #15
          movi v0.4s, #0
          movi v1.4s, #0
          movi v2.4s, #0
          movi v3.4s, #0
          mov w3, %w[run_depth]
            0: /* loop start */
            ld1 {v16.16b, v17.16b, v18.16b, v19.16b}, [%[lhs_ptr]], #64
            ld1 {v24.16b, v25.16b}, [%[rhs_ptr]], #32
            and v20.16b, v16.16b, v31.16b
            and v21.16b, v17.16b, v31.16b
            and v22.16b, v18.16b, v31.16b
            and v23.16b, v19.16b, v31.16b
            ushr v16.16b, v16.16b, #4
            ushr v17.16b, v17.16b, #4
            ushr v18.16b, v18.16b, #4
            ushr v19.16b, v19.16b, #4
            sdot v0.4s, v16.16b, v24.16b
            sdot v1.4s, v17.16b, v24.16b
            sdot v2.4s, v18.16b, v24.16b
            sdot v3.4s, v19.16b, v24.16b
            sdot v0.4s, v20.16b, v25.16b
            sdot v1.4s, v21.16b, v25.16b
            sdot v2.4s, v22.16b, v25.16b
            sdot v3.4s, v23.16b, v25.16b
            subs w3, w3, #1
            b.hi 0b /* loop branch */
          1: /* loop end */
          addp v0.4s, v0.4s, v1.4s
          addp v2.4s, v2.4s, v3.4s
          addp v0.4s, v0.4s, v2.4s
          st1 {v0.4s}, [%[dst]], #16
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst)
          : [run_depth] "r"(run_depth)
          : "cc", "memory", "w3", "v0", "v1", "v2", "v3", "v16", "v17", "v18",
            "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v31");
    }
  }
}

template <>
DOTPROD_ATTRIBUTE void NeonRunKernelSDot<4, 2, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 2;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  const int run_depth = depth;
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    __builtin_prefetch(lhs_ptr_data, 0, 3);
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      __builtin_prefetch(rhs_ptr, 0, 3);
      // General register allocation:
      // v0-v7: int32 accumulators for the dot products
      // v16-v19: high int4 values of LHS (after USHR)
      // v20-v23: low int4 values of LHS (after AND)
      // v24-v27: int8 values of RHS (rows 0 and 1)
      // v31: mask for AND
      asm volatile(
          R"asm(
          movi v31.16b, #15
          movi v0.4s, #0
          movi v1.4s, #0
          movi v2.4s, #0
          movi v3.4s, #0
          movi v4.4s, #0
          movi v5.4s, #0
          movi v6.4s, #0
          movi v7.4s, #0
          mov w3, %w[run_depth]
            0: /* loop start */
            ld1 {v16.16b, v17.16b, v18.16b, v19.16b}, [%[lhs_ptr]], #64
            ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[rhs_ptr]], #64
            and v20.16b, v16.16b, v31.16b
            and v21.16b, v17.16b, v31.16b
            and v22.16b, v18.16b, v31.16b
            and v23.16b, v19.16b, v31.16b
            ushr v16.16b, v16.16b, #4
            ushr v17.16b, v17.16b, #4
            ushr v18.16b, v18.16b, #4
            ushr v19.16b, v19.16b, #4
            sdot v0.4s, v16.16b, v24.16b
            sdot v1.4s, v17.16b, v24.16b
            sdot v2.4s, v18.16b, v24.16b
            sdot v3.4s, v19.16b, v24.16b
            sdot v0.4s, v20.16b, v25.16b
            sdot v1.4s, v21.16b, v25.16b
            sdot v2.4s, v22.16b, v25.16b
            sdot v3.4s, v23.16b, v25.16b
            sdot v4.4s, v16.16b, v26.16b
            sdot v5.4s, v17.16b, v26.16b
            sdot v6.4s, v18.16b, v26.16b
            sdot v7.4s, v19.16b, v26.16b
            sdot v4.4s, v20.16b, v27.16b
            sdot v5.4s, v21.16b, v27.16b
            sdot v6.4s, v22.16b, v27.16b
            sdot v7.4s, v23.16b, v27.16b
            subs w3, w3, #1
            b.hi 0b /* loop branch */
          1: /* loop end */
          addp v0.4s, v0.4s, v1.4s
          addp v2.4s, v2.4s, v3.4s
          addp v4.4s, v4.4s, v5.4s
          addp v6.4s, v6.4s, v7.4s
          addp v0.4s, v0.4s, v2.4s
          addp v4.4s, v4.4s, v6.4s
          st1 {v0.4s}, [%[dst]], #16
          st1 {v4.4s}, [%[dst]], #16
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst)
          : [run_depth] "r"(run_depth)
          : "cc", "memory", "w3", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
            "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
            "v25", "v26", "v27", "v31");
    }
  }
}

template <>
DOTPROD_ATTRIBUTE void NeonRunKernelSDot<4, 4, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 4;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  const int run_depth = depth;
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    __builtin_prefetch(lhs_ptr_data, 0, 3);
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      __builtin_prefetch(rhs_ptr, 0, 3);
      // General register allocation:
      // v0-v15: int32 accumulators for the dot products
      // v16-v19: high int4 values of LHS (after USHR)
      // v20-v23: low int4 values of LHS (after AND)
      // v24-v29: int8 values of RHS (hoisted and interleaved)
      // v31: mask for AND
      asm volatile(
          R"asm(
          movi v31.16b, #15
          movi v0.4s, #0
          movi v1.4s, #0
          movi v2.4s, #0
          movi v3.4s, #0
          movi v4.4s, #0
          movi v5.4s, #0
          movi v6.4s, #0
          movi v7.4s, #0
          movi v8.4s, #0
          movi v9.4s, #0
          movi v10.4s, #0
          movi v11.4s, #0
          movi v12.4s, #0
          movi v13.4s, #0
          movi v14.4s, #0
          movi v15.4s, #0
          mov w3, %w[run_depth]
            0: /* loop start */
            ld1 {v16.16b, v17.16b, v18.16b, v19.16b}, [%[lhs_ptr]], #64
            ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[rhs_ptr]], #64
            and v20.16b, v16.16b, v31.16b
            and v21.16b, v17.16b, v31.16b
            and v22.16b, v18.16b, v31.16b
            and v23.16b, v19.16b, v31.16b
            ushr v16.16b, v16.16b, #4
            ushr v17.16b, v17.16b, #4
            ushr v18.16b, v18.16b, #4
            ushr v19.16b, v19.16b, #4
            sdot v0.4s, v16.16b, v24.16b
            sdot v1.4s, v17.16b, v24.16b
            sdot v2.4s, v18.16b, v24.16b
            sdot v3.4s, v19.16b, v24.16b
            sdot v0.4s, v20.16b, v25.16b
            sdot v1.4s, v21.16b, v25.16b
            sdot v2.4s, v22.16b, v25.16b
            sdot v3.4s, v23.16b, v25.16b
            ld1 {v28.16b, v29.16b}, [%[rhs_ptr]], #32
            sdot v4.4s, v16.16b, v26.16b
            sdot v5.4s, v17.16b, v26.16b
            sdot v6.4s, v18.16b, v26.16b
            sdot v7.4s, v19.16b, v26.16b
            sdot v4.4s, v20.16b, v27.16b
            sdot v5.4s, v21.16b, v27.16b
            sdot v6.4s, v22.16b, v27.16b
            sdot v7.4s, v23.16b, v27.16b
            ld1 {v24.16b, v25.16b}, [%[rhs_ptr]], #32
            sdot v8.4s, v16.16b, v28.16b
            sdot v9.4s, v17.16b, v28.16b
            sdot v10.4s, v18.16b, v28.16b
            sdot v11.4s, v19.16b, v28.16b
            sdot v8.4s, v20.16b, v29.16b
            sdot v9.4s, v21.16b, v29.16b
            sdot v10.4s, v22.16b, v29.16b
            sdot v11.4s, v23.16b, v29.16b
            sdot v12.4s, v16.16b, v24.16b
            sdot v13.4s, v17.16b, v24.16b
            sdot v14.4s, v18.16b, v24.16b
            sdot v15.4s, v19.16b, v24.16b
            sdot v12.4s, v20.16b, v25.16b
            sdot v13.4s, v21.16b, v25.16b
            sdot v14.4s, v22.16b, v25.16b
            sdot v15.4s, v23.16b, v25.16b
            subs w3, w3, #1
            b.hi 0b /* loop branch */
          1: /* loop end */
          addp v0.4s, v0.4s, v1.4s
          addp v2.4s, v2.4s, v3.4s
          addp v4.4s, v4.4s, v5.4s
          addp v6.4s, v6.4s, v7.4s
          addp v8.4s, v8.4s, v9.4s
          addp v10.4s, v10.4s, v11.4s
          addp v12.4s, v12.4s, v13.4s
          addp v14.4s, v14.4s, v15.4s
          addp v0.4s, v0.4s, v2.4s
          addp v4.4s, v4.4s, v6.4s
          addp v8.4s, v8.4s, v10.4s
          addp v12.4s, v12.4s, v14.4s
          st1 {v0.4s}, [%[dst]], #16
          st1 {v4.4s}, [%[dst]], #16
          st1 {v8.4s}, [%[dst]], #16
          st1 {v12.4s}, [%[dst]], #16
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst)
          : [run_depth] "r"(run_depth)
          : "cc", "memory", "w3", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
            "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
            "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
            "v26", "v27", "v28", "v29", "v31");
    }
  }
}

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) ||
        // defined(__ARM_NEON))
