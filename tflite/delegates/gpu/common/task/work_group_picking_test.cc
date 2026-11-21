/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tflite/delegates/gpu/common/task/work_group_picking.h"

#include <gtest/gtest.h>

#include <vector>

#include "tflite/delegates/gpu/common/gpu_info.h"
#include "tflite/delegates/gpu/common/kernel_info.h"
#include "tflite/delegates/gpu/common/task/tuning_type.h"
#include "tflite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

GpuInfo GpuWithWorkGroupLimits(const int3& limits) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.max_work_group_size_x = limits.x;
  gpu_info.opencl_info.max_work_group_size_y = limits.y;
  gpu_info.opencl_info.max_work_group_size_z = limits.z;
  return gpu_info;
}

TEST(WorkGroupPicking, FastRespectsPerAxisLimit) {
  const auto gpu_info = GpuWithWorkGroupLimits({1024, 1024, 1024});
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = 2048;
  std::vector<int3> groups;
  GetPossibleWorkGroups(TuningType::kFast, gpu_info, kernel_info, {12800, 1, 1},
                        &groups);
  ASSERT_EQ(groups.size(), 1);
  EXPECT_LE(groups[0].x, 1024);
  EXPECT_EQ(groups[0].y, 1);
  EXPECT_EQ(groups[0].z, 1);
}

TEST(WorkGroupPicking, FastRespectsAsymmetricAxisLimits) {
  const int3 limits(128, 4, 2);
  const auto gpu_info = GpuWithWorkGroupLimits(limits);
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = 2048;
  for (const int3 grid : {int3(16384, 16, 32), int3(1, 16384, 32)}) {
    std::vector<int3> groups;
    GetPossibleWorkGroups(TuningType::kFast, gpu_info, kernel_info, grid,
                          &groups);
    ASSERT_EQ(groups.size(), 1);
    EXPECT_GE(groups[0].x, 1);
    EXPECT_GE(groups[0].y, 1);
    EXPECT_GE(groups[0].z, 1);
    EXPECT_LE(groups[0].x, limits.x);
    EXPECT_LE(groups[0].y, limits.y);
    EXPECT_LE(groups[0].z, limits.z);
    EXPECT_LE(groups[0].x * groups[0].y * groups[0].z,
              kernel_info.max_work_group_size);
  }
}

TEST(WorkGroupPicking, FastPreservesValidSelection) {
  const auto gpu_info = GpuWithWorkGroupLimits({1024, 1024, 1024});
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = 1024;
  std::vector<int3> groups;
  GetPossibleWorkGroups(TuningType::kFast, gpu_info, kernel_info, {12800, 1, 1},
                        &groups);
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0].x, 1024);
  EXPECT_EQ(groups[0].y, 1);
  EXPECT_EQ(groups[0].z, 1);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
