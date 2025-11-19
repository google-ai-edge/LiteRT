// Copyright 2025 Google LLC.
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

#include <gtest/gtest.h>
#include "litert/tools/benchmark_litert_model.h"

namespace litert {
namespace benchmark {

TEST(BenchmarkLiteRtModelQualcommTest, NPUAcceleration) {
  static constexpr char kModelPath[] =
      "/data/local/tmp/runfiles/litert/"
      "test/testdata/simple_add_op_qc_v75_precompiled.tflite";

  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<bool>("use_npu", true);
  params.Set<bool>("use_gpu", false);
  params.Set<bool>("use_cpu", false);
  params.Set<bool>("require_full_delegation", true);
  params.Set<std::string>(
      "dispatch_library_path",
      "/data/local/tmp/runfiles/litert/");
  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);
}

TEST(BenchmarkLiteRtModelQualcommTest, NPUAcceleration_JIT) {
  static constexpr char kModelPath[] =
      "/data/local/tmp/runfiles/litert/"
      "test/testdata/simple_add_op.tflite";

  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<bool>("use_npu", true);
  params.Set<bool>("use_gpu", false);
  params.Set<bool>("use_cpu", false);
  params.Set<bool>("require_full_delegation", true);
  params.Set<std::string>(
      "dispatch_library_path",
      "/data/local/tmp/runfiles/litert/");
  params.Set<std::string>(
      "compiler_plugin_library_path",
      "/data/local/tmp/runfiles/litert/");
  params.Set<std::string>(
      "compiler_cache_path",
      "/data/local/tmp/runfiles/litert/");
  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);
}

}  // namespace benchmark
}  // namespace litert
