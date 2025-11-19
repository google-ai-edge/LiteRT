/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "litert/tools/benchmark_litert_model.h"

#include <fcntl.h>
#include <sys/stat.h>


#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "tflite/core/c/c_api_types.h"
#include "tflite/tools/benchmark/benchmark_model.h"
#include "tflite/tools/benchmark/benchmark_params.h"

namespace litert {
namespace benchmark {
namespace {
using ::litert::benchmark::BenchmarkLiteRtModel;
using ::tflite::benchmark::BenchmarkListener;
using ::tflite::benchmark::BenchmarkParams;
using ::tflite::benchmark::BenchmarkResults;

static constexpr char kModelPath[] =
    "third_party/odml/litert/litert/test/testdata/"
    "mobilenet_v2_1.0_224.tflite";
static constexpr char kSignatureToRunFor[] = "<placeholder signature>";

class TestBenchmarkListener : public BenchmarkListener {
 public:
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    results_ = results;
  }

  BenchmarkResults results_;
};

TEST(BenchmarkLiteRtModelTest, GPUAcceleration) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif
  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<std::string>("signature_to_run_for", kSignatureToRunFor);
  params.Set<bool>("use_cpu", false);
  params.Set<bool>("use_gpu", true);
  params.Set<bool>("require_full_delegation", true);

  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);
}

TEST(BenchmarkLiteRtModelTest, GPUAccelerationWithProfiler) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif
  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<std::string>("signature_to_run_for", kSignatureToRunFor);
  params.Set<bool>("use_cpu", false);
  params.Set<bool>("use_gpu", true);
  params.Set<bool>("require_full_delegation", true);
  params.Set<bool>("use_profiler", true);

  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);
}

}  // namespace
}  // namespace benchmark
}  // namespace litert
