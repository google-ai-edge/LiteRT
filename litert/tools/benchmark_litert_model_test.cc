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

#include <fstream>
#include <ios>
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
    "litert/test/testdata/"
    "mobilenet_v2_1.0_224.tflite";
static constexpr char kSignatureToRunFor[] = "<placeholder signature>";

class TestBenchmarkListener : public BenchmarkListener {
 public:
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    results_ = results;
  }

  BenchmarkResults results_;
};

TEST(BenchmarkLiteRtModelTest, GetModelSizeFromPathSucceeded) {
  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<std::string>("signature_to_run_for", kSignatureToRunFor);
  params.Set<int>("num_runs", 1);
  params.Set<int>("warmup_runs", 0);
  params.Set<bool>("use_cpu", true);
  params.Set<bool>("use_gpu", false);
  params.Set<bool>("require_full_delegation", false);
  params.Set<bool>("use_profiler", true);

  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));
  TestBenchmarkListener listener;
  benchmark.AddListener(&listener);

  benchmark.Run();

  EXPECT_GE(listener.results_.model_size_mb(), 0);
}

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

TEST(BenchmarkLiteRtModelTest, BenchmarkWithResultFilePath) {
  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<std::string>("signature_to_run_for", kSignatureToRunFor);
  params.Set<bool>("use_cpu", true);
  params.Set<bool>("use_gpu", false);
  params.Set<bool>("require_full_delegation", false);

#if defined(__ANDROID__)
  std::string result_file_path = "/data/local/tmp/benchmark_result.pb";
#else
  std::string result_file_path = "/tmp/benchmark_result.pb";
#endif
  params.Set<std::string>("result_file_path", result_file_path);

  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));
  TestBenchmarkListener listener;
  benchmark.AddListener(&listener);

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);

  std::ifstream in_file(result_file_path, std::ios::binary | std::ios::in);
  BenchmarkResult result;
  result.ParseFromIstream(&in_file);

  // Verify latency metrics.
  EXPECT_FLOAT_EQ(result.latency_metrics().init_ms(),
                  listener.results_.startup_latency_us() / 1000.0);
  EXPECT_FLOAT_EQ(result.latency_metrics().first_inference_ms(),
                  listener.results_.warmup_time_us().first() / 1000.0);
  EXPECT_FLOAT_EQ(result.latency_metrics().average_warm_up_ms(),
                  listener.results_.warmup_time_us().avg() / 1000.0);
  EXPECT_FLOAT_EQ(result.latency_metrics().avg_ms(),
                  listener.results_.inference_time_us().avg() / 1000.0);
  EXPECT_FLOAT_EQ(result.latency_metrics().min_ms(),
                  listener.results_.inference_time_us().min() / 1000.0);
  EXPECT_FLOAT_EQ(result.latency_metrics().max_ms(),
                  listener.results_.inference_time_us().max() / 1000.0);
  EXPECT_FLOAT_EQ(
      result.latency_metrics().stddev_ms(),
      listener.results_.inference_time_us().std_deviation() / 1000.0);
  EXPECT_FLOAT_EQ(
      result.latency_metrics().median_ms(),
      listener.results_.inference_time_us().percentile(50) / 1000.0);
  EXPECT_FLOAT_EQ(
      result.latency_metrics().p95_ms(),
      listener.results_.inference_time_us().percentile(95) / 1000.0);
  EXPECT_FLOAT_EQ(
      result.latency_metrics().p5_ms(),
      listener.results_.inference_time_us().percentile(5) / 1000.0);

  // Verify memory metrics.
  EXPECT_EQ(result.memory_metrics().init_footprint_kb(),
            listener.results_.init_mem_usage().mem_footprint_kb);
  EXPECT_EQ(result.memory_metrics().overall_footprint_kb(),
            listener.results_.overall_mem_usage().mem_footprint_kb);
  EXPECT_EQ(result.memory_metrics().has_peak_mem_mb(), false);

  // Verify misc metrics.
  EXPECT_FLOAT_EQ(result.misc_metrics().model_size_mb(),
                  listener.results_.model_size_mb());
  EXPECT_EQ(result.misc_metrics().num_runs(),
            listener.results_.inference_time_us().count());
  EXPECT_EQ(result.misc_metrics().num_warmup_runs(),
            listener.results_.warmup_time_us().count());
  EXPECT_FLOAT_EQ(result.misc_metrics().model_throughput_in_mb_per_sec(),
                  listener.results_.throughput_MB_per_second());
}

TEST(BenchmarkLiteRtModelTest, BenchmarkWithModelRuntimeInfoFilePath) {
  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<std::string>("signature_to_run_for", kSignatureToRunFor);
  params.Set<bool>("use_cpu", true);
  params.Set<bool>("use_gpu", false);
  params.Set<bool>("require_full_delegation", false);
  std::string model_runtime_info_output_file;
#if defined(__ANDROID__)
  model_runtime_info_output_file = "/data/local/tmp/model_runtime_info.pb";
#else
  model_runtime_info_output_file = "/tmp/model_runtime_info.pb";
#endif
  params.Set<std::string>("model_runtime_info_output_file",
                          model_runtime_info_output_file);

  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));
  EXPECT_EQ(benchmark.Run(), kTfLiteOk);

  std::ifstream in_file(model_runtime_info_output_file,
                        std::ios::binary | std::ios::in);
  tflite::profiling::ModelRuntimeDetails model_runtime_details;
  model_runtime_details.ParseFromIstream(&in_file);
  EXPECT_EQ(model_runtime_details.subgraphs_size(), 1);
  EXPECT_GT(model_runtime_details.subgraphs(0).execution_plan_size(), 0);
  EXPECT_GT(model_runtime_details.subgraphs(0).nodes_size(), 0);
  EXPECT_GT(model_runtime_details.subgraphs(0).edges_size(), 0);
}

}  // namespace
}  // namespace benchmark
}  // namespace litert
