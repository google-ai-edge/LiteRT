#include <gtest/gtest.h>
#include "litert/tools/benchmark_litert_model.h"

namespace litert {
namespace benchmark {

static constexpr char kModelPath[] =
    "/data/local/tmp/runfiles/litert/"
    "test/testdata/simple_add_op_qc_v75_precompiled.tflite";

TEST(BenchmarkLiteRtModelQualcommTest, NPUAcceleration) {
  BenchmarkParams params = BenchmarkLiteRtModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<bool>("use_npu", true);
  params.Set<bool>("use_gpu", false);
  params.Set<bool>("use_cpu", false);
  params.Set<bool>("require_full_delegation", true);
  params.Set<std::string>(
      "qnn_dispatch_library_path",
      "/data/local/tmp/runfiles/litert/");
  BenchmarkLiteRtModel benchmark = BenchmarkLiteRtModel(std::move(params));

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);
}

}  // namespace benchmark
}  // namespace litert
