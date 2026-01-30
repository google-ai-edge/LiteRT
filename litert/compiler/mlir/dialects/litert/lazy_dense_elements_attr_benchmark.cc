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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/profiler.h"
#include "perftools/profiles/collector/heap/alloc_recorder.h"
#include "perftools/profiles/proto/builder.h"
#include "perftools/profiles/proto/encoder.h"
#include <gtest/gtest.h>
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"
#include "third_party/tcmalloc/malloc_extension.h"
#include "tflite/converter/ir/tfl_ops.h"

namespace litert {
namespace {

using ::testing::TestWithParam;
using ::testing::ValuesIn;
namespace heap_profile = ::perftools::profiles::collector::heap;

enum class Dialect { LRT, TFL };

enum class ProfileType { CPU, PEAK_HEAP, ALLOC_RECORDER };

struct BenchmarkTestCase {
  std::string test_name;
  Dialect dialect;
  ProfileType profile_type;
  size_t tensor_size_mb;
};

using LazyDenseElementsAttrBenchmark = TestWithParam<BenchmarkTestCase>;

std::unique_ptr<mlir::MLIRContext> CreateContext(Dialect dialect) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  if (dialect == Dialect::LRT) {
    registry.insert<litert::LITERTDialect>();
  } else if (dialect == Dialect::TFL) {
    registry.insert<mlir::TFL::TFLDialect>();
  }
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

// Prevents compiler from pre-allocating memory.
std::unique_ptr<std::vector<float>> GenerateVector(size_t size,
                                                   float default_value) {
  auto data = std::make_unique<std::vector<float>>(size, default_value);
  // defeat splat elements attrs by making the values non-uniform
  (*data)[size - 1] = default_value + 1;
  return data;
}

litert::LazyDenseElementsAttr CreateAttr(llvm::ArrayRef<float> data,
                                         mlir::FloatType type) {
  auto shape =
      mlir::RankedTensorType::get({static_cast<int64_t>(data.size())}, type);
  return litert::LazyDenseElementsAttr::get<float>(shape, data);
}

mlir::DenseElementsAttr CreateDenseAttr(llvm::ArrayRef<float> data,
                                        mlir::FloatType type) {
  auto shape =
      mlir::RankedTensorType::get({static_cast<int64_t>(data.size())}, type);
  return mlir::DenseElementsAttr::get<float>(shape, data);
}

mlir::OwningOpRef<mlir::ModuleOp> CreateProgram(
    Dialect dialect, size_t tensor_size,
    std::unique_ptr<std::vector<float>> dummy_data_0,
    std::unique_ptr<std::vector<float>> dummy_data_1,
    std::unique_ptr<std::vector<float>> dummy_data_2,
    mlir::MLIRContext* context) {
  mlir::OpBuilder b(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(b, b.getUnknownLoc());
  mlir::ImplicitLocOpBuilder module_builder(module->getLoc(),
                                            module->getBodyRegion());

  auto io_type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(tensor_size)}, b.getF32Type());

  const auto func_type = module_builder.getFunctionType({io_type}, {io_type});
  auto func =
      mlir::func::FuncOp::create(module_builder, "add_float", func_type);

  auto block = func.addEntryBlock();
  module_builder.setInsertionPointToEnd(block);

  if (dialect == Dialect::LRT) {
    auto cst_attr = CreateAttr(*dummy_data_0, module_builder.getF32Type());
    auto cst_0_attr = CreateAttr(*dummy_data_1, module_builder.getF32Type());
    auto cst_1_attr = CreateAttr(*dummy_data_2, module_builder.getF32Type());

    auto cst = mlir::arith::ConstantOp::create(module_builder, cst_attr);
    auto cst_0 = mlir::arith::ConstantOp::create(module_builder, cst_0_attr);
    auto cst_1 = mlir::arith::ConstantOp::create(module_builder, cst_1_attr);

    auto add_3 =
        litert::AddOp::create(module_builder, cst_attr.getType(), cst, cst_0);
    auto add_4 =
        litert::AddOp::create(module_builder, cst_attr.getType(), add_3, cst_1);
    mlir::Value add_5 =
        litert::AddOp::create(module_builder, cst_attr.getType(), add_4, cst_1);

    mlir::func::ReturnOp::create(module_builder, add_5);
  } else if (dialect == Dialect::TFL) {
    auto cst_attr = CreateDenseAttr(*dummy_data_0, module_builder.getF32Type());
    auto cst_0_attr =
        CreateDenseAttr(*dummy_data_1, module_builder.getF32Type());
    auto cst_1_attr =
        CreateDenseAttr(*dummy_data_2, module_builder.getF32Type());

    auto cst = mlir::arith::ConstantOp::create(module_builder, cst_attr);
    auto cst_0 = mlir::arith::ConstantOp::create(module_builder, cst_0_attr);
    auto cst_1 = mlir::arith::ConstantOp::create(module_builder, cst_1_attr);

    auto add_3 = mlir::TFL::AddOp::create(module_builder, cst, cst_0,
                                          b.getStringAttr("NONE"));
    auto add_4 = mlir::TFL::AddOp::create(module_builder, add_3, cst_1,
                                          b.getStringAttr("NONE"));
    mlir::Value add_5 = mlir::TFL::AddOp::create(module_builder, add_4, cst_1,
                                                 b.getStringAttr("NONE"));

    delete dummy_data_0.release();
    delete dummy_data_1.release();
    delete dummy_data_2.release();
    mlir::func::ReturnOp::create(module_builder, add_5);
  }

  return module;
}

class Profiler {
 public:
  virtual ~Profiler() = default;
  virtual void Dump() = 0;
};

class CpuProfiler : public Profiler {
 public:
  explicit CpuProfiler(const std::string& test_name)
      : test_name_(test_name), start_time_(absl::Now()) {
    ProfilerStartCollecting(nullptr);
  }

  ~CpuProfiler() override {
    auto profile = ProfilerStopCollecting();
    std::string path = absl::StrCat("/tmp/", test_name_, ".pprof");
    CHECK(profile != nullptr);
    auto profile_st = perftools::profiles::MakeCpuProfile(
        std::move(profile), absl::Now() - start_time_);
    CHECK_OK(profile_st);
    CHECK(perftools::profiles::Builder::MarshalToFile(*profile_st.value(),
                                                      path.c_str()));
  }

  void Dump() override {}

 private:
  std::string test_name_;
  absl::Time start_time_;
};

class AllocRecorderProfiler : public Profiler {
 public:
  explicit AllocRecorderProfiler(const std::string& test_name)
      : test_name_(test_name) {
    CHECK(heap_profile::AllocRecorderStartWithMmapTracking(
        absl::StrCat("/tmp/", test_name_)));
  }

  ~AllocRecorderProfiler() override { heap_profile::AllocRecorderStop(); }

  void Dump() override { heap_profile::AllocRecorderDump(); }

 private:
  std::string test_name_;
};

class MallocProfiler : public Profiler {
 public:
  MallocProfiler(const std::string& test_name,
                 tcmalloc::ProfileType profile_type)
      : test_name_(test_name), profile_type_(profile_type) {}

  ~MallocProfiler() override {
    std::string path = absl::StrCat("/tmp/", test_name_, ".pprof");
    auto profile_or = perftools::profiles::CollectMallocProto(profile_type_);
    CHECK_OK(profile_or);
    auto profile = *std::move(profile_or);
    CHECK(perftools::profiles::Builder::MarshalToFile(*profile.get(),
                                                      path.c_str()));
  }

  void Dump() override {}

 private:
  std::string test_name_;
  tcmalloc::ProfileType profile_type_;
};

TEST_P(LazyDenseElementsAttrBenchmark, BenchmarkAddFold) {
  const BenchmarkTestCase& test_case = GetParam();
  auto tensor_size = test_case.tensor_size_mb / sizeof(float);
  auto profile_type = test_case.profile_type;
  auto dummy_data_0 = GenerateVector(tensor_size, 1.0);
  auto dummy_data_1 = GenerateVector(tensor_size, 2.0);
  auto dummy_data_2 = GenerateVector(tensor_size, 3.0);
  auto context = CreateContext(test_case.dialect);

  std::unique_ptr<Profiler> profiler;

  switch (profile_type) {
    case ProfileType::CPU:
      profiler = std::make_unique<CpuProfiler>(test_case.test_name);
      break;
    case ProfileType::PEAK_HEAP:
      profiler = std::make_unique<MallocProfiler>(
          test_case.test_name, tcmalloc::ProfileType::kPeakHeap);
      break;
    case ProfileType::ALLOC_RECORDER:
      profiler = std::make_unique<AllocRecorderProfiler>(test_case.test_name);
      break;
  }

  auto module = CreateProgram(test_case.dialect, tensor_size,
                              std::move(dummy_data_0), std::move(dummy_data_1),
                              std::move(dummy_data_2), context.get());
  auto pm = mlir::PassManager::on<mlir::ModuleOp>(context.get());
  pm.addPass(mlir::createCanonicalizerPass());
  profiler->Dump();
  if (mlir::failed(pm.run(*module))) {
    LOG(FATAL) << "Error while runnning Canonicalize pass.";
  }
  profiler->Dump();
}

constexpr uint64_t MB = 1024 * 1024;

INSTANTIATE_TEST_SUITE_P(
    LazyDenseElementsAttrBenchmarkSuiteInstantiation,
    LazyDenseElementsAttrBenchmark,
    ValuesIn<BenchmarkTestCase>({
        {"LRT_256M_CPU", Dialect::LRT, ProfileType::CPU, 256 * MB},
        {"LRT_256M_PEAK_HEAP", Dialect::LRT, ProfileType::PEAK_HEAP, 256 * MB},
        {"LRT_256M_ALLOC_RECORDER", Dialect::LRT, ProfileType::ALLOC_RECORDER,
         256 * MB},
        {"TFL_256M_CPU", Dialect::TFL, ProfileType::CPU, 256 * MB},
        {"TFL_256M_PEAK_HEAP", Dialect::TFL, ProfileType::PEAK_HEAP, 256 * MB},
        {"TFL_256M_ALLOC_RECORDER", Dialect::TFL, ProfileType::ALLOC_RECORDER,
         256 * MB},
    }),
    [](const testing::TestParamInfo<LazyDenseElementsAttrBenchmark::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace litert
