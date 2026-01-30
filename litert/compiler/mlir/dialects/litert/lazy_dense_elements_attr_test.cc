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
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"  // IWYU pragma: keep

namespace litert {
namespace {

using ::testing::ContainerEq;

TEST(LazyDenseElementsAttrTest, TestAddToConstantOp) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();

  mlir::OpBuilder b(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(b, b.getUnknownLoc());
  mlir::ImplicitLocOpBuilder module_builder(module->getLoc(),
                                            module->getBodyRegion());
  auto shape = mlir::RankedTensorType::get({1, 1}, module_builder.getI32Type());
  std::vector<int> buffer{0};
  auto attr = LazyDenseElementsAttr::get<int>(shape, buffer);
  mlir::arith::ConstantOp::create(module_builder,
                                  module_builder.getUnknownLoc(), attr);
  // TODO(aarfaian): add assertion
}

TEST(LazyDenseElementsAttrTest, TestGetTemplatized) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({1, 2}, b.getI32Type());
  std::vector<int> buffer{0, 0};
  auto attr = LazyDenseElementsAttr::get<int>(shape, buffer);

  EXPECT_EQ(attr.getType(), shape);
}

TEST(LazyDenseElementsAttrTest, TestGetInt64) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({1, 2, 3}, b.getI64Type());
  std::vector<int64_t> buffer{1, 2, 3, 4, 5, 6};
  auto attr = LazyDenseElementsAttr::get<int64_t>(shape, buffer);

  EXPECT_EQ(attr.getType(), shape);

  auto data_handle = attr.GetDataHandle();
  auto data = data_handle.GetDataAs<int64_t>();
  EXPECT_THAT(buffer, ContainerEq(data));
}

TEST(LazyDenseElementsAttrTest, TestGetFloat) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({1, 2, 3}, b.getF32Type());
  std::vector<float> buffer{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto attr = LazyDenseElementsAttr::get<float>(shape, buffer);

  EXPECT_EQ(attr.getType(), shape);

  auto data_handle = attr.GetDataHandle();
  auto data = data_handle.GetDataAs<float>();
  EXPECT_THAT(buffer, ContainerEq(data));
}

TEST(LazyDenseElementsAttrTest, TestGetDouble) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({1, 2, 3}, b.getF64Type());
  std::vector<double> buffer{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto attr = LazyDenseElementsAttr::get<double>(shape, buffer);

  EXPECT_EQ(attr.getType(), shape);

  auto data_handle = attr.GetDataHandle();
  auto data = data_handle.GetDataAs<double>();
  EXPECT_THAT(buffer, ContainerEq(data));
}

TEST(LazyDenseElementsAttrTest, TestGetNonTemplatized) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({1, 2}, b.getI32Type());
  std::vector<int> buffer{0, 0};
  llvm::ArrayRef<uint8_t> recast_buffer(
      reinterpret_cast<uint8_t*>(buffer.data()), buffer.size() * sizeof(int));
  auto attr = LazyDenseElementsAttr::get(shape, recast_buffer, alignof(int));

  EXPECT_EQ(attr.getType(), shape);
}

TEST(LazyDenseElementsAttrTest, TestPrint) {
  std::string expected = "#litert.lazy_dense<[[0,0]]> : tensor<1x2xi32>";
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({1, 2}, b.getI32Type());
  std::vector<int> buffer{0, 0};
  llvm::ArrayRef<uint8_t> recast_buffer(
      reinterpret_cast<uint8_t*>(buffer.data()), buffer.size() * sizeof(int));
  mlir::Attribute attr =
      LazyDenseElementsAttr::get(shape, recast_buffer, alignof(int));
  std::string result;
  llvm::raw_string_ostream ostream(result);

  attr.print(ostream);

  EXPECT_EQ(result, expected);
}

TEST(LazyDenseElementsAttrTest, TestPrintLargeElements) {
  mlir::registerAsmPrinterCLOptions();
  std::string expected = "#litert.lazy_dense<__elided__> : tensor<10x10xi32>";
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  auto shape = mlir::RankedTensorType::get({10, 10}, b.getI32Type());
  std::vector<int> buffer(100, 0);
  llvm::ArrayRef<uint8_t> recast_buffer(
      reinterpret_cast<uint8_t*>(buffer.data()), buffer.size() * sizeof(int));
  mlir::Attribute attr =
      LazyDenseElementsAttr::get(shape, recast_buffer, alignof(int));
  std::string result;
  llvm::raw_string_ostream ostream(result);

  attr.print(ostream);

  EXPECT_EQ(result, expected);
}

}  // namespace
}  // namespace litert
