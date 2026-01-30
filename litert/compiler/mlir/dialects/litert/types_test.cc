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
#include "litert/compiler/mlir/dialects/litert/types.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"

namespace litert {
namespace {

using ::testing::ContainerEq;
using ::testing::Eq;

TEST(TypesTest, TestSymTensorTypeGetShape) {
  mlir::DialectRegistry registry;
  registry.insert<litert::LITERTDialect>();
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);
  ctx->loadAllAvailableDialects();
  llvm::SmallVector<SymDimAttr, 4> shape;
  shape.emplace_back(SymDimAttr::get(ctx.get(), 1));
  shape.emplace_back(SymDimAttr::get(ctx.get(), 2));
  shape.emplace_back(SymDimAttr::get(ctx.get(), 4));
  auto element_type = mlir::Float32Type::get(ctx.get());

  auto srtt = SymTensorType::get(ctx.get(), shape, element_type, {});
  mlir::Type srtt_as_type = srtt;

  EXPECT_THAT(shape, ContainerEq(srtt.getShape()));
  EXPECT_TRUE(mlir::isa<SymTensorType>(srtt));
  EXPECT_TRUE(mlir::isa<SymTensorType>(srtt_as_type));
}

}  // namespace
}  // namespace litert
