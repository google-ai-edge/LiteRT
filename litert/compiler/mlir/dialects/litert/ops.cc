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
#include "litert/compiler/mlir/dialects/litert/ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @com_google_absl
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"
#include "litert/compiler/mlir/dialects/litert/traits.h"
#include "litert/compiler/mlir/dialects/litert/types.h"
#include "litert/compiler/xnn_runner.h"

#define GET_OP_CLASSES
#include "litert/compiler/mlir/dialects/litert/ops.cc.inc"  // IWYU pragma: keep

namespace litert {

//===----------------------------------------------------------------------===//
// Pattern helpers
//===----------------------------------------------------------------------===//

static llvm::FailureOr<llvm::APInt> GetIntOrSplatIntValue(
    mlir::Attribute attr) {
  llvm::APInt value;
  if (mlir::matchPattern(attr, mlir::m_ConstantInt(&value))) return value;

  return mlir::failure();
}

static bool IsZero(mlir::Attribute attr) {
  auto splattr = mlir::dyn_cast_or_null<mlir::SplatElementsAttr>(attr);
  if (!splattr) {
    return false;
  }

  if (mlir::isa<mlir::FloatType>(splattr.getElementType())) {
    return splattr.getValues<llvm::APFloat>()[0].isZero();
  } else if (mlir::isa<mlir::IntegerType>(splattr.getElementType())) {
    return splattr.getValues<llvm::APInt>()[0].isZero();
  }

  return false;
}

static bool IsOne(mlir::Attribute attr) {
  auto splattr = mlir::dyn_cast_or_null<mlir::SplatElementsAttr>(attr);
  if (!splattr) {
    return false;
  }

  if (auto float_ty =
          mlir::dyn_cast_or_null<mlir::FloatType>(splattr.getElementType())) {
    auto one =
        llvm::APFloat::getOne(float_ty.getFloatSemantics(), /*Negative=*/false);
    return splattr.getValues<llvm::APFloat>()[0] == one;
  } else if (mlir::isa<mlir::IntegerType>(splattr.getElementType())) {
    return splattr.getValues<llvm::APInt>()[0].isOne();
  }

  return false;
}

namespace {

// Returns new shape with rank 'new_dims' with padded ones on the
// left if needed.
inline std::vector<int64_t> GetPaddedShape(llvm::ArrayRef<int64_t> old_shape,
                                           int new_dims) {
  std::vector<int64_t> new_shape(new_dims, 1);
  std::copy_backward(old_shape.begin(), old_shape.end(), new_shape.end());
  return new_shape;
}

// Helper method that given and 'current_index' representing
// index in broadcasted tensor, get the index in the flat original tensor.
// 'shape' is the original shape with padding to match result shape.
int64_t GetElementIndex(const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& current_index) {
  int64_t ind = 0;
  int64_t mul = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    ind += (current_index[i] % shape[i]) * mul;
    mul *= shape[i];
  }
  return ind;
}

// Helper method that increment index represented in 'current_index_ptr'
// in the shape of 'result_shape'.
void IncrementIndex(llvm::ArrayRef<int64_t> result_shape,
                    std::vector<int64_t>* current_index_ptr) {
  std::vector<int64_t>& current_index = *current_index_ptr;
  for (int i = result_shape.size() - 1; i >= 0; --i) {
    current_index[i]++;
    if (current_index[i] == result_shape[i]) {
      current_index[i] = 0;
    } else {
      break;
    }
  }
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the both operands are verified to have value
/// attributes of broadcastable types.
template <class DenseElementsT,
          class ElementT = typename DenseElementsT::ValueType,
          class ResultT = ElementT,
          class CalculationT = llvm::function_ref<ResultT(ElementT, ElementT)>>
mlir::Attribute ConstFoldBinaryOpDenseDense(mlir::ShapedType result_type,
                                            mlir::DenseElementsAttr lhs,
                                            mlir::DenseElementsAttr rhs,
                                            const CalculationT& calculate) {
  auto type = llvm::dyn_cast_or_null<mlir::ShapedType>(
      mlir::OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType()));
  if (!type) {
    return {};
  }

  type = type.clone(result_type.getElementType());

  const bool rhs_is_splat = rhs.isSplat();
  const bool lhs_is_splat = lhs.isSplat();

  auto lhs_values = lhs.try_value_begin<ElementT>();
  auto rhs_values = rhs.try_value_begin<ElementT>();
  if (failed(lhs_values) || failed(rhs_values)) {
    return {};
  }

  // If both of them are splat, compute and return.
  if (lhs_is_splat && rhs_is_splat) {
    return DenseElementsT::get(
        type, calculate(*lhs_values.value(), *rhs_values.value()));
  }

  auto num_elements = type.getNumElements();

  mlir::SmallVector<ResultT> new_values;
  new_values.reserve(num_elements);
  const auto result_shape = type.getShape();
  std::vector<int64_t> current_index(type.getRank(), 0);

  // Create the new shape with ones padded to the left.
  const auto lhs_new_shape =
      GetPaddedShape(lhs.getType().getShape(), type.getRank());
  const auto rhs_new_shape =
      GetPaddedShape(rhs.getType().getShape(), type.getRank());

  // Add each pair of the corresponding values in the dense elements
  // attributes.
  for (int64_t i = 0; i < num_elements; ++i) {
    // current_index represents the index
    // in the N-dimension tensor. GetElementIndex returns
    // the index in the flat representation of the original tensor
    // to use.
    const int64_t lhs_index =
        lhs_is_splat ? 0 : GetElementIndex(lhs_new_shape, current_index);
    const int64_t rhs_index =
        rhs_is_splat ? 0 : GetElementIndex(rhs_new_shape, current_index);

    new_values.push_back(calculate(*(lhs_values.value() + lhs_index),
                                   *(rhs_values.value() + rhs_index)));
    IncrementIndex(result_shape, &current_index);
  }
  return DenseElementsT::get(type, new_values);
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the two operands are verified to have value
/// attributes of broadcastable types.
template <class DenseElementsT,
          class ElementT = typename DenseElementsT::ValueType,
          class ResultT = ElementT,
          class CalculationT = llvm::function_ref<ResultT(ElementT, ElementT)>>
mlir::Attribute ConstFoldBinaryOp(mlir::ShapedType result_type,
                                  mlir::Attribute operand1,
                                  mlir::Attribute operand2,
                                  const CalculationT& calculate) {
  if (mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(operand1) &&
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(operand2)) {
    return ConstFoldBinaryOpDenseDense<DenseElementsT, ElementT, ResultT,
                                       CalculationT>(
        result_type, mlir::cast<mlir::DenseElementsAttr>(operand1),
        mlir::cast<mlir::DenseElementsAttr>(operand2), calculate);
  }

  // TODO: support other attribute kinds

  return {};
}

/// Performs const folding with broadcast behavior on the two attributes in
/// `operands` and returns the result if possible.
/// Depending on the given `resultType`, either `floatCalculate` or
/// `intCalculate` is chosen to conduct the calculate.
mlir::Attribute ConstFoldBinaryOp(
    mlir::ShapedType type, llvm::ArrayRef<mlir::Attribute> operands,
    llvm::function_ref<llvm::APFloat(llvm::APFloat, llvm::APFloat)>
        float_calculate,
    llvm::function_ref<llvm::APInt(llvm::APInt, llvm::APInt)> int_calculate) {
  auto elemType = type.getElementType();

  if (mlir::isa<mlir::FloatType>(elemType))
    return ConstFoldBinaryOp<mlir::DenseFPElementsAttr, llvm::APFloat,
                             llvm::APFloat>(type, operands[0], operands[1],
                                            float_calculate);

  if (elemType.isSignlessInteger())
    return ConstFoldBinaryOp<mlir::DenseIntElementsAttr, llvm::APInt,
                             llvm::APInt>(type, operands[0], operands[1],
                                          int_calculate);

  return {};
}

#include "litert/compiler/mlir/dialects/litert/canonicalize.inc"

}  // namespace

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult GetDimensionSizeOp::inferReturnTypes(
    mlir::MLIRContext* context, std::optional<mlir::Location> loc,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  inferred_return_types.push_back(
      mlir::RankedTensorType::get({1}, mlir::IntegerType::get(context, 32)));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

template <typename T>
  requires std::is_same_v<T, mlir::RankedTensorType> ||
           std::is_same_v<T, SymTensorType>
static llvm::LogicalResult TransposeOpInferReturnTypes(
    std::optional<mlir::Location> loc, T ranked_type,
    const std::vector<int64_t>& permutation,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  int64_t rank = ranked_type.getRank();
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return mlir::emitOptionalError(loc, "TransposeOp operand rank ", rank,
                                   " does not match permutation size ",
                                   permutation.size());
  }

  std::vector<int64_t> range(rank);
  std::iota(range.begin(), range.end(), 0);
  if (!std::is_permutation(range.begin(), range.end(), permutation.begin())) {
    return mlir::emitOptionalError(loc,
                                   "attribute permutation must be a permutation"
                                   " of [",
                                   range, "] but got ", permutation);
  }

  if constexpr (std::is_same_v<T, mlir::RankedTensorType>) {
    llvm::ArrayRef<int64_t> input_shape = ranked_type.getShape();

    llvm::SmallVector<int64_t> result_shape;
    for (int64_t dim : permutation) {
      result_shape.push_back(input_shape[dim]);
    }

    inferred_return_types.push_back(mlir::RankedTensorType::get(
        result_shape, ranked_type.getElementType()));
  } else if constexpr (std::is_same_v<T, SymTensorType>) {
    llvm::ArrayRef<SymDimAttr> input_shape = ranked_type.getShape();

    llvm::SmallVector<SymDimAttr> result_shape;
    for (int64_t dim : permutation) {
      result_shape.push_back(input_shape[dim]);
    }

    inferred_return_types.push_back(
        SymTensorType::get(result_shape, ranked_type.getElementType()));
  }

  return mlir::success();
}

llvm::LogicalResult TransposeOp::inferReturnTypes(
    mlir::MLIRContext*, std::optional<mlir::Location> loc,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);

  mlir::DenseElementsAttr permutation_attr;
  auto perm_is_constant =
      matchPattern(adaptor.getPermutation(), m_Constant(&permutation_attr));
  if (!(perm_is_constant && !permutation_attr.empty())) {
    return mlir::emitOptionalError(loc,
                                   "TransposeOp shape inference currently "
                                   "requires a constant permutation.");
  }

  std::vector<int64_t> permutation;
  for (const auto& p : permutation_attr.getValues<llvm::APInt>()) {
    permutation.push_back(p.getSExtValue());
  }

  if (auto ranked_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
          adaptor.getOperand().getType())) {
    return TransposeOpInferReturnTypes(loc, ranked_type, permutation,
                                       inferred_return_types);
  }

  if (auto symbolic_ranked_type = mlir::dyn_cast_or_null<SymTensorType>(
          adaptor.getOperand().getType())) {
    return TransposeOpInferReturnTypes(loc, symbolic_ranked_type, permutation,
                                       inferred_return_types);
  }

  return mlir::emitOptionalError(
      loc,
      "TransposeOp operand type must be RankedTensorType or "
      "SymTensorType.");
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult AddOp::inferReturnTypes(
    mlir::MLIRContext* context, std::optional<mlir::Location> loc,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  AddOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto result_type = GetBroadcastedType(adaptor.getLhs().getType(),
                                        adaptor.getRhs().getType());
  if (!result_type) {
    return mlir::emitOptionalError(loc, "non-broadcastable operands");
  }
  inferred_return_types.push_back(result_type);
  return mlir::success();
}

mlir::OpFoldResult AddFoldLazyDense(LazyDenseElementsAttr lhs,
                                    LazyDenseElementsAttr rhs,
                                    mlir::Type type) {
  auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
  std::vector<size_t> shape;
  for (auto dim : shaped_type.getShape()) {
    shape.push_back(dim);
  }

  auto lhs_data_handle = lhs.GetDataHandle();
  auto lhs_data = lhs_data_handle.GetDataAs<float>();

  auto rhs_data_handle = rhs.GetDataHandle();
  auto rhs_data = rhs_data_handle.GetDataAs<float>();
  std::vector<float> result(rhs_data.size());

  auto xnn_runner_or = XnnRunner::Create();
  if (!xnn_runner_or.ok()) {
    LOG(FATAL) << "Couldn't create XnnRunner";
  }
  auto xnn_runner = *std::move(xnn_runner_or);
  if (auto status = xnn_runner.BinaryAdd(shape, lhs_data, shape, rhs_data,
                                         shape, &result);
      !status.ok()) {
    LOG(FATAL) << "Couldn't run XnnRunner";
  }

  return LazyDenseElementsAttr::get<float>(shaped_type, result);
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs_ld = mlir::dyn_cast_or_null<LazyDenseElementsAttr>(adaptor.getLhs());
  auto rhs_ld = mlir::dyn_cast_or_null<LazyDenseElementsAttr>(adaptor.getRhs());

  if (lhs_ld && rhs_ld) {
    return AddFoldLazyDense(lhs_ld, rhs_ld, getType());
  }

  auto is_zero = [](mlir::Attribute a) {
    return matchPattern(a, mlir::m_Zero()) ||
           matchPattern(a, mlir::m_AnyZeroFloat());
  };

  // TODO: Handle quantized types.

  auto lhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getRhs());

  if (!lhs || !rhs) {
    return nullptr;
  }

  if (is_zero(lhs) && rhs.getType() == getType()) {
    return rhs;
  }

  if (is_zero(rhs) && lhs.getType() == getType()) {
    return lhs;
  }

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (auto elty = mlir::cast<mlir::ShapedType>(getType()).getElementType();
      elty.isFloat()) {
    // TODO: Implement this via LazyDenseElementsAttr.
    return nullptr;
  }

  bool overflow = false;

  // Generic fallback with APFloat
  auto result = ConstFoldBinaryOp(
      mlir::cast<mlir::ShapedType>(getType()), adaptor.getOperands(),
      [&](llvm::APFloat a, llvm::APFloat b) {
        if (b.isZero()) {
          return a;
        }
        if (a.add(b, llvm::APFloat::rmNearestTiesToEven) !=
            llvm::APFloat::opOK) {
          overflow = true;
        }
        return a;
      },
      [&](llvm::APInt a, llvm::APInt b) {
        if (overflow || !b) {
          return a;
        }
        return a.sadd_ov(b, overflow);
      });

  return overflow ? nullptr : result;
}

void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) {
  patterns.add<AddAddConstant, AddSubConstantRHS, AddSubConstantLHS,
               AddMulNegativeOneRhs, AddMulNegativeOneLhs>(context);
}

llvm::LogicalResult AddOp::verifySymbolUses(
    mlir::SymbolTableCollection& symbolTable) {
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult DivOp::inferReturnTypes(
    mlir::MLIRContext* context, std::optional<mlir::Location> loc,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  DivOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto result_type = GetBroadcastedType(adaptor.getLhs().getType(),
                                        adaptor.getRhs().getType());
  if (!result_type) {
    return mlir::emitOptionalError(loc, "non-broadcastable operands");
  }
  inferred_return_types.push_back(result_type);
  return mlir::success();
}

mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto is_zero = [](mlir::Attribute a) {
    return matchPattern(a, mlir::m_Zero()) ||
           matchPattern(a, mlir::m_AnyZeroFloat());
  };
  auto is_one = [](mlir::Attribute a) {
    return matchPattern(a, mlir::m_One()) ||
           matchPattern(a, mlir::m_OneFloat());
  };

  // TODO: Handle quantized types.

  auto lhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getRhs());

  // TODO: Handle LazyDenseElementsAttr.
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (is_zero(lhs) && lhs.getType() == getType()) {
    return lhs;
  }

  if (is_one(rhs) && getLhs().getType() == getType()) {
    return getLhs();
  }

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (auto elty = mlir::cast<mlir::ShapedType>(getType()).getElementType();
      elty.isFloat()) {
    // TODO: Implement this via LazyDenseElementsAttr.
    return nullptr;
  }

  bool overflow_or_div_0 = false;

  // Generic fallback with APFloat
  auto result = ConstFoldBinaryOp(
      mlir::cast<mlir::ShapedType>(getType()), adaptor.getOperands(),
      [&](llvm::APFloat a, llvm::APFloat b) {
        if (overflow_or_div_0 || b.isZero()) {
          overflow_or_div_0 = true;
          return a;
        }
        if (a.divide(b, llvm::APFloat::rmNearestTiesToEven) !=
            llvm::APFloat::opOK) {
          overflow_or_div_0 = true;
        }
        return a;
      },
      [&](llvm::APInt a, llvm::APInt b) {
        if (overflow_or_div_0 || !b) {
          overflow_or_div_0 = true;
          return a;
        }
        return a.sdiv_ov(b, overflow_or_div_0);
      });

  return overflow_or_div_0 ? nullptr : result;
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult MulOp::inferReturnTypes(
    mlir::MLIRContext* context, std::optional<mlir::Location> loc,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  MulOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto result_type = GetBroadcastedType(adaptor.getLhs().getType(),
                                        adaptor.getRhs().getType());
  if (!result_type) {
    return mlir::emitOptionalError(loc, "non-broadcastable operands");
  }
  inferred_return_types.push_back(result_type);
  return mlir::success();
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto is_zero = [](mlir::Attribute a) {
    return matchPattern(a, mlir::m_Zero()) ||
           matchPattern(a, mlir::m_AnyZeroFloat());
  };
  auto is_one = [](mlir::Attribute a) {
    return matchPattern(a, mlir::m_One()) ||
           matchPattern(a, mlir::m_OneFloat());
  };

  // TODO: Handle quantized types.

  auto lhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getRhs());

  // TODO: Handle LazyDenseElementsAttr.
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (lhs) {
    if (is_zero(lhs) && lhs.getType() == getType()) {
      return lhs;
    }
    if (is_one(lhs) && getRhs().getType() == getType()) {
      return getRhs();
    }
  }

  if (rhs) {
    if (is_zero(rhs) && rhs.getType() == getType()) {
      return rhs;
    }
    if (is_one(rhs) && getLhs().getType() == getType()) {
      return getLhs();
    }
  }

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (auto elty = mlir::cast<mlir::ShapedType>(getType()).getElementType();
      elty.isFloat()) {
    // TODO: Implement this via LazyDenseElementsAttr.
    return nullptr;
  }

  bool overflow = false;

  // Generic fallback with APFloat
  auto result = ConstFoldBinaryOp(
      mlir::cast<mlir::ShapedType>(getType()), adaptor.getOperands(),
      [&](llvm::APFloat a, llvm::APFloat b) {
        if (overflow || b.isZero()) {
          return a;
        }
        if (a.multiply(b, llvm::APFloat::rmNearestTiesToEven) !=
            llvm::APFloat::opOK) {
          overflow = true;
        }
        return a;
      },
      [&](llvm::APInt a, llvm::APInt b) {
        if (overflow || !b) {
          return a;
        }
        return a.smul_ov(b, overflow);
      });

  return overflow ? nullptr : result;
}

void MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                        mlir::MLIRContext* context) {
  results.add<MulMulConstant, MulZero, MulOne>(context);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                            mlir::MLIRContext* context) {
  results.add<FuseConsecutiveReshapeOps>(context);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult SubOp::inferReturnTypes(
    mlir::MLIRContext* context, std::optional<mlir::Location> loc,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferred_return_types) {
  SubOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto result_type = GetBroadcastedType(adaptor.getLhs().getType(),
                                        adaptor.getRhs().getType());
  if (!result_type) {
    return mlir::emitOptionalError(loc, "non-broadcastable operands");
  }
  inferred_return_types.push_back(result_type);
  return mlir::success();
}

mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto is_zero = [](mlir::Attribute a) {
    return matchPattern(a, mlir::m_Zero()) ||
           matchPattern(a, mlir::m_AnyZeroFloat());
  };

  // TODO: Handle quantized types.

  auto lhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getRhs());

  // TODO: Handle LazyDenseElementsAttr.
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (is_zero(rhs) && lhs.getType() == getType()) {
    return lhs;
  }

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (auto elty = mlir::cast<mlir::ShapedType>(getType()).getElementType();
      elty.isFloat()) {
    // TODO: Implement this via LazyDenseElementsAttr.
    return nullptr;
  }

  bool overflow = false;

  // Generic fallback with APFloat
  auto result = ConstFoldBinaryOp(
      mlir::cast<mlir::ShapedType>(getType()), adaptor.getOperands(),
      [&](llvm::APFloat a, llvm::APFloat b) {
        if (overflow || b.isZero()) {
          return a;
        }
        if (a.subtract(b, llvm::APFloat::rmNearestTiesToEven) !=
            llvm::APFloat::opOK) {
          overflow = true;
        }
        return a;
      },
      [&](llvm::APInt a, llvm::APInt b) {
        if (overflow || !b) {
          return a;
        }
        return a.ssub_ov(b, overflow);
      });

  return overflow ? nullptr : result;
}

void SubOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) {
  patterns.add<SubRHSAddConstant, SubLHSAddConstant, SubRHSSubConstantRHS,
               SubRHSSubConstantLHS, SubLHSSubConstantRHS, SubLHSSubConstantLHS,
               SubSubLHSRHSLHS>(context);
}

void LITERTDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "litert/compiler/mlir/dialects/litert/ops.cc.inc"  // IWYU pragma: keep
      >();
}

}  // namespace litert
