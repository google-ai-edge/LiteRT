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
#include "litert/compiler/mlir/dialects/litert/traits.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/types.h"

namespace litert {

namespace {

struct EphemeralSymDim {
  int64_t size;
  mlir::SymbolRefAttr symbol = nullptr;

  bool IsDynamic() const { return mlir::ShapedType::isDynamic(size); }
  bool IsDynamicWithoutSymbol() const { return IsDynamic() && !symbol; }
  bool IsDynamicWithSymbol() const { return IsDynamic() && symbol; }

  bool operator==(const EphemeralSymDim& other) const {
    // In the case where either dim is dynamic but not symbolic we do not have
    // equality.
    if (mlir::ShapedType::isDynamic(size) ||
        mlir::ShapedType::isDynamic(other.size)) {
      if (!symbol && !other.symbol) {
        return false;
      }
    }
    return size == other.size && symbol == other.symbol;
  }
};

/// Returns the shape of the given type. Scalars will be considered as having a
/// shape with zero dimensions.
llvm::SmallVector<EphemeralSymDim> GetShape(mlir::Type type) {
  llvm::SmallVector<EphemeralSymDim, 4> new_shape;
  if (auto s_type = mlir::dyn_cast<mlir::ShapedType>(type)) {
    llvm::ArrayRef<int64_t> shape = s_type.getShape();
    for (const auto& s : shape) {
      new_shape.emplace_back(s);
    }
  } else if (auto s_type = mlir::dyn_cast<SymTensorType>(type)) {
    for (const auto& sym_dim : s_type.getShape()) {
      new_shape.emplace_back(sym_dim.getSize(), sym_dim.getSymbol());
    }
  }
  return new_shape;
}

bool GetBroadcastedShape(llvm::ArrayRef<EphemeralSymDim> shape1,
                         llvm::ArrayRef<EphemeralSymDim> shape2,
                         llvm::SmallVectorImpl<EphemeralSymDim>& result_shape) {
  // To compute the result broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working the
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.

  result_shape.clear();
  if (shape1.size() > shape2.size()) {
    llvm::append_range(result_shape, shape1);
  } else {
    llvm::append_range(result_shape, shape2);
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = result_shape.rbegin();

  for (; i1 != e1 && i2 != e2; ++i1, ++i2, ++iR) {
    // Both dimensions dynamic and symbolic
    // - If the symbols are equal then the result dimension is set to the
    //   symbol.
    if (i1->IsDynamicWithSymbol() && i2->IsDynamicWithSymbol() &&
        i1->symbol == i2->symbol) {
      *iR = *i1;
      continue;
    }

    // One dimension dynamic and the other is static
    // - If either dimension is static and equal to 1 while the other is
    //   dynamic then the resulting dimension is dynamic. This also propagates
    //   the symbol if present.
    if (i1->IsDynamic() || i2->IsDynamic()) {
      if (i1->size == 1) {
        *iR = *i2;
        continue;
      } else if (i2->size == 1) {
        *iR = *i1;
        continue;
      }
    }

    // Both dimensions static
    // - If both dimensions are equal then the result dimension is set to the
    //   size.
    // - If either dimension is equal to 1 then the result dimension is set to
    //   the opposite dimension.
    if (*i1 == *i2 || i2->size == 1) {
      *iR = *i1;
      continue;
    } else if (i1->size == 1) {
      *iR = *i2;
      continue;
    }

    // Not a supported case, e.g.
    // - Both dimensions are dynamic and one is symbolic while the other is not.
    // - Both dimensions are dynamic and symbolic but the symbols are not equal.
    // - Both dimensions are dynamic and not symbolic.
    // - One dimension is dynamic while the other is static and greater than 1.
    // - Both dimensions are static and greater than 1 and not equal.
    result_shape.clear();
    return false;
  }

  return true;
}

mlir::Type GetElementTypeOrSelf(mlir::Type type) {
  if (auto sym_tensor_type = mlir::dyn_cast_or_null<SymTensorType>(type)) {
    return sym_tensor_type.getElementType();
  }
  return mlir::getElementTypeOrSelf(type);
}

bool IsARankedOrSymTensorType(mlir::Type type) {
  return mlir::isa<mlir::RankedTensorType, SymTensorType>(type);
}

}  // namespace

mlir::Type GetBroadcastedType(mlir::Type type1, mlir::Type type2,
                              mlir::Type element_type) {
  if (!element_type) {
    // If the element_type is not specified, then the use the common element
    // type of the inputs or fail if there is no common element type.
    element_type = GetElementTypeOrSelf(type1);
    if (element_type != GetElementTypeOrSelf(type2)) {
      return {};
    }
  }

  // If one of the types is unranked tensor, then the other type shouldn't be
  // vector and the result should have unranked tensor type.
  if (mlir::isa<mlir::UnrankedTensorType>(type1) ||
      mlir::isa<mlir::UnrankedTensorType>(type2)) {
    if (mlir::isa<mlir::VectorType>(type1) ||
        mlir::isa<mlir::VectorType>(type2)) {
      return {};
    }

    return mlir::UnrankedTensorType::get(element_type);
  }

  // Returns the type kind if the given type is a vector, symbolic tensor, or
  // ranked tensor type. Returns std::nullopt otherwise.
  auto get_composite_type_kind =
      [](mlir::Type type) -> std::optional<mlir::TypeID> {
    if (mlir::isa<mlir::VectorType, mlir::RankedTensorType, SymTensorType>(
            type)) {
      return type.getTypeID();
    }

    return std::nullopt;
  };

  // Make sure the composite type, if has, is consistent.
  std::optional<mlir::TypeID> composite_kind_1 = get_composite_type_kind(type1);
  std::optional<mlir::TypeID> composite_kind_2 = get_composite_type_kind(type2);
  std::optional<mlir::TypeID> result_composite_kind;

  if (composite_kind_1 && composite_kind_2) {
    if (composite_kind_1 == composite_kind_2) {
      result_composite_kind = composite_kind_1;
    } else {
      // Handling heterogenous types here
      // Only allow mixing ranked tensor with symbolic tensor.
      if (!IsARankedOrSymTensorType(type1) &&
          !IsARankedOrSymTensorType(type2)) {
        return {};
      }
      if (composite_kind_1 == SymTensorType::getTypeID() ||
          composite_kind_2 == SymTensorType::getTypeID()) {
        result_composite_kind = SymTensorType::getTypeID();
      } else {
        result_composite_kind = composite_kind_1;
      }
    }
  } else if (composite_kind_1) {
    result_composite_kind = composite_kind_1;
  } else if (composite_kind_2) {
    result_composite_kind = composite_kind_2;
  }

  // Get the shape of each type.
  llvm::SmallVector<EphemeralSymDim, 4> result_shape;
  if (!GetBroadcastedShape(GetShape(type1), GetShape(type2), result_shape)) {
    return {};
  }

  // Compose the final broadcasted type
  if (result_composite_kind == mlir::VectorType::getTypeID()) {
    llvm::SmallVector<int64_t, 4> result_shape_i64;
    for (const auto& s : result_shape) {
      result_shape_i64.push_back(s.size);
    }
    return mlir::VectorType::get(result_shape_i64, element_type);
  }

  if (result_composite_kind == mlir::RankedTensorType::getTypeID()) {
    llvm::SmallVector<int64_t, 4> result_shape_i64;
    for (const auto& s : result_shape) {
      result_shape_i64.push_back(s.size);
    }
    return mlir::RankedTensorType::get(result_shape_i64, element_type);
  }

  if (result_composite_kind == SymTensorType::getTypeID()) {
    llvm::SmallVector<SymDimAttr, 4> result_shape_attr;
    for (const auto& d : result_shape) {
      if (d.IsDynamic()) {
        result_shape_attr.push_back(
            SymDimAttr::get(type1.getContext(), d.symbol));
      } else {
        result_shape_attr.push_back(
            SymDimAttr::get(type1.getContext(), d.size));
      }
    }
    return SymTensorType::get(result_shape_attr, element_type);
  }
  return element_type;
}

namespace impl {

namespace {

bool IsCompatibleInferredReturnShape(llvm::ArrayRef<EphemeralSymDim> inferred,
                                     llvm::ArrayRef<EphemeralSymDim> existing) {
  // If both inferred and existing dimensions are static, they must be equal.
  auto is_compatible = [](const EphemeralSymDim& inferred_dim,
                          const EphemeralSymDim& existing_dim) {
    return existing_dim.IsDynamic() || inferred_dim.IsDynamic() ||
           inferred_dim == existing_dim;
  };

  if (inferred.size() != existing.size()) {
    return false;
  }

  for (auto [inferred_dim, existing_dim] :
       llvm::zip_equal(inferred, existing)) {
    if (!is_compatible(inferred_dim, existing_dim)) {
      return false;
    }
  }

  return true;
}

std::string GetShapeString(llvm::ArrayRef<EphemeralSymDim> shape) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << '\'';
  llvm::interleave(
      shape, ss,
      [&](EphemeralSymDim dim) {
        if (dim.IsDynamic() && dim.symbol) {
          ss << dim.symbol;
        } else {
          ss << dim.size;
        }
      },
      "x");
  ss << '\'';
  return ret;
}

/// Returns a tuple corresponding to whether range has tensor or vector type.
template <typename iterator_range>
std::tuple<bool, bool> HasTensorOrVectorType(iterator_range types) {
  return {llvm::any_of(types, llvm::IsaPred<mlir::TensorType>),
          llvm::any_of(types, llvm::IsaPred<mlir::VectorType>)};
}

}  // namespace

llvm::LogicalResult VerifyOperandsBroadcastable(mlir::Operation* op) {
  // Ensure broadcasting only tensor or only vector types.
  auto operands_has_tensor_vector_type =
      HasTensorOrVectorType(op->getOperandTypes());
  auto results_has_tensor_vector_type =
      HasTensorOrVectorType(op->getResultTypes());
  if ((std::get<0>(operands_has_tensor_vector_type) ||
       std::get<0>(results_has_tensor_vector_type)) &&
      (std::get<1>(operands_has_tensor_vector_type) ||
       std::get<1>(results_has_tensor_vector_type))) {
    return op->emitError("cannot broadcast vector with tensor");
  }

  auto ranked_operands =
      make_filter_range(op->getOperandTypes(),
                        llvm::IsaPred<mlir::RankedTensorType, SymTensorType>);

  // If all operands are unranked, then all result shapes are possible.
  if (ranked_operands.empty()) return mlir::success();

  // Compute broadcasted shape of operands (which requires that operands are
  // broadcast compatible). The results need to be broadcast compatible with
  // this result shape.
  llvm::SmallVector<EphemeralSymDim, 4> result_shape;
  (void)GetBroadcastedShape(GetShape(*ranked_operands.begin()), {},
                            result_shape);
  auto it = ranked_operands.begin();
  if (it != ranked_operands.end()) {
    it++;
  }
  for (; it != ranked_operands.end(); it++) {
    llvm::SmallVector<EphemeralSymDim, 4> temp = result_shape;
    if (!GetBroadcastedShape(temp, GetShape(*it), result_shape))
      return op->emitOpError("operands don't have broadcast-compatible shapes");
  }

  auto ranked_results =
      make_filter_range(op->getResultTypes(),
                        llvm::IsaPred<mlir::RankedTensorType, SymTensorType>);

  // If all of the results are unranked then no further verification.
  if (ranked_results.empty()) return mlir::success();

  for (auto type : ranked_results) {
    auto s = GetShape(type);
    llvm::ArrayRef<EphemeralSymDim> shape = s;
    llvm::ArrayRef<EphemeralSymDim> actual_suffix =
        shape.take_back(result_shape.size());
    if (!IsCompatibleInferredReturnShape(result_shape, actual_suffix))
      return op->emitOpError()
             << "result type " << GetShapeString(GetShape(type))
             << " not broadcast compatible with broadcasted operands's shapes "
             << GetShapeString(result_shape);
  }
  return mlir::success();
}

}  // namespace impl
}  // namespace litert
