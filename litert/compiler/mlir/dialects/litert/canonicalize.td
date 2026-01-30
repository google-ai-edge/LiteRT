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
include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Constraints.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OpBase.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/PatternBase.td"
include "third_party/odml/litert/litert/compiler/mlir/dialects/litert/ops.td"

//===----------------------------------------------------------------------===//
// Pattern Helpers
//===----------------------------------------------------------------------===//

// Create zero attribute of type matching the argument's type.
def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr($0.getType())">;

def IsZero : Constraint<CPred<"IsZero($0)">>;

def IsOne : Constraint<CPred<"IsOne($0)">>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// add is commutative and associative for int values and will be canonicalized
// to have its constants appear as the second operand.

// add(add(x, c0), c1) -> add(x, add(c0, c1))
def AddAddConstant :
    Pat<(LITERT_AddOp
          (LITERT_AddOp $x, (Arith_ConstantOp:$c0 $_)),
          (Arith_ConstantOp:$c1 $_)),
        (LITERT_AddOp $x, (LITERT_AddOp $c0, $c1))>;

// add(sub(x, c0), c1) -> add(x, sub(c1, c0))
def AddSubConstantRHS :
    Pat<(LITERT_AddOp
          (LITERT_SubOp $x, (Arith_ConstantOp:$c0 $_)),
          (Arith_ConstantOp:$c1 $_)),
        (LITERT_AddOp $x, (LITERT_SubOp $c1, $c0))>;

// add(sub(c0, x), c1) -> sub(add(c0, c1), x)
def AddSubConstantLHS :
    Pat<(LITERT_AddOp
          (LITERT_SubOp (Arith_ConstantOp:$c0 $_), $x),
          (Arith_ConstantOp:$c1 $_)),
        (LITERT_SubOp (LITERT_AddOp $c0, $c1), $x)>;

def IsScalarOrSplatNegativeOne :
    Constraint<And<[
      CPred<"succeeded(GetIntOrSplatIntValue($0))">,
      CPred<"GetIntOrSplatIntValue($0)->isAllOnes()">]>>;

// add(x, mul(y, -1)) -> sub(x, y)
def AddMulNegativeOneRhs :
    Pat<(LITERT_AddOp
           $x,
           (LITERT_MulOp $y, (ConstantLikeMatcher AnyAttr:$c0))),
        (LITERT_SubOp $x, $y),
        [(IsScalarOrSplatNegativeOne $c0)]>;

// add(mul(x, -1), y) -> sub(y, x)
def AddMulNegativeOneLhs :
    Pat<(LITERT_AddOp
           (LITERT_MulOp $x, (ConstantLikeMatcher AnyAttr:$c0)),
           $y),
        (LITERT_SubOp $y, $x),
        [(IsScalarOrSplatNegativeOne $c0)]>;

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

// mul(mul(x, c0), c1) -> mul(x, mul(c0, c1))
def MulMulConstant : Pat<
  (LITERT_MulOp
    (LITERT_MulOp $x, (Arith_ConstantOp:$c0 $_)),
    (Arith_ConstantOp:$c1 $_)),
  (LITERT_MulOp $x, (LITERT_MulOp $c0, $c1))>;

// mul(x, 0) -> 0
def MulZero : Pat<
  (LITERT_MulOp $x, (Arith_ConstantOp:$y $cst)),
  (replaceWithValue $y),
  [(IsZero $cst)]>;

// mul(x, 1) -> x
def MulOne : Pat<
  (LITERT_MulOp $x, (Arith_ConstantOp $c0)),
  (replaceWithValue $x),
  [(IsOne $c0)]>;

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def FuseConsecutiveReshapeOps : Pat<
  (LITERT_ReshapeOp
    (LITERT_ReshapeOp $input, $_),
    $target_shape
  ),
  (LITERT_ReshapeOp $input, $target_shape)>;

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

// sub(add(x, c0), c1) -> add(x, sub(c0, c1))
def SubRHSAddConstant :
    Pat<(LITERT_SubOp
          (LITERT_AddOp $x, (Arith_ConstantOp:$c0 $_)),
          (Arith_ConstantOp:$c1 $_)),
        (LITERT_AddOp $x, (LITERT_SubOp $c0, $c1))>;

// sub(c1, add(x, c0)) -> sub(sub(c1, c0), x)
def SubLHSAddConstant :
    Pat<(LITERT_SubOp
          (Arith_ConstantOp:$c1 $_),
          (LITERT_AddOp $x, (Arith_ConstantOp:$c0 $_))),
        (LITERT_SubOp (LITERT_SubOp $c1, $c0), $x)>;

// sub(sub(x, c0), c1) -> sub(x, add(c0, c1))
def SubRHSSubConstantRHS :
    Pat<(LITERT_SubOp
          (LITERT_SubOp $x, (Arith_ConstantOp:$c0 $_)),
          (Arith_ConstantOp:$c1 $_)),
        (LITERT_SubOp $x, (LITERT_AddOp $c0, $c1))>;

// sub(sub(c0, x), c1) -> sub(sub(c0, c1), x)
def SubRHSSubConstantLHS :
    Pat<(LITERT_SubOp
          (LITERT_SubOp (Arith_ConstantOp:$c0 $_), $x),
          (Arith_ConstantOp:$c1 $_)),
        (LITERT_SubOp (LITERT_SubOp $c0, $c1), $x)>;

// sub(c1, sub(x, c0)) -> sub(add(c0, c1), x)
def SubLHSSubConstantRHS :
    Pat<(LITERT_SubOp
          (Arith_ConstantOp:$c1 $_),
          (LITERT_SubOp $x, (Arith_ConstantOp:$c0 $_))),
        (LITERT_SubOp (LITERT_AddOp $c0, $c1), $x)>;

// sub(c1, sub(c0, x)) -> add(x, sub(c1, c0))
def SubLHSSubConstantLHS :
    Pat<(LITERT_SubOp
          (Arith_ConstantOp:$c1 $_),
          (LITERT_SubOp (Arith_ConstantOp:$c0 $_), $x)),
        (LITERT_AddOp $x, (LITERT_SubOp $c1, $c0))>;

// sub(sub(a, b), a) -> sub(0, b)
def SubSubLHSRHSLHS :
    Pat<(LITERT_SubOp (LITERT_SubOp $x, $y), $x),
        (LITERT_SubOp (Arith_ConstantOp (GetZeroAttr $y)), $y)>;
