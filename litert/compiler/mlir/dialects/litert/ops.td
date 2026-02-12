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
#ifndef LITERT_OPS
#define LITERT_OPS

include "litert/compiler/mlir/dialects/litert/dialect.td"
include "litert/compiler/mlir/dialects/litert/type_constraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// TODO(aarfaian): move traits to a more appropriate location
def BroadcastableOperands : NativeOpTrait<"BroadcastableOperands">;

class LITERT_Op<string mnemonic, list<Trait> traits = []> : Op<LITERT_Dialect, mnemonic, traits> {
}

class LITERT_BinaryOp<string mnemonic, list<Trait> traits = []> :
  LITERT_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins LITERT_Tensor:$lhs, LITERT_Tensor:$rhs);
  let results = (outs LITERT_Tensor:$output);
}

class LITERT_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
  LITERT_BinaryOp<mnemonic, traits> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

class LITERT_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
  LITERT_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins LITERT_Tensor:$input);
  let results = (outs LITERT_Tensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($output)
  }];
}

def LITERT_AbsOp : LITERT_ElementwiseUnaryOp<"abs">;

def LITERT_AddOp : LITERT_ElementwiseBinaryOp<"add", [
    Pure,
    Commutative,
    BroadcastableOperands,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def LITERT_AndOp : LITERT_ElementwiseBinaryOp<"and">;

def LITERT_BatchMatmulOp : LITERT_BinaryOp<"batch_matmul"> {
  let summary = "Batch Matrix Multiplication operation";

  let arguments = (ins
    LITERT_Tensor:$lhs,
    LITERT_Tensor:$rhs,

    DefaultValuedAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedAttr<BoolAttr, "false">:$adj_y
  );

  let results = (outs
    LITERT_Tensor:$output
  );

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def LITERT_BroadcastInDimOp : LITERT_Op<"broadcast_in_dim">,
    Arguments<(ins LITERT_Tensor:$operand,
                   DenseI64ArrayAttr:$dims)>,
    Results<(outs LITERT_Tensor:$output)> {
}

def LITERT_BroadcastToOp : LITERT_Op<"broadcast_to", [Pure]>,
    Arguments<(ins LITERT_Tensor:$operand,
                   LITERT_I64Tensor:$shape)>,
    Results<(outs LITERT_Tensor:$output)> {
}

def LITERT_CastOp : LITERT_Op<"cast"> {
  let arguments = (ins LITERT_Tensor:$x,
                       TypeAttrOf<LITERT_AllowedTensorElementTypes>:$dtype);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_CompositeOp : LITERT_Op<"composite"> {
  let arguments = (ins Variadic<LITERT_Tensor>:$inputs,
                       StrAttr:$name,
                       DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$composite_attributes,
                       FlatSymbolRefAttr:$decomposition,
                       DefaultValuedOptionalAttr<I32Attr, "0">:$version);
  let results = (outs Variadic<LITERT_Tensor>:$result);
}

def LITERT_ConcatOp : LITERT_Op<"concat"> {
  let arguments = (ins Variadic<LITERT_Tensor>:$tensors,
                       I64Attr:$dim);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_CosOp : LITERT_ElementwiseUnaryOp<"cos">;

def LITERT_DivOp : LITERT_ElementwiseBinaryOp<"div", [
    Pure,
    BroadcastableOperands,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let hasFolder = 1;
}

def LITERT_DynamicUpdateSliceOp : LITERT_Op<"dynamic_update_slice"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       LITERT_Tensor:$update,
                       Variadic<LITERT_Tensor>:$start_indices);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_EqOp : LITERT_ElementwiseBinaryOp<"eq">;

def LITERT_ExpOp : LITERT_ElementwiseUnaryOp<"exp">;

def LITERT_FillOp : LITERT_Op<"fill"> {
  let arguments = (ins LITERT_I64Tensor:$dims,
                       LITERT_Tensor:$value);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_ExpandDimsOp : LITERT_Op<"expand_dims", [Pure]> {
  let arguments = (ins LITERT_Tensor:$input, LITERT_I64Tensor:$dim);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_GeOp : LITERT_ElementwiseBinaryOp<"ge">;

def LITERT_GeluOp : LITERT_Op<"gelu", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins LITERT_Tensor:$input,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate);
  let results = (outs LITERT_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($output)
  }];
}

def LITERT_GatherOp : LITERT_Op<"gather"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       LITERT_Tensor:$indices,
                       I64Attr:$axis,
                       I64Attr:$batch_dims);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_GtOp : LITERT_ElementwiseBinaryOp<"gt">;

def LITERT_IotaOp : LITERT_Op<"iota">,
    Arguments<(ins I64Attr:$dimension)>,
    Results<(outs LITERT_Tensor:$output)> {

  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
}

def LITERT_LogOp : LITERT_ElementwiseUnaryOp<"log">;

def LITERT_LogisticOp : LITERT_ElementwiseUnaryOp<"logistic">;

def LITERT_LtOp : LITERT_ElementwiseBinaryOp<"lt">;

def LITERT_MaxOp : LITERT_ElementwiseBinaryOp<"max">;

def LITERT_MinOp : LITERT_ElementwiseBinaryOp<"min">;

def LITERT_MeanOp : LITERT_Op<"mean"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       LITERT_I64Tensor:$axes,
                       DefaultValuedAttr<BoolAttr, "false">:$keep_dims);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_MulOp : LITERT_ElementwiseBinaryOp<"mul", [
    Pure,
    Commutative,
    BroadcastableOperands,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def LITERT_NeOp : LITERT_ElementwiseBinaryOp<"ne">;

def LITERT_NegOp : LITERT_ElementwiseUnaryOp<"neg">;

def LITERT_PowOp : LITERT_ElementwiseBinaryOp<"pow">;

def LITERT_RangeOp : LITERT_Op<"range"> {
  let arguments = (ins LITERT_Tensor:$start,
                       LITERT_Tensor:$limit,
                       LITERT_Tensor:$delta);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_ReduceMaxOp : LITERT_Op<"reduce_max"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       LITERT_I64Tensor:$axes);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_ReduceSumOp : LITERT_Op<"reduce_sum"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       LITERT_I64Tensor:$axes);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_RemOp : LITERT_ElementwiseBinaryOp<"rem">;

def LITERT_ReshapeOp : LITERT_Op<"reshape", [Pure]>,
    Arguments<(ins LITERT_Tensor:$operand,
                   LITERT_I64Tensor:$new_sizes)>,
    Results<(outs LITERT_Tensor:$output)> {
  let hasCanonicalizer = 1;
}

def LITERT_RsqrtOp : LITERT_ElementwiseUnaryOp<"rsqrt">;

def LITERT_SelectNOp : LITERT_Op<"select_n"> {
  let arguments = (ins LITERT_Tensor:$which,
                       Variadic<LITERT_Tensor>:$cases);
  let results = (outs LITERT_Tensor:$output);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def LITERT_SignOp : LITERT_ElementwiseUnaryOp<"sign">;

def LITERT_SinOp : LITERT_ElementwiseUnaryOp<"sin">;

def LITERT_SliceOp : LITERT_Op<"slice"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       LITERT_I64Tensor:$start_indices,
                       LITERT_I64Tensor:$limit_indices,
                       LITERT_I64Tensor:$strides);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_SoftmaxOp : LITERT_ElementwiseUnaryOp<"softmax">;

def LITERT_SplitOp : LITERT_Op<"split"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       DenseI64ArrayAttr:$sizes,
                       I64Attr:$axis);
  let results = (outs Variadic<LITERT_Tensor>:$result);
}

def LITERT_SqrtOp : LITERT_ElementwiseUnaryOp<"sqrt">;

def LITERT_SquareOp : LITERT_ElementwiseUnaryOp<"square">;

def LITERT_SqueezeOp : LITERT_Op<"squeeze"> {
  let arguments = (ins LITERT_Tensor:$operand,
                       DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$dimensions);
  let results = (outs LITERT_Tensor:$output);
}

def LITERT_SubOp : LITERT_ElementwiseBinaryOp<"sub", [
    Pure,
    BroadcastableOperands,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}


def LITERT_TanhOp : LITERT_ElementwiseUnaryOp<"tanh">;

def LITERT_TransposeOp : LITERT_Op<"transpose", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>,
    Arguments<(ins LITERT_Tensor:$operand,
                   LITERT_I64Tensor:$permutation)>,
    Results<(outs LITERT_Tensor:$output)> {
}

def LITERT_DotGeneralOp : LITERT_Op<"dot_general">,
    Arguments<(ins LITERT_Tensor:$lhs,
                   LITERT_Tensor:$rhs,
                   DenseI64ArrayAttr:$lhs_batching_dims,
                   DenseI64ArrayAttr:$rhs_batching_dims,
                   DenseI64ArrayAttr:$lhs_contracting_dims,
                   DenseI64ArrayAttr:$rhs_contracting_dims)>,
    Results<(outs LITERT_Tensor:$output)> {
}

def LITERT_GetDimensionSizeOp: LITERT_Op<"get_dimension_size", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins
    LITERT_Tensor:$operand,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension
  );

  let results = (outs LITERT_Tensor);
}

def LITERT_BindSymbolOp : LITERT_Op<"bind_symbol", []> {
  let arguments = (ins
    LITERT_Tensor:$dim_size,
    SymbolNameAttr:$sym_name);

  let extraClassDeclaration = [{
    // mlir::StringRef getName() { return getSymName(); }
    //mlir::FlatSymbolRefAttr getFlatSymbolRefAttr() { return mlir::FlatSymbolRefAttr::get(sym_name); }

    bool isDeclaration() { return false; }
  }];
}


#endif // LITERT
