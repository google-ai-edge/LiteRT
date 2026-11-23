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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_PATTERN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_PATTERN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"

// Region patterns and their matcher: a pattern describes a region of a graph
// (a small DAG of ops with predicates on its tensors) that a backend can run
// as one kernel, and the matcher finds every occurrence in a subgraph. Nothing
// here reads or writes files: the carve tool rewrites model files with the
// regions found, and a compiler plugin can call the matcher on the subgraphs it
// is asked to partition.
//
// A pattern description is text, one statement per line ("#" starts a
// comment):
//
//   pattern NAME                    starts a pattern; NAME names the composite
//   in V [: [const] TYPE]           an input of the region, in the composite's
//                                   input order; "const" inputs must be
//                                   constants and stay constants of the graph
//   out V                           an output of the region, in the composite's
//                                   output order
//   V[, V] = OP(A, ...) [{O = E, ...}] [: TYPE[, TYPE]]
//                                   an op of the region: OP is the TFLite
//                                   builtin operator name, each argument A the
//                                   name of a value or "_" (any constant), O
//                                   an option of the op (see OptionOf in
//                                   carve_pattern.cc) and TYPE a predicate on
//                                   the output of the same position
//   check E                         a condition the match must satisfy
//   attr K = E                      an attribute of the composite
//
//   TYPE = [ELEMENT][<SCALE>][SHAPE]
//     ELEMENT  f16 f32 f64 bf16 i2 i4 i8 i16 i32 i64 u4 u8 u16 u32 u64 bool
//     <SCALE>  the tensor is quantized per tensor with zero point 0; SCALE is
//              a symbol that takes the tensor's scale
//     SHAPE    [D, ...] where D is a number, "_" (any) or a symbol that takes
//              the dimension; [] is a scalar
//   A symbol has one value per match: every use after the first must agree.
//
//   E = numbers, "strings", true, false, symbols, [E, ...], + - * /, ==, !=,
//       parentheses, and the name of a constant with integer data, which is
//       the list of its elements.
//
// An argument that is neither an input nor the result of an op of the pattern
// is a constant of the region: it must be a constant tensor, and it moves
// into the decomposition. Optional inputs a model leaves out (the bias of a
// FULLY_CONNECTED without one) are not listed. Every result that is not an
// output must be used only inside the region. The last op statement is the
// anchor: matching starts at each op of its kind and follows the pattern's
// edges from there, so every statement must be connected to it.
namespace litert::tools {

// The value of a pattern expression: a composite attribute or a check operand.
using CarveValue =
    std::variant<bool, int64_t, double, std::string, std::vector<int64_t>>;

std::string CarveValueToString(const CarveValue& value);

// A predicate on a tensor's element type, quantization and shape.
struct CarveTypeSpec {
  struct Dim {
    enum class Kind { kAny, kLiteral, kSymbol };
    Kind kind = Kind::kAny;
    // The dimension for kLiteral, the index of the symbol for kSymbol.
    int64_t value = 0;
  };
  bool has_element_type = false;
  LiteRtElementType element_type = kLiteRtElementTypeNone;
  // The index of the symbol that takes the scale, or -1.
  int scale_symbol = -1;
  bool has_shape = false;
  std::vector<Dim> shape;
};

// An expression tree.
struct CarveExpr {
  enum class Kind { kLiteral, kSymbol, kConstant, kList, kBinary };
  Kind kind = Kind::kLiteral;
  CarveValue literal;
  // The index of the symbol (kSymbol) or of the value (kConstant).
  int ref = -1;
  // One of + - * / = (for ==) ! (for !=), for kBinary.
  char op = 0;
  std::vector<CarveExpr> args;
};

// A named tensor of a pattern.
struct CarveValueDecl {
  enum class Role {
    // An input of the region that is not required to be a constant.
    kInput,
    // An input of the region that must be a constant.
    kConstInput,
    // The result of an op of the pattern.
    kResult,
    // A constant used by an op of the pattern; it moves into the
    // decomposition.
    kConstant,
  };
  std::string name;
  Role role = Role::kConstant;
  bool has_type = false;
  CarveTypeSpec type;
};

struct CarveOpStmt {
  LiteRtOpCode op_code = kLiteRtOpCodeTflCustom;
  // Indices into CarvePattern::values; -1 is "_".
  std::vector<int> inputs;
  std::vector<int> outputs;
  std::vector<std::pair<std::string, CarveExpr>> options;
  // The line of the description, for messages.
  int line = 0;
};

// One step of the matching order: the statement to bind next, found from a
// tensor that is already bound.
struct CarvePlanStep {
  int stmt = 0;
  // The value the op is found from, and its position among the statement's
  // outputs (the op is the tensor's producer) or inputs (the op is one of the
  // tensor's users).
  int via_value = 0;
  bool via_output = true;
  int position = 0;
};

// A parsed pattern. ParseCarvePatterns makes these; they are plain data so
// that tests and reports can read them.
struct CarvePattern {
  // The composite's name.
  std::string name;
  std::vector<CarveValueDecl> values;
  std::vector<std::string> symbols;
  std::vector<CarveOpStmt> stmts;
  // Indices into values, in the composite's input and output order.
  std::vector<int> inputs;
  std::vector<int> outputs;
  std::vector<CarveExpr> checks;
  // The text of each check, for messages.
  std::vector<std::string> check_texts;
  std::vector<std::pair<std::string, CarveExpr>> attributes;
  // The statements after the anchor (the last one), in matching order.
  std::vector<CarvePlanStep> plan;
};

// Parses a description, which may hold several patterns.
absl::StatusOr<std::vector<CarvePattern>> ParseCarvePatterns(
    absl::string_view text);

// The description of the patterns that are compiled in.
absl::string_view BuiltinCarvePatternText();

// One occurrence of a pattern in a subgraph.
struct CarveRegion {
  const CarvePattern* pattern = nullptr;
  // The ops of the region, in the subgraph's order.
  std::vector<const LiteRtOpT*> ops;
  // The composite's inputs and outputs, in the pattern's order.
  std::vector<const LiteRtTensorT*> inputs;
  std::vector<const LiteRtTensorT*> outputs;
  // The composite's attributes, in the pattern's order.
  std::vector<std::pair<std::string, CarveValue>> attributes;
};

// A place where a pattern almost matched: at least a quarter of its ops were
// found, or all of them and a predicate failed.
struct CarveNearMiss {
  const CarvePattern* pattern = nullptr;
  // The op the anchor statement was tried on.
  const LiteRtOpT* anchor = nullptr;
  std::string reason;
};

struct CarveMatches {
  std::vector<CarveRegion> regions;
  std::vector<CarveNearMiss> near_misses;
};

// Finds the occurrences of `patterns` in `subgraph`. Regions do not overlap:
// an op belongs to the first region that claims it, in the order of the
// patterns and then of the subgraph's ops.
CarveMatches FindCarveRegions(absl::Span<const CarvePattern> patterns,
                              const LiteRtSubgraphT& subgraph);

// The flexbuffer map a composite stores its attributes in.
std::vector<uint8_t> PackCarveAttributes(
    const std::vector<std::pair<std::string, CarveValue>>& attributes);

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_PATTERN_H_
