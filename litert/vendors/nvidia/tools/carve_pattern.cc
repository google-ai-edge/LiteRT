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

#include "litert/vendors/nvidia/tools/carve_pattern.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools {
namespace {

using ::litert::internal::IsConstant;

// A constant's elements are read into a list for checks and attributes; a
// longer one is weights, not a parameter.
constexpr size_t kMaxConstantElements = 1024;

const char* OpName(LiteRtOpCode code) {
  return tflite::EnumNameBuiltinOperator(
      static_cast<tflite::BuiltinOperator>(code));
}

struct ElementTypeName {
  absl::string_view name;
  LiteRtElementType type;
};

constexpr ElementTypeName kElementTypeNames[] = {
    {"f16", kLiteRtElementTypeFloat16}, {"f32", kLiteRtElementTypeFloat32},
    {"f64", kLiteRtElementTypeFloat64}, {"bf16", kLiteRtElementTypeBFloat16},
    {"i2", kLiteRtElementTypeInt2},     {"i4", kLiteRtElementTypeInt4},
    {"i8", kLiteRtElementTypeInt8},     {"i16", kLiteRtElementTypeInt16},
    {"i32", kLiteRtElementTypeInt32},   {"i64", kLiteRtElementTypeInt64},
    {"u4", kLiteRtElementTypeUInt4},    {"u8", kLiteRtElementTypeUInt8},
    {"u16", kLiteRtElementTypeUInt16},  {"u32", kLiteRtElementTypeUInt32},
    {"u64", kLiteRtElementTypeUInt64},  {"bool", kLiteRtElementTypeBool},
};

std::optional<LiteRtElementType> ElementTypeOf(absl::string_view name) {
  for (const auto& entry : kElementTypeNames) {
    if (entry.name == name) return entry.type;
  }
  return std::nullopt;
}

absl::string_view ElementTypeNameOf(LiteRtElementType type) {
  for (const auto& entry : kElementTypeNames) {
    if (entry.type == type) return entry.name;
  }
  return "?";
}

std::optional<LiteRtOpCode> OpCodeOf(absl::string_view name) {
  const char* const* names = tflite::EnumNamesBuiltinOperator();
  for (const auto code : tflite::EnumValuesBuiltinOperator()) {
    if (name == names[code]) return static_cast<LiteRtOpCode>(code);
  }
  return std::nullopt;
}

// EXPRESSION VALUES

std::optional<double> AsNumber(const CarveValue& value) {
  if (const auto* i = std::get_if<int64_t>(&value)) {
    return static_cast<double>(*i);
  }
  if (const auto* d = std::get_if<double>(&value)) return *d;
  return std::nullopt;
}

bool ValuesEqual(const CarveValue& a, const CarveValue& b) {
  if (a.index() == b.index()) return a == b;
  const auto x = AsNumber(a);
  const auto y = AsNumber(b);
  return x && y && *x == *y;
}

absl::StatusOr<CarveValue> Arithmetic(char op, const CarveValue& a,
                                      const CarveValue& b) {
  const auto* i = std::get_if<int64_t>(&a);
  const auto* j = std::get_if<int64_t>(&b);
  if (i != nullptr && j != nullptr) {
    switch (op) {
      case '+':
        return CarveValue(*i + *j);
      case '-':
        return CarveValue(*i - *j);
      case '*':
        return CarveValue(*i * *j);
      default:
        // A size that does not divide is a different region, not a rounded
        // one.
        if (*j == 0 || *i % *j != 0) {
          return absl::FailedPreconditionError(
              absl::StrCat(*i, " / ", *j, " is not a whole number"));
        }
        return CarveValue(*i / *j);
    }
  }
  const auto x = AsNumber(a);
  const auto y = AsNumber(b);
  if (!x || !y) {
    return absl::InvalidArgumentError(
        absl::StrCat("'", std::string(1, op), "' needs numbers, got ",
                     CarveValueToString(a), " and ", CarveValueToString(b)));
  }
  switch (op) {
    case '+':
      return CarveValue(*x + *y);
    case '-':
      return CarveValue(*x - *y);
    case '*':
      return CarveValue(*x * *y);
    default:
      return CarveValue(*x / *y);
  }
}

// OP OPTIONS
//
// The options a pattern can constrain. A pattern that needs another one adds a
// line here.

template <class Options>
const Options& OrDefault(const Options* options) {
  static const Options* const kDefault = new Options();
  return options != nullptr ? *options : *kDefault;
}

std::string Activation(tflite::ActivationFunctionType type) {
  return tflite::EnumNameActivationFunctionType(type);
}

absl::StatusOr<CarveValue> OptionOf(const LiteRtOpT& op,
                                    absl::string_view name) {
  const auto& options = litert::internal::GetTflOptions(op);
  switch (op.OpCode()) {
    case kLiteRtOpCodeTflFullyConnected: {
      const auto& o = OrDefault(options.AsFullyConnectedOptions());
      if (name == "fused_activation_function") {
        return CarveValue(Activation(o.fused_activation_function));
      }
      if (name == "keep_num_dims") return CarveValue(o.keep_num_dims);
      if (name == "asymmetric_quantize_inputs") {
        return CarveValue(o.asymmetric_quantize_inputs);
      }
      if (name == "weights_format") {
        return CarveValue(
            std::string(tflite::EnumNameFullyConnectedOptionsWeightsFormat(
                o.weights_format)));
      }
      break;
    }
    case kLiteRtOpCodeTflConcatenation: {
      const auto& o = OrDefault(options.AsConcatenationOptions());
      if (name == "axis") return CarveValue(static_cast<int64_t>(o.axis));
      if (name == "fused_activation_function") {
        return CarveValue(Activation(o.fused_activation_function));
      }
      break;
    }
    case kLiteRtOpCodeTflBatchMatmul: {
      const auto& o = OrDefault(options.AsBatchMatMulOptions());
      if (name == "adj_x") return CarveValue(o.adj_x);
      if (name == "adj_y") return CarveValue(o.adj_y);
      if (name == "asymmetric_quantize_inputs") {
        return CarveValue(o.asymmetric_quantize_inputs);
      }
      break;
    }
    case kLiteRtOpCodeTflAdd:
      if (name == "fused_activation_function") {
        return CarveValue(Activation(
            OrDefault(options.AsAddOptions()).fused_activation_function));
      }
      break;
    case kLiteRtOpCodeTflSub:
      if (name == "fused_activation_function") {
        return CarveValue(Activation(
            OrDefault(options.AsSubOptions()).fused_activation_function));
      }
      break;
    case kLiteRtOpCodeTflMul:
      if (name == "fused_activation_function") {
        return CarveValue(Activation(
            OrDefault(options.AsMulOptions()).fused_activation_function));
      }
      break;
    case kLiteRtOpCodeTflDiv:
      if (name == "fused_activation_function") {
        return CarveValue(Activation(
            OrDefault(options.AsDivOptions()).fused_activation_function));
      }
      break;
    default:
      break;
  }
  return absl::NotFoundError(
      absl::StrCat(OpName(op.OpCode()), " has no option '", name,
                   "' the matcher knows (see OptionOf in carve_pattern.cc)"));
}

// PARSING

struct Token {
  enum class Kind { kEnd, kName, kInt, kFloat, kString, kPunct };
  Kind kind = Kind::kEnd;
  // The name, the string's contents or the punctuation.
  std::string text;
  int64_t int_value = 0;
  double float_value = 0;
};

absl::Status Tokenize(absl::string_view line, std::vector<Token>* tokens) {
  size_t i = 0;
  while (i < line.size()) {
    const char c = line[i];
    if (c == '#') break;
    if (absl::ascii_isspace(c)) {
      ++i;
      continue;
    }
    Token token;
    if (absl::ascii_isalpha(c) || c == '_') {
      size_t end = i + 1;
      while (end < line.size() && (absl::ascii_isalnum(line[end]) ||
                                   line[end] == '_' || line[end] == '.')) {
        ++end;
      }
      token.kind = Token::Kind::kName;
      token.text = std::string(line.substr(i, end - i));
      i = end;
    } else if (absl::ascii_isdigit(c)) {
      size_t end = i + 1;
      bool is_float = false;
      while (end < line.size()) {
        const char d = line[end];
        if (absl::ascii_isdigit(d)) {
          ++end;
        } else if (d == '.' || d == 'e' || d == 'E') {
          is_float = true;
          ++end;
          if ((d == 'e' || d == 'E') && end < line.size() &&
              (line[end] == '-' || line[end] == '+')) {
            ++end;
          }
        } else {
          break;
        }
      }
      const absl::string_view text = line.substr(i, end - i);
      if (is_float) {
        token.kind = Token::Kind::kFloat;
        if (!absl::SimpleAtod(text, &token.float_value)) {
          return absl::InvalidArgumentError(
              absl::StrCat("bad number '", text, "'"));
        }
      } else {
        token.kind = Token::Kind::kInt;
        if (!absl::SimpleAtoi(text, &token.int_value)) {
          return absl::InvalidArgumentError(
              absl::StrCat("bad number '", text, "'"));
        }
      }
      i = end;
    } else if (c == '"') {
      const size_t end = line.find('"', i + 1);
      if (end == absl::string_view::npos) {
        return absl::InvalidArgumentError("unterminated string");
      }
      token.kind = Token::Kind::kString;
      token.text = std::string(line.substr(i + 1, end - i - 1));
      i = end + 1;
    } else if ((c == '=' || c == '!') && i + 1 < line.size() &&
               line[i + 1] == '=') {
      token.kind = Token::Kind::kPunct;
      token.text = std::string(line.substr(i, 2));
      i += 2;
    } else if (absl::string_view("=(),:[]{}<>+-*/").find(c) !=
               absl::string_view::npos) {
      token.kind = Token::Kind::kPunct;
      token.text = std::string(1, c);
      ++i;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("unexpected character '", std::string(1, c), "'"));
    }
    tokens->push_back(std::move(token));
  }
  tokens->emplace_back();
  return absl::OkStatus();
}

// Parses the statements of one line into the pattern being built.
class LineParser {
 public:
  LineParser(CarvePattern* pattern, const std::vector<Token>& tokens, int line)
      : pattern_(*pattern), tokens_(tokens), line_(line) {}

  absl::Status ParseIn() {
    ++pos_;
    std::string name;
    if (auto status = TakeName(&name); !status.ok()) return status;
    const int value = ValueIndex(name);
    auto& decl = pattern_.values[value];
    if (std::find(pattern_.inputs.begin(), pattern_.inputs.end(), value) !=
        pattern_.inputs.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("input '", name, "' is declared twice"));
    }
    decl.role = CarveValueDecl::Role::kInput;
    if (TakePunct(":")) {
      if (Peek().kind == Token::Kind::kName && Peek().text == "const") {
        ++pos_;
        decl.role = CarveValueDecl::Role::kConstInput;
      }
      if (Peek().kind != Token::Kind::kEnd) {
        if (auto status = ParseType(&decl.type); !status.ok()) return status;
        decl.has_type = true;
      }
    }
    pattern_.inputs.push_back(value);
    return ExpectEnd();
  }

  absl::Status ParseOut() {
    ++pos_;
    std::string name;
    if (auto status = TakeName(&name); !status.ok()) return status;
    pattern_.outputs.push_back(ValueIndex(name));
    return ExpectEnd();
  }

  absl::Status ParseCheck(absl::string_view text) {
    ++pos_;
    CarveExpr expr;
    if (auto status = ParseExpr(&expr); !status.ok()) return status;
    pattern_.checks.push_back(std::move(expr));
    pattern_.check_texts.push_back(
        std::string(absl::StripAsciiWhitespace(text)));
    return ExpectEnd();
  }

  absl::Status ParseAttr() {
    ++pos_;
    std::string key;
    if (auto status = TakeName(&key); !status.ok()) return status;
    if (!TakePunct("=")) return Missing("'='");
    CarveExpr expr;
    if (auto status = ParseExpr(&expr); !status.ok()) return status;
    for (const auto& [existing, unused] : pattern_.attributes) {
      if (existing == key) {
        return absl::InvalidArgumentError(
            absl::StrCat("attribute '", key, "' is set twice"));
      }
    }
    pattern_.attributes.emplace_back(std::move(key), std::move(expr));
    return ExpectEnd();
  }

  // V[, V] = OP(A, ...) [{O = E, ...}] [: TYPE[, TYPE]]
  absl::Status ParseOp() {
    CarveOpStmt stmt;
    stmt.line = line_;
    do {
      std::string name;
      if (auto status = TakeName(&name); !status.ok()) return status;
      stmt.outputs.push_back(ValueIndex(name));
    } while (TakePunct(","));
    if (!TakePunct("=")) return Missing("'='");
    std::string op_name;
    if (auto status = TakeName(&op_name); !status.ok()) return status;
    const auto op_code = OpCodeOf(op_name);
    if (!op_code) {
      return absl::InvalidArgumentError(
          absl::StrCat("'", op_name, "' is not a TFLite builtin operator"));
    }
    stmt.op_code = *op_code;
    if (!TakePunct("(")) return Missing("'('");
    if (!TakePunct(")")) {
      do {
        std::string name;
        if (auto status = TakeName(&name); !status.ok()) return status;
        stmt.inputs.push_back(name == "_" ? -1 : ValueIndex(name));
      } while (TakePunct(","));
      if (!TakePunct(")")) return Missing("')'");
    }
    if (TakePunct("{")) {
      do {
        std::string option;
        if (auto status = TakeName(&option); !status.ok()) return status;
        if (!TakePunct("=")) return Missing("'='");
        CarveExpr expr;
        if (auto status = ParseExpr(&expr); !status.ok()) return status;
        stmt.options.emplace_back(std::move(option), std::move(expr));
      } while (TakePunct(","));
      if (!TakePunct("}")) return Missing("'}'");
    }
    if (TakePunct(":")) {
      size_t output = 0;
      do {
        if (output >= stmt.outputs.size()) {
          return absl::InvalidArgumentError("more types than results");
        }
        auto& decl = pattern_.values[stmt.outputs[output++]];
        if (auto status = ParseType(&decl.type); !status.ok()) return status;
        decl.has_type = true;
      } while (TakePunct(","));
    }
    for (const int value : stmt.outputs) {
      auto& decl = pattern_.values[value];
      if (decl.role == CarveValueDecl::Role::kResult) {
        return absl::InvalidArgumentError(
            absl::StrCat("'", decl.name, "' is the result of two ops"));
      }
      if (decl.role != CarveValueDecl::Role::kConstant) {
        return absl::InvalidArgumentError(absl::StrCat(
            "'", decl.name, "' is an input and the result of an op"));
      }
      decl.role = CarveValueDecl::Role::kResult;
    }
    pattern_.stmts.push_back(std::move(stmt));
    return ExpectEnd();
  }

 private:
  const Token& Peek() const { return tokens_[pos_]; }

  bool TakePunct(absl::string_view punct) {
    if (Peek().kind == Token::Kind::kPunct && Peek().text == punct) {
      ++pos_;
      return true;
    }
    return false;
  }

  absl::Status Missing(absl::string_view what) const {
    return absl::InvalidArgumentError(
        absl::StrCat("expected ", what,
                     Peek().kind == Token::Kind::kEnd
                         ? " at the end of the line"
                         : absl::StrCat(" before '", Peek().text, "'")));
  }

  absl::Status ExpectEnd() const {
    return Peek().kind == Token::Kind::kEnd ? absl::OkStatus()
                                            : Missing("the end of the line");
  }

  absl::Status TakeName(std::string* name) {
    if (Peek().kind != Token::Kind::kName) return Missing("a name");
    *name = Peek().text;
    ++pos_;
    return absl::OkStatus();
  }

  int ValueIndex(const std::string& name) {
    for (size_t i = 0; i < pattern_.values.size(); ++i) {
      if (pattern_.values[i].name == name) return static_cast<int>(i);
    }
    pattern_.values.emplace_back().name = name;
    return static_cast<int>(pattern_.values.size() - 1);
  }

  int SymbolIndex(const std::string& name) {
    for (size_t i = 0; i < pattern_.symbols.size(); ++i) {
      if (pattern_.symbols[i] == name) return static_cast<int>(i);
    }
    pattern_.symbols.push_back(name);
    return static_cast<int>(pattern_.symbols.size() - 1);
  }

  // [ELEMENT][<SCALE>][SHAPE]
  absl::Status ParseType(CarveTypeSpec* type) {
    const size_t start = pos_;
    if (Peek().kind == Token::Kind::kName) {
      const auto element_type = ElementTypeOf(Peek().text);
      if (!element_type) {
        return absl::InvalidArgumentError(
            absl::StrCat("'", Peek().text, "' is not an element type"));
      }
      type->has_element_type = true;
      type->element_type = *element_type;
      ++pos_;
    }
    if (TakePunct("<")) {
      std::string symbol;
      if (auto status = TakeName(&symbol); !status.ok()) return status;
      type->scale_symbol = SymbolIndex(symbol);
      if (!TakePunct(">")) return Missing("'>'");
    }
    if (TakePunct("[")) {
      type->has_shape = true;
      if (!TakePunct("]")) {
        do {
          CarveTypeSpec::Dim dim;
          if (Peek().kind == Token::Kind::kInt) {
            dim.kind = CarveTypeSpec::Dim::Kind::kLiteral;
            dim.value = Peek().int_value;
            ++pos_;
          } else if (Peek().kind == Token::Kind::kName) {
            if (Peek().text != "_") {
              dim.kind = CarveTypeSpec::Dim::Kind::kSymbol;
              dim.value = SymbolIndex(Peek().text);
            }
            ++pos_;
          } else {
            return Missing("a dimension");
          }
          type->shape.push_back(dim);
        } while (TakePunct(","));
        if (!TakePunct("]")) return Missing("']'");
      }
    }
    return pos_ == start ? Missing("a type") : absl::OkStatus();
  }

  // E = SUM [(== | !=) SUM]
  absl::Status ParseExpr(CarveExpr* expr) {
    if (auto status = ParseSum(expr); !status.ok()) return status;
    if (Peek().kind == Token::Kind::kPunct &&
        (Peek().text == "==" || Peek().text == "!=")) {
      const char op = Peek().text[0];
      ++pos_;
      CarveExpr rhs;
      if (auto status = ParseSum(&rhs); !status.ok()) return status;
      *expr = Binary(op, std::move(*expr), std::move(rhs));
    }
    return absl::OkStatus();
  }

  absl::Status ParseSum(CarveExpr* expr) {
    if (auto status = ParseProduct(expr); !status.ok()) return status;
    while (Peek().kind == Token::Kind::kPunct &&
           (Peek().text == "+" || Peek().text == "-")) {
      const char op = Peek().text[0];
      ++pos_;
      CarveExpr rhs;
      if (auto status = ParseProduct(&rhs); !status.ok()) return status;
      *expr = Binary(op, std::move(*expr), std::move(rhs));
    }
    return absl::OkStatus();
  }

  absl::Status ParseProduct(CarveExpr* expr) {
    if (auto status = ParseAtom(expr); !status.ok()) return status;
    while (Peek().kind == Token::Kind::kPunct &&
           (Peek().text == "*" || Peek().text == "/")) {
      const char op = Peek().text[0];
      ++pos_;
      CarveExpr rhs;
      if (auto status = ParseAtom(&rhs); !status.ok()) return status;
      *expr = Binary(op, std::move(*expr), std::move(rhs));
    }
    return absl::OkStatus();
  }

  absl::Status ParseAtom(CarveExpr* expr) {
    const Token& token = Peek();
    switch (token.kind) {
      case Token::Kind::kInt:
        expr->literal = token.int_value;
        ++pos_;
        return absl::OkStatus();
      case Token::Kind::kFloat:
        expr->literal = token.float_value;
        ++pos_;
        return absl::OkStatus();
      case Token::Kind::kString:
        expr->literal = token.text;
        ++pos_;
        return absl::OkStatus();
      case Token::Kind::kName:
        if (token.text == "true" || token.text == "false") {
          expr->literal = token.text == "true";
        } else {
          // Symbols may be introduced by a later line: ResolveNames tells
          // them from constants once the pattern is complete.
          expr->kind = CarveExpr::Kind::kSymbol;
          expr->literal = token.text;
        }
        ++pos_;
        return absl::OkStatus();
      case Token::Kind::kPunct:
        if (TakePunct("(")) {
          if (auto status = ParseExpr(expr); !status.ok()) return status;
          return TakePunct(")") ? absl::OkStatus() : Missing("')'");
        }
        if (TakePunct("[")) {
          expr->kind = CarveExpr::Kind::kList;
          if (!TakePunct("]")) {
            do {
              if (auto status = ParseExpr(&expr->args.emplace_back());
                  !status.ok()) {
                return status;
              }
            } while (TakePunct(","));
            if (!TakePunct("]")) return Missing("']'");
          }
          return absl::OkStatus();
        }
        if (TakePunct("-")) {
          CarveExpr operand;
          if (auto status = ParseAtom(&operand); !status.ok()) return status;
          CarveExpr zero;
          zero.literal = int64_t{0};
          *expr = Binary('-', std::move(zero), std::move(operand));
          return absl::OkStatus();
        }
        break;
      default:
        break;
    }
    return Missing("a value");
  }

  static CarveExpr Binary(char op, CarveExpr lhs, CarveExpr rhs) {
    CarveExpr expr;
    expr.kind = CarveExpr::Kind::kBinary;
    expr.op = op;
    expr.args.push_back(std::move(lhs));
    expr.args.push_back(std::move(rhs));
    return expr;
  }

  CarvePattern& pattern_;
  const std::vector<Token>& tokens_;
  const int line_;
  size_t pos_ = 0;
};

// Turns the names of an expression into symbols and constants.
absl::Status ResolveNames(const CarvePattern& pattern, CarveExpr* expr) {
  if (expr->kind == CarveExpr::Kind::kSymbol) {
    const auto& name = std::get<std::string>(expr->literal);
    for (size_t i = 0; i < pattern.symbols.size(); ++i) {
      if (pattern.symbols[i] == name) {
        expr->ref = static_cast<int>(i);
        return absl::OkStatus();
      }
    }
    for (size_t i = 0; i < pattern.values.size(); ++i) {
      if (pattern.values[i].name == name) {
        expr->kind = CarveExpr::Kind::kConstant;
        expr->ref = static_cast<int>(i);
        return absl::OkStatus();
      }
    }
    return absl::InvalidArgumentError(
        absl::StrCat("'", name, "' is neither a symbol of a type nor a value"));
  }
  for (auto& arg : expr->args) {
    if (auto status = ResolveNames(pattern, &arg); !status.ok()) return status;
  }
  return absl::OkStatus();
}

// The order in which the statements are bound: from the anchor along the
// pattern's edges, through a tensor's producer where possible because a tensor
// has one producer and may have many users.
absl::Status PlanMatching(CarvePattern* pattern) {
  const auto& stmts = pattern->stmts;
  std::vector<bool> planned(stmts.size(), false);
  std::vector<bool> bound(pattern->values.size(), false);
  auto bind = [&](size_t stmt) {
    planned[stmt] = true;
    for (const int value : stmts[stmt].inputs) {
      if (value >= 0) bound[value] = true;
    }
    for (const int value : stmts[stmt].outputs) bound[value] = true;
  };
  bind(stmts.size() - 1);
  auto next_step = [&]() -> std::optional<CarvePlanStep> {
    for (size_t s = 0; s < stmts.size(); ++s) {
      if (planned[s]) continue;
      for (size_t j = 0; j < stmts[s].outputs.size(); ++j) {
        const int value = stmts[s].outputs[j];
        if (bound[value]) {
          return CarvePlanStep{static_cast<int>(s), value, true,
                               static_cast<int>(j)};
        }
      }
    }
    // A constant can have users all over the graph: a result or an input is
    // the better tensor to find an op from.
    for (const bool through_constants : {false, true}) {
      for (size_t s = 0; s < stmts.size(); ++s) {
        if (planned[s]) continue;
        for (size_t i = 0; i < stmts[s].inputs.size(); ++i) {
          const int value = stmts[s].inputs[i];
          if (value < 0 || !bound[value]) continue;
          const auto role = pattern->values[value].role;
          const bool is_constant = role == CarveValueDecl::Role::kConstant ||
                                   role == CarveValueDecl::Role::kConstInput;
          if (is_constant == through_constants) {
            return CarvePlanStep{static_cast<int>(s), value, false,
                                 static_cast<int>(i)};
          }
        }
      }
    }
    return std::nullopt;
  };
  for (size_t n = 1; n < stmts.size(); ++n) {
    const auto step = next_step();
    if (!step) {
      const size_t stray =
          std::find(planned.begin(), planned.end(), false) - planned.begin();
      return absl::InvalidArgumentError(absl::StrCat(
          "line ", stmts[stray].line,
          ": the op is not connected to the last op of the pattern"));
    }
    pattern->plan.push_back(*step);
    bind(step->stmt);
  }
  return absl::OkStatus();
}

absl::Status FinishPattern(CarvePattern* pattern) {
  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(
        absl::StrCat("pattern ", pattern->name, ": ", message));
  };
  if (pattern->stmts.empty()) return error("no ops");
  if (pattern->outputs.empty()) return error("no outputs");
  for (const int value : pattern->outputs) {
    if (pattern->values[value].role != CarveValueDecl::Role::kResult) {
      return error(absl::StrCat("output '", pattern->values[value].name,
                                "' is not the result of an op"));
    }
  }
  std::vector<bool> used(pattern->values.size(), false);
  for (const auto& stmt : pattern->stmts) {
    for (const int value : stmt.inputs) {
      if (value >= 0) used[value] = true;
    }
  }
  for (const int value : pattern->inputs) {
    if (!used[value]) {
      return error(absl::StrCat("input '", pattern->values[value].name,
                                "' is not used by an op"));
    }
  }
  for (const auto& symbol : pattern->symbols) {
    for (const auto& value : pattern->values) {
      if (value.name == symbol) {
        return error(absl::StrCat("'", symbol, "' names a symbol and a value"));
      }
    }
  }
  auto resolve = [&](CarveExpr* expr) -> absl::Status {
    if (auto status = ResolveNames(*pattern, expr); !status.ok()) {
      return error(status.message());
    }
    return absl::OkStatus();
  };
  for (auto& stmt : pattern->stmts) {
    for (auto& [name, expr] : stmt.options) {
      if (auto status = resolve(&expr); !status.ok()) return status;
    }
  }
  for (auto& expr : pattern->checks) {
    if (auto status = resolve(&expr); !status.ok()) return status;
  }
  for (auto& [key, expr] : pattern->attributes) {
    if (auto status = resolve(&expr); !status.ok()) return status;
  }
  if (auto status = PlanMatching(pattern); !status.ok()) {
    return error(status.message());
  }
  return absl::OkStatus();
}

// MATCHING

std::string TensorDescription(const LiteRtTensorT& tensor) {
  if (tensor.Type().first != kLiteRtRankedTensorType) return "unranked";
  const auto& type = tensor.Type().second.ranked_tensor_type;
  std::vector<int32_t> dims(type.layout.dimensions,
                            type.layout.dimensions + type.layout.rank);
  return absl::StrCat(ElementTypeNameOf(type.element_type), " [",
                      absl::StrJoin(dims, ", "), "]");
}

// Matches one pattern at the ops of one subgraph.
class Matcher {
 public:
  Matcher(const CarvePattern& pattern, const LiteRtSubgraphT& subgraph,
          const absl::flat_hash_map<const LiteRtOpT*, size_t>& positions)
      : pattern_(pattern), subgraph_(subgraph), positions_(positions) {}

  // Tries the anchor on `op`. On a match fills `region`; otherwise `reason`
  // says why if the attempt came close.
  bool Match(const LiteRtOpT& op, CarveRegion* region, std::string* reason) {
    tensors_.assign(pattern_.values.size(), nullptr);
    ops_.assign(pattern_.stmts.size(), nullptr);
    num_bound_ = 0;
    best_depth_ = 0;
    best_reason_.clear();
    region_ = region;
    if (Bind(static_cast<int>(pattern_.stmts.size()) - 1, op, 0)) return true;
    // The anchor alone matches every op of its kind, and short chains of
    // common ops are everywhere: only an attempt that found a quarter of the
    // ops is worth a line in the report.
    if (best_depth_ >= std::max<size_t>(2, pattern_.stmts.size() / 4)) {
      *reason = best_reason_;
    } else {
      reason->clear();
    }
    return false;
  }

 private:
  // Records why the attempt stopped; the deepest stop is the one reported.
  bool Fail(size_t depth, std::string reason) {
    if (depth > best_depth_) {
      best_depth_ = depth;
      best_reason_ = std::move(reason);
    }
    return false;
  }

  std::string Where(const CarveOpStmt& stmt) const {
    return absl::StrCat("line ", stmt.line, " (", OpName(stmt.op_code), ")");
  }

  // Binds statement `stmt` to `op` and goes on with step `next_step` of the
  // plan. Undoes the binding when the rest does not match.
  bool Bind(int stmt_index, const LiteRtOpT& op, size_t next_step) {
    const auto& stmt = pattern_.stmts[stmt_index];
    if (op.OpCode() != stmt.op_code) {
      return Fail(
          num_bound_,
          absl::StrCat(Where(stmt), ": found ", OpName(op.OpCode()), " after ",
                       num_bound_, " of ", pattern_.stmts.size(), " ops"));
    }
    if (op.NumInputs() != stmt.inputs.size() ||
        op.NumOutputs() != stmt.outputs.size()) {
      return Fail(num_bound_,
                  absl::StrCat(Where(stmt), ": the op has ", op.NumInputs(),
                               " inputs and ", op.NumOutputs(),
                               " outputs, the pattern ", stmt.inputs.size(),
                               " and ", stmt.outputs.size()));
    }
    if (std::find(ops_.begin(), ops_.end(), &op) != ops_.end()) {
      return Fail(num_bound_,
                  absl::StrCat(Where(stmt), ": the op is already part of ",
                               "the match"));
    }
    std::vector<int> newly_bound;
    auto unify = [&](int value, const LiteRtTensorT& tensor) {
      if (value < 0) return true;
      if (tensors_[value] == nullptr) {
        tensors_[value] = &tensor;
        newly_bound.push_back(value);
        return true;
      }
      return tensors_[value] == &tensor;
    };
    bool unified = true;
    for (size_t i = 0; unified && i < stmt.inputs.size(); ++i) {
      unified = unify(stmt.inputs[i], op.Input(i));
      if (!unified) {
        Fail(num_bound_,
             absl::StrCat(Where(stmt), ": input ", i, " is not '",
                          pattern_.values[stmt.inputs[i]].name, "'"));
      }
    }
    for (size_t j = 0; unified && j < stmt.outputs.size(); ++j) {
      unified = unify(stmt.outputs[j], op.Output(j));
      if (!unified) {
        Fail(num_bound_,
             absl::StrCat(Where(stmt), ": output ", j, " is not '",
                          pattern_.values[stmt.outputs[j]].name, "'"));
      }
    }
    if (unified) {
      ops_[stmt_index] = &op;
      ++num_bound_;
      if (Continue(next_step)) return true;
      --num_bound_;
      ops_[stmt_index] = nullptr;
    }
    for (const int value : newly_bound) tensors_[value] = nullptr;
    return false;
  }

  bool Continue(size_t step_index) {
    if (step_index == pattern_.plan.size()) return CheckPredicates();
    const auto& step = pattern_.plan[step_index];
    const auto& stmt = pattern_.stmts[step.stmt];
    const LiteRtTensorT& via = *tensors_[step.via_value];
    if (step.via_output) {
      const LiteRtOpT* producer = via.DefiningOp();
      if (producer == nullptr ||
          via.DefiningOpOutInd() != static_cast<size_t>(step.position)) {
        return Fail(
            num_bound_,
            absl::StrCat(Where(stmt), ": '",
                         pattern_.values[step.via_value].name,
                         "' is not the result of an op (", num_bound_, " of ",
                         pattern_.stmts.size(), " ops matched)"));
      }
      return Bind(step.stmt, *producer, step_index + 1);
    }
    for (size_t use = 0; use < via.NumUses(); ++use) {
      if (via.UserArgInds()[use] == static_cast<size_t>(step.position) &&
          Bind(step.stmt, *via.Users()[use], step_index + 1)) {
        return true;
      }
    }
    return Fail(num_bound_,
                absl::StrCat(Where(stmt), ": no user of '",
                             pattern_.values[step.via_value].name,
                             "' matches (", num_bound_, " of ",
                             pattern_.stmts.size(), " ops matched)"));
  }

  // A stop after every op was found.
  bool Reject(std::string reason) {
    return Fail(pattern_.stmts.size() + 1, std::move(reason));
  }

  bool CheckType(const CarveValueDecl& decl, const LiteRtTensorT& tensor) {
    const auto& spec = decl.type;
    if (tensor.Type().first != kLiteRtRankedTensorType) {
      return Reject(absl::StrCat("'", decl.name, "' is unranked"));
    }
    const auto& type = tensor.Type().second.ranked_tensor_type;
    if (spec.has_element_type && type.element_type != spec.element_type) {
      return Reject(absl::StrCat("'", decl.name, "' is ",
                                 TensorDescription(tensor), ", expected ",
                                 ElementTypeNameOf(spec.element_type)));
    }
    if (spec.scale_symbol >= 0) {
      const auto& quantization = tensor.Qparams();
      if (quantization.first != kLiteRtQuantizationPerTensor ||
          quantization.second.per_tensor.zero_point != 0) {
        return Reject(absl::StrCat(
            "'", decl.name, "' is not quantized per tensor with zero point 0"));
      }
      const double scale = quantization.second.per_tensor.scale;
      auto& symbol = symbols_[spec.scale_symbol];
      if (!symbol) {
        symbol = CarveValue(scale);
      } else if (!ValuesEqual(*symbol, CarveValue(scale))) {
        return Reject(absl::StrCat("'", decl.name, "' has scale ", scale,
                                   ", but ",
                                   pattern_.symbols[spec.scale_symbol], " is ",
                                   CarveValueToString(*symbol)));
      }
    }
    if (!spec.has_shape) return true;
    if (type.layout.rank != spec.shape.size()) {
      return Reject(absl::StrCat("'", decl.name, "' is ",
                                 TensorDescription(tensor), ", expected rank ",
                                 spec.shape.size()));
    }
    for (size_t d = 0; d < spec.shape.size(); ++d) {
      const int64_t dim = type.layout.dimensions[d];
      const auto& expected = spec.shape[d];
      if (expected.kind == CarveTypeSpec::Dim::Kind::kLiteral) {
        if (dim != expected.value) {
          return Reject(absl::StrCat(
              "'", decl.name, "' is ", TensorDescription(tensor),
              ", expected dimension ", d, " to be ", expected.value));
        }
      } else if (expected.kind == CarveTypeSpec::Dim::Kind::kSymbol) {
        auto& symbol = symbols_[expected.value];
        if (!symbol) {
          symbol = CarveValue(dim);
        } else if (!ValuesEqual(*symbol, CarveValue(dim))) {
          return Reject(absl::StrCat("'", decl.name, "' is ",
                                     TensorDescription(tensor), ", but ",
                                     pattern_.symbols[expected.value], " is ",
                                     CarveValueToString(*symbol)));
        }
      }
    }
    return true;
  }

  // A constant of the region moves into the decomposition. One with blockwise
  // quantization cannot: its scale tensors would move with it, and the TFLite
  // interpreter does not renumber them when it inlines a decomposition, so
  // the weights of a region have to be inputs.
  bool CheckRegionConstant(absl::string_view name,
                           const LiteRtTensorT& tensor) {
    if (!IsConstant(tensor)) {
      return Reject(absl::StrCat(name, " is not a constant"));
    }
    if (tensor.Qparams().first == kLiteRtQuantizationBlockWise) {
      return Reject(absl::StrCat(
          name, " has blockwise quantization: declare it as a const input"));
    }
    return true;
  }

  absl::StatusOr<CarveValue> Eval(const CarveExpr& expr) const {
    switch (expr.kind) {
      case CarveExpr::Kind::kLiteral:
        return expr.literal;
      case CarveExpr::Kind::kSymbol:
        if (!symbols_[expr.ref]) {
          return absl::FailedPreconditionError(absl::StrCat(
              pattern_.symbols[expr.ref], " has no value in this match"));
        }
        return *symbols_[expr.ref];
      case CarveExpr::Kind::kConstant:
        return ConstantElements(pattern_.values[expr.ref].name,
                                *tensors_[expr.ref]);
      case CarveExpr::Kind::kList: {
        std::vector<int64_t> list;
        for (const auto& arg : expr.args) {
          auto element = Eval(arg);
          if (!element.ok()) return element.status();
          const auto* value = std::get_if<int64_t>(&*element);
          if (value == nullptr) {
            return absl::InvalidArgumentError(
                "a list holds whole numbers only");
          }
          list.push_back(*value);
        }
        return CarveValue(std::move(list));
      }
      case CarveExpr::Kind::kBinary: {
        auto lhs = Eval(expr.args[0]);
        if (!lhs.ok()) return lhs.status();
        auto rhs = Eval(expr.args[1]);
        if (!rhs.ok()) return rhs.status();
        if (expr.op == '=') return CarveValue(ValuesEqual(*lhs, *rhs));
        if (expr.op == '!') return CarveValue(!ValuesEqual(*lhs, *rhs));
        return Arithmetic(expr.op, *lhs, *rhs);
      }
    }
    return absl::InternalError("unknown expression");
  }

  static absl::StatusOr<CarveValue> ConstantElements(
      absl::string_view name, const LiteRtTensorT& tensor) {
    if (!IsConstant(tensor) || tensor.Type().first != kLiteRtRankedTensorType) {
      return absl::FailedPreconditionError(
          absl::StrCat("'", name, "' is not a constant"));
    }
    const auto element_type =
        tensor.Type().second.ranked_tensor_type.element_type;
    const size_t width = element_type == kLiteRtElementTypeInt32   ? 4
                         : element_type == kLiteRtElementTypeInt64 ? 8
                                                                   : 0;
    const auto data = tensor.Weights().Buffer();
    if (width == 0 || data.Size() == 0 || data.Size() % width != 0 ||
        data.Size() / width > kMaxConstantElements) {
      return absl::FailedPreconditionError(absl::StrCat(
          "'", name, "' is not a constant of at most ", kMaxConstantElements,
          " i32 or i64 elements whose data is in the model"));
    }
    std::vector<int64_t> elements(data.Size() / width);
    for (size_t i = 0; i < elements.size(); ++i) {
      if (width == 4) {
        int32_t element;
        std::memcpy(&element, data.Data() + i * width, width);
        elements[i] = element;
      } else {
        std::memcpy(&elements[i], data.Data() + i * width, width);
      }
    }
    return CarveValue(std::move(elements));
  }

  // Everything that is not the shape of the graph: the predicates on the
  // tensors, the options, the checks, and that the ops can become one op.
  bool CheckPredicates() {
    symbols_.assign(pattern_.symbols.size(), std::nullopt);
    for (size_t v = 0; v < pattern_.values.size(); ++v) {
      const auto& decl = pattern_.values[v];
      const LiteRtTensorT& tensor = *tensors_[v];
      if (decl.has_type && !CheckType(decl, tensor)) return false;
      if (decl.role == CarveValueDecl::Role::kConstInput &&
          !IsConstant(tensor)) {
        return Reject(
            absl::StrCat("input '", decl.name, "' is not a constant"));
      }
      if (decl.role == CarveValueDecl::Role::kConstant &&
          !CheckRegionConstant(absl::StrCat("'", decl.name, "'"), tensor)) {
        return false;
      }
    }
    absl::flat_hash_set<const LiteRtOpT*> region(ops_.begin(), ops_.end());
    for (size_t s = 0; s < pattern_.stmts.size(); ++s) {
      const auto& stmt = pattern_.stmts[s];
      for (size_t i = 0; i < stmt.inputs.size(); ++i) {
        if (stmt.inputs[i] < 0 &&
            !CheckRegionConstant(
                absl::StrCat(Where(stmt), ": input ", i, " ('_')"),
                ops_[s]->Input(i))) {
          return false;
        }
      }
      for (const auto& [name, expr] : stmt.options) {
        auto actual = OptionOf(*ops_[s], name);
        if (!actual.ok()) return Reject(std::string(actual.status().message()));
        auto expected = Eval(expr);
        if (!expected.ok()) {
          return Reject(absl::StrCat(Where(stmt), ": option ", name, ": ",
                                     expected.status().message()));
        }
        if (!ValuesEqual(*actual, *expected)) {
          return Reject(absl::StrCat(Where(stmt), ": option ", name, " is ",
                                     CarveValueToString(*actual), ", expected ",
                                     CarveValueToString(*expected)));
        }
      }
    }
    for (size_t c = 0; c < pattern_.checks.size(); ++c) {
      auto result = Eval(pattern_.checks[c]);
      if (!result.ok()) {
        return Reject(absl::StrCat(pattern_.check_texts[c], ": ",
                                   result.status().message()));
      }
      const auto* holds = std::get_if<bool>(&*result);
      if (holds == nullptr || !*holds) {
        return Reject(absl::StrCat(pattern_.check_texts[c], " does not hold"));
      }
    }

    // The composite takes the place of the region's last op, so nothing
    // outside may read a result before that, and only outputs may be read
    // outside at all.
    size_t last_position = 0;
    for (const auto* op : ops_) {
      last_position = std::max(last_position, positions_.at(op));
    }
    for (size_t v = 0; v < pattern_.values.size(); ++v) {
      const auto& decl = pattern_.values[v];
      const LiteRtTensorT& tensor = *tensors_[v];
      if (decl.role != CarveValueDecl::Role::kResult) {
        if (tensor.DefiningOp() != nullptr &&
            region.contains(tensor.DefiningOp())) {
          return Reject(
              absl::StrCat("'", decl.name, "' is computed inside the region"));
        }
        continue;
      }
      const bool is_output =
          std::find(pattern_.outputs.begin(), pattern_.outputs.end(),
                    static_cast<int>(v)) != pattern_.outputs.end();
      if (!is_output && litert::internal::FindOutput(subgraph_, tensor)) {
        return Reject(absl::StrCat("'", decl.name,
                                   "' is an output of the subgraph but not ",
                                   "of the pattern"));
      }
      for (const auto* user : tensor.Users()) {
        if (region.contains(user)) continue;
        if (!is_output) {
          return Reject(absl::StrCat("'", decl.name, "' is also read by ",
                                     OpName(user->OpCode()),
                                     " outside the region"));
        }
        if (positions_.at(user) < last_position) {
          return Reject(absl::StrCat("output '", decl.name, "' is read by ",
                                     OpName(user->OpCode()),
                                     " before the region ends"));
        }
      }
    }

    CarveRegion result;
    result.pattern = &pattern_;
    result.ops.assign(ops_.begin(), ops_.end());
    std::sort(result.ops.begin(), result.ops.end(),
              [&](const LiteRtOpT* a, const LiteRtOpT* b) {
                return positions_.at(a) < positions_.at(b);
              });
    for (const int value : pattern_.inputs) {
      result.inputs.push_back(tensors_[value]);
    }
    for (const int value : pattern_.outputs) {
      result.outputs.push_back(tensors_[value]);
    }
    for (const auto& [key, expr] : pattern_.attributes) {
      auto value = Eval(expr);
      if (!value.ok()) {
        return Reject(
            absl::StrCat("attribute ", key, ": ", value.status().message()));
      }
      result.attributes.emplace_back(key, std::move(*value));
    }
    *region_ = std::move(result);
    return true;
  }

  const CarvePattern& pattern_;
  const LiteRtSubgraphT& subgraph_;
  const absl::flat_hash_map<const LiteRtOpT*, size_t>& positions_;

  // The tensor of each value and the op of each statement of the pattern.
  std::vector<const LiteRtTensorT*> tensors_;
  std::vector<const LiteRtOpT*> ops_;
  std::vector<std::optional<CarveValue>> symbols_;
  size_t num_bound_ = 0;
  size_t best_depth_ = 0;
  std::string best_reason_;
  CarveRegion* region_ = nullptr;
};

}  // namespace

std::string CarveValueToString(const CarveValue& value) {
  struct Printer {
    std::string operator()(bool b) const { return b ? "true" : "false"; }
    std::string operator()(int64_t i) const { return absl::StrCat(i); }
    std::string operator()(double d) const { return absl::StrCat(d); }
    std::string operator()(const std::string& s) const {
      return absl::StrCat("\"", s, "\"");
    }
    std::string operator()(const std::vector<int64_t>& list) const {
      return absl::StrCat("[", absl::StrJoin(list, ", "), "]");
    }
  };
  return std::visit(Printer(), value);
}

absl::StatusOr<std::vector<CarvePattern>> ParseCarvePatterns(
    absl::string_view text) {
  std::vector<CarvePattern> patterns;
  int line_number = 0;
  for (const absl::string_view line : absl::StrSplit(text, '\n')) {
    ++line_number;
    std::vector<Token> tokens;
    absl::Status status = Tokenize(line, &tokens);
    if (status.ok() && tokens.front().kind != Token::Kind::kEnd) {
      const Token& first = tokens.front();
      const bool is_keyword = first.kind == Token::Kind::kName;
      if (is_keyword && first.text == "pattern") {
        if (tokens.size() != 3 || tokens[1].kind != Token::Kind::kName) {
          status = absl::InvalidArgumentError("expected: pattern NAME");
        } else {
          if (!patterns.empty()) status = FinishPattern(&patterns.back());
          patterns.emplace_back().name = tokens[1].text;
        }
      } else if (patterns.empty()) {
        status = absl::InvalidArgumentError(
            "a description starts with: pattern NAME");
      } else {
        LineParser parser(&patterns.back(), tokens, line_number);
        if (is_keyword && first.text == "in") {
          status = parser.ParseIn();
        } else if (is_keyword && first.text == "out") {
          status = parser.ParseOut();
        } else if (is_keyword && first.text == "check") {
          status = parser.ParseCheck(
              line.substr(0, std::min(line.find('#'), line.size())));
        } else if (is_keyword && first.text == "attr") {
          status = parser.ParseAttr();
        } else {
          status = parser.ParseOp();
        }
      }
    }
    if (!status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_number, ": ", status.message()));
    }
  }
  if (patterns.empty()) {
    return absl::InvalidArgumentError("the description holds no pattern");
  }
  if (auto status = FinishPattern(&patterns.back()); !status.ok()) {
    return status;
  }
  return patterns;
}

CarveMatches FindCarveRegions(absl::Span<const CarvePattern> patterns,
                              const LiteRtSubgraphT& subgraph) {
  absl::flat_hash_map<const LiteRtOpT*, size_t> positions;
  for (size_t i = 0; i < subgraph.Ops().size(); ++i) {
    positions[subgraph.Ops()[i]] = i;
  }
  CarveMatches matches;
  absl::flat_hash_set<const LiteRtOpT*> claimed;
  for (const auto& pattern : patterns) {
    Matcher matcher(pattern, subgraph, positions);
    const LiteRtOpCode anchor = pattern.stmts.back().op_code;
    for (const auto* op : subgraph.Ops()) {
      if (op->OpCode() != anchor) continue;
      CarveRegion region;
      std::string reason;
      if (matcher.Match(*op, &region, &reason)) {
        const bool overlaps = std::any_of(
            region.ops.begin(), region.ops.end(),
            [&](const LiteRtOpT* o) { return claimed.contains(o); });
        if (overlaps) {
          matches.near_misses.push_back(
              {&pattern, op, "overlaps a region that was found before it"});
        } else {
          claimed.insert(region.ops.begin(), region.ops.end());
          matches.regions.push_back(std::move(region));
        }
      } else if (!reason.empty()) {
        matches.near_misses.push_back({&pattern, op, std::move(reason)});
      }
    }
  }
  return matches;
}

std::vector<uint8_t> PackCarveAttributes(
    const std::vector<std::pair<std::string, CarveValue>>& attributes) {
  flexbuffers::Builder builder;
  // The existing composites store 32-bit floats, and a tensor's scale is one.
  struct Packer {
    void operator()(bool b) const { builder.Bool(key, b); }
    void operator()(int64_t i) const { builder.Int(key, i); }
    void operator()(double d) const {
      builder.Float(key, static_cast<float>(d));
    }
    void operator()(const std::string& s) const { builder.String(key, s); }
    void operator()(const std::vector<int64_t>& list) const {
      builder.Vector(key, [&] {
        for (const int64_t element : list) builder.Int(element);
      });
    }
    flexbuffers::Builder& builder;
    const char* key;
  };
  builder.Map([&] {
    for (const auto& [key, value] : attributes) {
      std::visit(Packer{builder, key.c_str()}, value);
    }
  });
  builder.Finish();
  return builder.GetBuffer();
}

}  // namespace litert::tools
