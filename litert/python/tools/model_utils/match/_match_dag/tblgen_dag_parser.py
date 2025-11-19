# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TableGen DAG parser."""

import functools
import lark

# TableGen grammar for DAG definitions.
GRAMMAR = r"""
start: simple_value_dag

// --- Values ---

// Use ? to inline rules that only have one child (cleans up the tree)
?value: simple_value value_suffix*
      | value "#" value?           -> paste_operator

// Suffixes
?value_suffix: "{" range_list "}"    -> suffix_range
             | "[" slice_elements "]" -> suffix_slice
             | "." TOK_IDENTIFIER     -> suffix_field

?simple_value: simple_value_tok
             | simple_value_bool
             | simple_value_question
             | simple_value_list
             | simple_value_list_type
             | simple_value_dag
             | simple_value_identifier
             | simple_value_class
             | simple_value_operator

// --- DAG Definitions ---

simple_value_dag: "(" dag_arg dag_arg_list? ")"

dag_arg_list: dag_arg ("," dag_arg)* ","?

// A DagArg can be a value with an optional variable name, or just a variable name
dag_arg: value (":" TOK_VAR_NAME)?
       | TOK_VAR_NAME

// --- Arguments & Lists ---

// ValueListNE ::= Value ("," Value)*
value_list_ne: value ("," value)* ","?

// ValueList ::= ValueListNE?
value_list: value_list_ne?

// Simple Lists
simple_value_list: "{" value_list "}"

// Typed Lists
simple_value_list_type: "[" value_list "]" "<" type ">"

// --- Operators & Functions ---

simple_value_operator: bang_operator ("<" type ">")? "(" value_list_ne ")"
                     | cond_operator "(" cond_clause ("," cond_clause)* ")"

cond_clause: value ":" value

// --- Ranges & Slices ---

range_list: range_piece ("," range_piece)*

range_piece: TOK_INTEGER
           | TOK_INTEGER "..." TOK_INTEGER -> range_ellipsis
           | TOK_INTEGER "-" TOK_INTEGER   -> range_hyphen
           | TOK_INTEGER TOK_INTEGER       -> range_pair

slice_elements: (slice_element ",")* slice_element ","?

slice_element: value
             | value "..." value -> slice_ellipsis
             | value "-" value   -> slice_hyphen
             | value TOK_INTEGER -> slice_int

// --- Arguments Parsing ---

arg_value_list: positional_arg_value_list ","? named_arg_value_list

positional_arg_value_list: (value ("," value)*)?

named_arg_value_list: (name_value "=" value ("," name_value "=" value)*)?

name_value: value

// --- Types & Classes ---

simple_value_class: class_id "<" arg_value_list ">"

class_id: TOK_IDENTIFIER

type: "bit"    -> type_bit
    | "int"    -> type_int
    | "string" -> type_string
    | "dag"    -> type_dag
    | "bits" "<" TOK_INTEGER ">" -> type_bits
    | "list" "<" type ">"        -> type_list
    | class_id

// --- Atomic Values ---

simple_value_tok: TOK_INTEGER
                | TOK_STRING+
                | TOK_CODE

simple_value_bool: "true" -> true
                 | "false" -> false

simple_value_question: "?"

simple_value_identifier: TOK_IDENTIFIER

// --- Terminals ---

// Operators
!bang_operator: "!add" | "!and" | "!cast" | "!con" | "!dag"
              | "!div" | "!empty" | "!eq" | "!exists" | "!filter"
              | "!find" | "!foldl" | "!foreach" | "!ge" | "!getdagarg"
              | "!getdagname" | "!getdagop" | "!gt" | "!head" | "!if"
              | "!interleave" | "!isa" | "!le" | "!listconcat" | "!listremove"
              | "!listsplat" | "!logtwo" | "!lt" | "!mul" | "!ne"
              | "!not" | "!or" | "!range" | "!repr" | "!setdagarg"
              | "!setdagname" | "!setdagop" | "!shl" | "!size" | "!sra"
              | "!srl" | "!strconcat" | "!sub" | "!subst" | "!substr"
              | "!tail" | "!tolower" | "!toupper" | "!xor"

!cond_operator: "!cond"

// Primitives
TOK_STRING: /"(\\.|[^"\\])*"/
TOK_CODE: /(?s)\[\{.*?\}\]/

TOK_INTEGER: /\+?-?[0-9]+/
           | /0x[0-9a-fA-F]+/
           | /0b[01]+/

TOK_IDENTIFIER: /[0-9]*[a-zA-Z_][a-zA-Z0-9_]*/
TOK_VAR_NAME: /\$[a-zA-Z_][a-zA-Z0-9_]*/

%ignore /[ \t\r\n]+/
%ignore /\/\/[^\n]*/
%ignore /(?s)\/\*.*?\*\//
"""


@functools.cache
def parser():
  return lark.Lark(GRAMMAR.strip(), propagate_positions=True)


def _get_error_context(text, line_idx: int, col_idx: int, span=4) -> str:
  """Returns the error context around the error."""
  lines = text.splitlines()
  idx = line_idx - 1

  # safe window calculation
  start = max(0, idx - span)
  end = min(len(lines), idx + span + 1)

  output = []
  for i in range(start, end):
    line_num = i + 1
    line_content = lines[i].replace(
        "\t", "    "
    )  # Expand tabs to avoid misalignment

    if i == idx:
      # --- Target Line ---
      # Structure: "  >> " (5 chars) + line_num (4 chars) + " | " (3 chars)
      # Total prefix width = 12 chars
      prefix = f"  >> {line_num:4d} | "
      output.append(f"{prefix}{line_content}")

      # --- Pointer Line ---
      # Indent = prefix width (12) + column index (minus 1 because it's 1-based)
      indent = " " * (12 + col_idx - 1)
      output.append(f"{indent}^")
    else:
      # --- Neighbor Line ---
      prefix = f"     {line_num:4d} | "
      output.append(f"{prefix}{line_content}")

  return "\n".join(output)


def parse_dag(text: str) -> lark.tree.Tree:
  """Parses a TableGen DAG definition."""
  try:
    return parser().parse(text)

  except lark.exceptions.UnexpectedToken as e:
    error_msg = f"Line {e.line}, Column {e.column}:"
    error_msg += f" Unexpected token: {repr(e.token.value)}\n"

    error_msg += "\n--- Context ---\n"
    error_msg += _get_error_context(text, e.line, e.column)
    error_msg += "\n---------------\n"

    raise ValueError(error_msg) from e

  except lark.exceptions.UnexpectedCharacters as e:
    error_msg = f"Line {e.line}, Column {e.column}:"
    error_msg += f" Unexpected character: {repr(e.char)}\n"

    error_msg += "\n--- Context ---\n"
    error_msg += _get_error_context(text, e.line, e.column)
    error_msg += "\n---------------\n"
    raise ValueError(error_msg) from e
