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
"""Code generator for Match DAG."""

import dataclasses
import functools
import json

import lark

from litert.python.tools.model_utils.match._match_dag import instruction as inst
from litert.python.tools.model_utils.match._match_dag import tblgen_dag_parser

ROOT_VAR = inst.Var("_root")
Token = lark.Token
Transformer = lark.Transformer


def _tree_flatten(value):
  result = []
  if isinstance(value, (list, tuple)):
    for item in value:
      result.extend(_tree_flatten(item))
  else:
    result.append(value)
  return result


@functools.lru_cache(2000)
def generate_match_dag_instructions(dag_text: str) -> list[inst.Instruction]:
  """Parses a LLVM DAG code into a sequence of matching instructions."""

  tree = tblgen_dag_parser.parse_dag(dag_text)

  transformer = MatchDagCodeGenTransformer(text=dag_text)
  codegen = transformer.transform(tree)

  instructions = codegen(ROOT_VAR, 0)
  instructions = _tree_flatten(instructions)
  instructions = [i for i in instructions if not isinstance(i, inst.Null)]
  return instructions


class CodeGen:
  """Code generation helper for MatchDagCodeGenBuilder."""

  def __init__(self, gen=None):
    self.gen = gen

  def __call__(self, *args):
    if self.gen is None:
      return inst.Null()
    elif isinstance(self.gen, (list, tuple, inst.Instruction)):
      return self.gen
    elif callable(self.gen):
      return self.gen(*args)
    else:
      return self.gen


@dataclasses.dataclass
class MatchDagCodeGenTransformer(Transformer):
  """Lark Transformer for DAG matching code/instructions generation."""

  #  Accept source text in init to allow slicing later
  text: str

  def __post_init__(self):
    self.var_cnt = 0

  def _gen_extract(self, x, idx: int):
    if idx == 0:
      return (inst.Null(), x)
    else:
      var = inst.Var(f"var_{self.var_cnt}")
      self.var_cnt += 1
      return inst.SetVar(var, inst.Extract(x, idx)), var

  # --- Atomic Values ---

  def start(self, children):
    return children[0]

  def simple_value_bool(self, children):
    return CodeGen(inst.Value(children[0].value.title()))

  def simple_value_tok(self, children):
    token = children[0]

    if token.type == "TOK_INTEGER":
      return CodeGen(inst.Value(token.value))

    if token.type == "TOK_STRING":
      # Join multiple string tokens if present
      full_str = "".join([json.loads(t.value) for t in children])
      return CodeGen(inst.Value(repr(full_str)))

    if token.type == "TOK_CODE":
      raise ValueError("Code [{...}] is not allowed in MatchDag")

    return CodeGen(inst.Value(token.value))

  def simple_value_identifier(self, children):
    identifier = children[0].value

    def codegen(x=None, *_):
      if x is None:
        return inst.LookupDef(identifier)
      return inst.Invoke(inst.LookupDef(identifier), x)

    return CodeGen(codegen)

  # --- Forbidden Syntax ---

  def simple_value_question(self, _):
    raise ValueError("'?' is not allowed in MatchDag")

  def simple_value_operator(self, _):
    raise ValueError("Bang operators '!' and '#' are not allowed in MatchDag")

  def simple_value_list_type(self, _):
    raise ValueError("Syntax [...]<...> is not allowed in MatchDag")

  # --- Lists & Classes ---

  def value_list_ne(self, children):
    return CodeGen(inst.Value([gen() for gen in children]))

  def value_list(self, children):
    return children[0] if children else CodeGen(None)

  def simple_value_list(self, children):
    return children[0]

  def simple_value_class(self, children):
    fn = children[0]
    args_codegen = children[1]

    def codegen(x=None, *_):
      if x is None:
        return inst.Invoke(inst.LookupDef(fn), args_codegen())
      return inst.Invoke(inst.Invoke(inst.LookupDef(fn), args_codegen()), x)

    return CodeGen(codegen)

  def class_id(self, children):
    return children[0].value

  # --- Arguments ---

  def positional_arg_value_list(self, children):
    return CodeGen(inst.Value([gen() for gen in children]))

  def arg_value_list(self, children):
    positional = children[0]
    # If there's a second child (ignoring None), it's the named list
    if len(children) > 1 and children[-1] is not None:
      raise NotImplementedError("NamedArgValueList handler is not implemented")
    return positional

  # --- Values & DAGs ---

  def value(self, children):
    # Check for forbidden operator '#'
    for child in children:
      if isinstance(child, Token) and child.value == "#":
        raise ValueError("Operator '#' is not supported in MatchDag")

    if len(children) > 1:
      raise NotImplementedError("valueSuffix is not implemented")

    return children[0]

  @lark.v_args(tree=True)
  def dag_arg(self, tree):
    # Grammar: dag_arg: value (":" TOK_VAR_NAME)? | TOK_VAR_NAME
    assert len(tree.children) in (1, 2)

    tok_name, value_codegen = None, None
    for child in tree.children:
      if isinstance(child, CodeGen):
        value_codegen = child
      else:
        assert isinstance(child, Token)
        tok_name = child.value

    def codegen(x, i):
      nonlocal value_codegen, tok_name, tree
      assign, var = self._gen_extract(x, i)

      if value_codegen:
        meta = tree.meta
        # Slicing the source string passed to __init__
        value_repr = (
            self.text[meta.start_pos : meta.end_pos]
            .replace(" ", "")
            .replace("\n", " ")
        )

        begin_comment = inst.Comment(f"# === BEGIN: {value_repr} ===")
        end_comment = inst.Comment(f"# === END: {value_repr} ===")

        # Manually trigger transformation for the child tree
        value_code = value_codegen(var, 0)
      else:
        begin_comment = inst.Null()
        end_comment = inst.Null()
        value_code = CodeGen()(var, 0)

      if tok_name:
        tok_code = inst.SetResult(tok_name, var)
      else:
        tok_code = inst.Null()

      return [assign, begin_comment, value_code, tok_code, end_comment]

    return CodeGen(codegen)

  def dag_arg_list(self, children):
    # Filter out commas (Tokens) from children
    arg_codegens = [
        c for c in children if isinstance(c, (CodeGen, inst.Instruction))
    ]

    return CodeGen(
        lambda x, start_i: [
            codegen(x, i)
            for i, codegen in enumerate(arg_codegens, start=start_i)
        ]
    )

  def simple_value_dag(self, children):
    dag_arg_codegen = children[0]

    dag_arg_list_codegen = None
    if len(children) > 1 and children[1] is not None:
      dag_arg_list_codegen = children[1]

    list_runner = CodeGen(
        dag_arg_list_codegen if dag_arg_list_codegen else None
    )

    def codegen(x, i):
      assign, new_var = self._gen_extract(x, i)
      return [
          assign,
          dag_arg_codegen(new_var, 0),
          list_runner(new_var, 1),
      ]

    return CodeGen(codegen)
