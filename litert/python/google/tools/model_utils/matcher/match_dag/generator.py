import dataclasses
import functools
import json

import antlr4
import jax

from litert.python.google.tools.model_utils.core import tblgen
from litert.python.google.tools.model_utils.matcher.match_dag import instruction as inst

ROOT_VAR = inst.Var("_root")


class _ErrorListener(antlr4.error.ErrorListener.ErrorListener):

  def __init__(self, text):
    super().__init__()
    self.text = text
    self.syntax_errors = []

  def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
    self.syntax_errors.append(
        "-- line " + str(line) + ":" + str(column) + " " + msg
    )

  def raise_error(self):
    if not self.syntax_errors:
      return
    error_msg = f"Failed to parse TableGen DAG: {self.text}\n\nSyntax Errors:\n"
    for msg in self.syntax_errors:
      error_msg = error_msg + msg + "\n"
    raise ValueError(error_msg)


@functools.lru_cache(2000)
def parse_match_dag(dag_text: str) -> list[inst.Instruction]:
  error_listener = _ErrorListener(dag_text)

  lexer = tblgen.Lexer(antlr4.InputStream(dag_text))
  lexer.removeErrorListeners()
  lexer.addErrorListener(error_listener)

  stream = antlr4.CommonTokenStream(lexer)

  parser = tblgen.Parser(stream)
  parser.removeErrorListeners()
  parser.addErrorListener(error_listener)

  tree = parser.simpleValueDag()
  error_listener.raise_error()

  visitor = MatchDagCodeGenBuilder()
  codegen = visitor.visit(tree)

  instructions = codegen(ROOT_VAR, 0)
  instructions, _ = jax.tree.flatten(instructions)
  instructions = [i for i in instructions if not isinstance(i, inst.Null)]
  return instructions


class CodeGen:

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
class MatchDagCodeGenBuilder(tblgen.Visitor):

  def __init__(self):
    self.var_cnt = 0

  def _gen_extract(self, x: inst.Var, idx: int):
    if idx == 0:
      return (inst.Null(), x)
    else:
      var = inst.Var()
      return inst.SetVar(var, inst.Extract(x, idx)), var

  def visitSimpleValueQuestion(self, ctx):
    raise ValueError("'?' is not allowed in MatchDag")

  def visitSimpleValueOperator(self, ctx):
    raise ValueError("Bang operators '!' and '#' are not allowed in MatchDag")

  def visitSimpleValueListType(self, ctx):
    raise ValueError("Syntax [...]<...> is not allowed in MatchDag")

  def visitSimpleValueBool(self, ctx: tblgen.Parser.SimpleValueBoolContext):
    return CodeGen(inst.Value(ctx.getText().title()))

  def visitValueListNE(self, ctx: tblgen.Parser.ValueListNEContext):
    values = ctx.value()
    codegens = [self.visit(v) for v in values]
    return CodeGen(inst.Value([CodeGen(gen)() for gen in codegens]))

  def visitValueList(self, ctx: tblgen.Parser.ValueListContext):
    if not ctx.valueListNE():
      return CodeGen(None)
    return self.visit(ctx.valueListNE())

  def visitSimpleValueList(self, ctx: tblgen.Parser.SimpleValueListContext):
    return self.visit(ctx.valueList())

  def visitSimpleValueTok(self, ctx: tblgen.Parser.SimpleValueTokContext):
    if ctx.TokInteger():
      return CodeGen(inst.Value(ctx.getText()))
    if ctx.TokString():
      data = repr(
          "".join([json.loads(child.getText()) for child in ctx.children])
      )
      return CodeGen(inst.Value(data))
    if ctx.TokCode():
      raise ValueError("Code [{...}] is not allowed in MatchDag")

  def visitPositionalArgValueList(
      self, ctx: tblgen.Parser.PositionalArgValueListContext
  ):
    value_codegens = [self.visit(v) for v in ctx.value()]
    return CodeGen(inst.Value([CodeGen(gen)() for gen in value_codegens]))

  def visitArgValueList(self, ctx: tblgen.Parser.ArgValueListContext):
    if ctx.namedArgValueList().getText():
      raise NotImplementedError("NamedArgValueList handler is not implemented")
    return self.visit(ctx.positionalArgValueList())

  def visitSimpleValueClass(self, ctx: tblgen.Parser.SimpleValueClassContext):
    fn = ctx.classID().getText()
    args_codegen = self.visit(ctx.argValueList())

    def codegen(x=None, *_):
      if x is None:
        return inst.Invoke(inst.LookupDef(fn), args_codegen())
      return inst.Invoke(inst.Invoke(inst.LookupDef(fn), args_codegen()), x)

    return CodeGen(codegen)

  def visitSimpleValueIdentifier(
      self, ctx: tblgen.Parser.SimpleValueIdentifierContext
  ):
    identifier = ctx.getText()

    def codegen(x=None, *args):
      if x is None:
        return inst.LookupDef(identifier)
      return inst.Invoke(inst.LookupDef(identifier), x)

    return CodeGen(codegen)

  def visitValue(self, ctx: tblgen.Parser.ValueContext):
    if ctx.valueSuffix():
      raise NotImplementedError("valueSuffix is not implemented")
    if any(child.getText() == "#" for child in ctx.children):
      raise ValueError("Operator '#' is not supported in MatchDag")
    return self.visit(ctx.simpleValue())

  def visitDagArg(self, ctx: tblgen.Parser.DagArgContext):
    value = ctx.value()
    tok = ctx.TokVarName()

    def codegen(x, i):
      assign, var = self._gen_extract(x, i)

      if value:
        value_repr = value.getText().replace("\n", " ")
        begin_comment = inst.Comment(f"# === BEGIN: {value_repr} ===")
        value_code = CodeGen(self.visit(value))(var, 0)
      else:
        begin_comment = inst.Null()
        value_code = CodeGen()(var, 0)

      if tok:
        tok_code = inst.SetResult(tok.getText(), var)
      else:
        tok_code = inst.Null()

      return [assign, begin_comment, value_code, tok_code]

    return CodeGen(codegen)

  def visitDagArgList(self, ctx: tblgen.Parser.DagArgListContext):
    arg_codegens = [self.visit(dag_arg) for dag_arg in ctx.dagArg()]
    return CodeGen(
        lambda x, start_i: [
            codegen(x, i)
            for i, codegen in enumerate(arg_codegens, start=start_i)
        ]
    )

  def visitSimpleValueDag(self, ctx: tblgen.Parser.SimpleValueDagContext):
    dag_arg_list = ctx.dagArgList()

    dag_arg_codegen = self.visit(ctx.dagArg())
    dag_arg_list_codegen = CodeGen(
        self.visit(dag_arg_list) if dag_arg_list else None
    )

    def codegen(x, i):
      assign, new_var = self._gen_extract(x, i)
      return [
          assign,
          dag_arg_codegen(new_var, 0),
          dag_arg_list_codegen(new_var, 1),
      ]

    return CodeGen(codegen)
