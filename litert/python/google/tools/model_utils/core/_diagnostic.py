import io

import xdsl
import xdsl.ir
import xdsl.utils.diagnostic


class OpFocusDiagnostic(xdsl.utils.diagnostic.Diagnostic):
  FOCUS_RANGE = 3

  def raise_exception(
      self,
      message: str,
      ir_node,
      exception_type: type[Exception] = RuntimeError,
      underlying_error: Exception | None = None,
  ):
    basic_raise_exception = lambda: super().raise_exception(
        message, ir_node, exception_type, underlying_error
    )

    if len(self.op_messages) != 1:
      basic_raise_exception()

    target_op = next(iter(self.op_messages.keys()))
    parent = target_op.parent

    if not isinstance(parent, xdsl.ir.Block):
      basic_raise_exception()

    ops = list(parent.ops)
    target_idx = ops.index(target_op)

    if target_idx is None:
      basic_raise_exception()

    f = io.StringIO()
    nullstream = io.StringIO()
    p = xdsl.printer.Printer(
        stream=f,
        diagnostic=self,
        print_generic_format=True,
    )

    for i, op in enumerate(ops[: target_idx + self.FOCUS_RANGE + 1]):
      if abs(i - target_idx) <= self.FOCUS_RANGE:
        p.stream = f
      else:
        p.stream = nullstream
      p._print_new_line()
      p.print_op(op)

    raise exception_type(message + "\n\n" + f.getvalue()) from underlying_error
