import dataclasses
import functools
import sys

from xdsl.irdl import *

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl
from litert.python.tools.model_utils.tool import mlir_opt

print = functools.partial(print, file=sys.stderr, flush=True)


@mlir_opt.register_module_pass()
@dataclasses.dataclass(kw_only=True)
class TestPase(mu.core.ModulePassBase):
  name = "test-pass"
  description = "!! This is a test pass for mlir-opt !!"

  def call(self, module: mlir.ModuleOp):
    print("!!!!!!!!TEST-PASS!!!!!!!")

    for op in module.walk():
      if op.name == tfl.PseudoConstOp.name:
        data = op.numpy()
        print("PSEUDO_CONST: DATA:::", data.shape, data.dtype)


if __name__ == "__main__":
  mlir_opt.main(sys.argv)
