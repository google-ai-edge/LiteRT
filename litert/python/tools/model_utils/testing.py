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
"""Testing utilities for ModelUtils and MLIR."""

import functools

from litert.python.mlir import ir

from absl.testing import absltest as googletest
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils import transform
from litert.python.tools.model_utils.dialect import mlir


def run_in_ir_context(fn):

  @functools.wraps(fn)
  def new_fn(*args, **kwargs):
    with transform.get_ir_context():
      return fn(*args, **kwargs)

  return new_fn


def ir_text(
    module: mlir.ModuleOp | ir.Module | ir.Operation,
    *,
    enable_debug_info=False,
) -> str:
  """Converts the given module to MLIR text.

  Args:
    module: The module to convert.
    enable_debug_info: Whether to enable debug info.

  Returns:
    The MLIR text of the module.
  """
  if isinstance(module, mlir.ModuleOp):
    module = transform.convert_to_mlir(module)

  if isinstance(module, ir.Module):
    module = module.operation

  if not isinstance(module, ir.Operation):
    raise ValueError("Module must be an ir.Operation")

  module.verify()
  return module.get_asm(
      enable_debug_info=enable_debug_info,
      large_elements_limit=1000,
  )


def print_ir(
    name: str,
    module: mlir.ModuleOp | ir.Module | ir.Operation,
    *,
    enable_debug_info=False,
):
  """Prints the MLIR text of the given module.

  This is intended for use in FileCheck tests.

  Args:
    name: The name of the test. This is printed before the module in the form of
      "TEST: <name>".
    module: The module to print.
    enable_debug_info: Whether to enable debug info.
  """
  print("\nTEST:" + name, flush=True)
  print(ir_text(module, enable_debug_info=enable_debug_info), flush=True)


class ModelUtilsTestCase(googletest.TestCase):
  """Base class for ModelUtils tests."""

  def setUp(self):
    super().setUp()
    self.ir_context = transform.get_ir_context()
    self.enter_context(self.ir_context)

  def assert_filecheck(
      self,
      actual: mlir.ModuleOp | ir.Module | ir.Operation | str,
      check: str,
  ):
    """Assert that FileCheck runs successfully on the given actual output."""
    if isinstance(actual, (mlir.ModuleOp, ir.Module, ir.Operation)):
      actual = ir_text(actual)

    if not core.pybind.filecheck_check_input(actual, check):
      self.fail(f"Got output:\n\n{actual}\nExpected to match:\n{check}")
