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
"""Utilities for accessing and updating flatbuffer signatures."""

import collections
import uuid

from xdsl import irdl

from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue


class SignatureBuilder:
  """A builder for flatbuffer signatures."""

  def __init__(self, func_op: func.FuncOp):
    if not isinstance(func_op, func.FuncOp):
      raise ValueError(
          f"SignatureBuilder can only be used for FuncOp. Got {type(func_op)}"
      )

    def _get_signature_name():
      array_attr = func_op.attributes.get("tf_saved_model.exported_names")
      if (
          array_attr
          and isinstance(array_attr, mlir.ArrayAttr)
          and len(array_attr) >= 1
          and isinstance(array_attr[0], mlir.StringAttr)
      ):
        return array_attr[0].data
      elif func_op.sym_name == "main":
        return "serving_default"
      else:
        return f"model_utils_signature_{uuid.uuid4().hex}"

    def _get_names_from_tf_index_path(attr_name: str):
      array_attr = func_op.attributes.get(attr_name)
      if not array_attr or not isinstance(array_attr, mlir.ArrayAttr):
        return None

      names = []
      for attr in array_attr:
        if not isinstance(attr, mlir.DictAttr):
          return None
        attr = attr.get("tf_saved_model.index_path")
        if not isinstance(attr, mlir.ArrayAttr):
          return None
        attr = attr[0]
        if not isinstance(attr, mlir.StringAttr):
          return None
        names.append(attr.data)
      return names

    self._func_op = func_op
    self._input_names = tuple(
        _get_names_from_tf_index_path("arg_attrs")
        or [f"args_{i}" for i in range(len(self.inputs))]
    )
    self._output_names = tuple(
        _get_names_from_tf_index_path("res_attrs")
        or [f"output_{i}" for i in range(len(self.outputs))]
    )
    self._name = _get_signature_name()

  def _apply_changes(self):
    """Updates the binded func_op's attributes with the signature changes."""
    self.func_op.update_function_type()

    self.func_op.attributes["tf_saved_model.exported_names"] = mlir.ArrayAttr(
        [mlir.StringAttr(self.name)]
    )

    def _update_tf_entry_function_names(
        key: str, names: list[str], name_suffix: str = ""
    ):
      if "tf.entry_function" not in self.func_op.attributes:
        self.func_op.attributes["tf.entry_function"] = mlir.DictAttr({})

      prefix = self.name + "_"
      self.func_op.attributes["tf.entry_function"][key] = mlir.StringAttr(
          ",".join([prefix + name + name_suffix for name in names])
      )

    if self.input_names:
      self.func_op.attributes["arg_attrs"] = mlir.ArrayAttr([
          mlir.DictAttr({
              "tf_saved_model.index_path": mlir.ArrayAttr(
                  [mlir.StringAttr(name)]
              ),
          })
          for name in self.input_names
      ])
      _update_tf_entry_function_names("inputs", self.input_names)
    else:
      self.func_op.attributes.pop("arg_attrs", None)
      _update_tf_entry_function_names("inputs", [])

    if self.output_names:
      self._func_op.attributes["res_attrs"] = mlir.ArrayAttr([
          mlir.DictAttr({
              "tf_saved_model.index_path": mlir.ArrayAttr(
                  [mlir.StringAttr(name)]
              ),
          })
          for name in self.output_names
      ])
      _update_tf_entry_function_names(
          "outputs", self.output_names, name_suffix="__output"
      )
    else:
      self.func_op.attributes.pop("res_attrs", None)
      _update_tf_entry_function_names("outputs", [])

  @property
  def func_op(self) -> func.FuncOp:
    """Binded func_op."""
    return self._func_op

  @property
  def inputs(self) -> tuple[SSAValue, ...]:
    """Input SSAValues (arguments of the func_op)."""
    return tuple(self.func_op.body.block.args)

  @property
  def outputs(self) -> tuple[SSAValue, ...]:
    """Output SSAValues (operands of the func.return op)."""
    return tuple(self.func_op.return_op.operands)

  @property
  def input_names(self) -> tuple[str, ...]:
    """Signature input names."""
    return tuple(self._input_names)

  @input_names.setter
  def input_names(self, names: tuple[str, ...]):
    """Sets the signature input names."""
    if len(self.input_names) != len(set(self.input_names)):
      raise ValueError(f"Input names are not unique: {self.input_names}")
    self._input_names = tuple(names)
    self._apply_changes()

  @property
  def output_names(self) -> tuple[str, ...]:
    """Signature output names."""
    return tuple(self._output_names)

  @output_names.setter
  def output_names(self, names: tuple[str, ...]):
    """Sets the signature output names."""
    if len(self.output_names) != len(set(self.output_names)):
      raise ValueError(f"Output names are not unique: {self.output_names}")
    self._output_names = tuple(names)
    self._apply_changes()

  @property
  def name(self) -> str:
    """Signature name."""
    return self._name

  @name.setter
  def name(self, name: str):
    """Sets the signature name."""
    self._name = str(name)
    self._apply_changes()

  def insert_input(self, arg_type: irdl.Attribute, index: int, name: str):
    """Inserts an argument to the func_op."""
    self.func_op.body.block.insert_arg(arg_type, index)

    input_names = list(self.input_names)
    input_names.insert(index, name)
    self.input_names = input_names

  def erase_input(self, index_or_name_or_value: int | str | SSAValue):
    """Erases an argument from the func_op."""
    index = None
    if isinstance(index_or_name_or_value, SSAValue):
      for i, value in enumerate(self.inputs):
        if value is index_or_name_or_value:
          index = i
          break
    elif isinstance(index_or_name_or_value, str):
      for i, name in enumerate(self.input_names):
        if name == index_or_name_or_value:
          index = i
          break
    else:
      index = index_or_name_or_value

    if index is None or index < 0 or index >= len(self.inputs):
      raise ValueError(f"Input not found: {index_or_name_or_value}")

    self.func_op.body.block.erase_arg(self.inputs[index])
    self.input_names = tuple(
        self.input_names[:index] + self.input_names[index + 1 :]
    )

  def insert_output(self, output: SSAValue, index: int, name: str):
    """Inserts an output to the func.return op."""
    operands = list(self.func_op.return_op.operands)
    self.func_op.return_op.operands = [
        *operands[:index],
        output,
        *operands[index:],
    ]

    output_names = list(self.output_names)
    output_names.insert(index, name)
    self.output_names = tuple(output_names)

  def erase_output(self, index_or_name_or_value: int | str | SSAValue):
    """Erases an output from the func.return op."""
    index = None
    if isinstance(index_or_name_or_value, SSAValue):
      for i, value in enumerate(self.outputs):
        if value is index_or_name_or_value:
          index = i
          break
    elif isinstance(index_or_name_or_value, str):
      for i, name in enumerate(self.output_names):
        if name == index_or_name_or_value:
          index = i
          break
    else:
      index = index_or_name_or_value

    if index is None or index < 0 or index >= len(self.outputs):
      raise ValueError(f"Output not found: {index_or_name_or_value}")

    operands = list(self.func_op.return_op.operands)
    self.func_op.return_op.operands = [
        *operands[:index],
        *operands[index + 1 :],
    ]
    self.output_names = tuple(
        self.output_names[:index] + self.output_names[index + 1 :]
    )

  def get_inputs_map(self) -> collections.OrderedDict[str, SSAValue]:
    """Returns a map of input names to SSAValues."""
    if len(self.input_names) != len(self.inputs):
      raise ValueError(
          "Input names and inputs are not the same length. "
          f"Input names: {self.input_names}, inputs: {self.inputs}"
      )
    return collections.OrderedDict(zip(self.input_names, self.inputs))

  def get_outputs_map(self) -> collections.OrderedDict[str, SSAValue]:
    """Returns a map of output names to SSAValues."""
    if len(self.output_names) != len(self.outputs):
      raise ValueError(
          "Output names and outputs are not the same length. "
          f"Output names: {self.output_names}, outputs: {self.outputs}"
      )
    return collections.OrderedDict(zip(self.output_names, self.outputs))

  def __repr__(self):
    return (
        f'SignatureBuilder(name="{self.name}", input_names={self.input_names},'
        f" output_names={self.output_names})"
    )
