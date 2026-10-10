# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

import flatbuffers

from absl.testing import absltest as googletest
from litert.tools import litert_lint_rules
from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import


def _make_tensor(
    shape, buffer_index: int = 0, tensor_type=schema.TensorType.FLOAT32
):
  tensor = schema.TensorT()
  tensor.shape = shape
  tensor.type = tensor_type
  tensor.buffer = buffer_index
  return tensor


def _make_buffer(data=None):
  buf = schema.BufferT()
  buf.data = data
  return buf


def _make_op_code(builtin_code, custom_code=None):
  op_code = schema.OperatorCodeT()
  op_code.builtinCode = builtin_code
  op_code.customCode = custom_code
  return op_code


def _make_op(
    opcode_index: int = 0,
    inputs=None,
    outputs=None,
    builtin_options=None,
    builtin_options_type: int = 0,
):
  op = schema.OperatorT()
  op.opcodeIndex = opcode_index
  op.inputs = inputs or []
  op.outputs = outputs or []
  if builtin_options is not None:
    op.builtinOptionsType = builtin_options_type
    op.builtinOptions = builtin_options
  return op


def _make_subgraph(tensors=None, operators=None, inputs=None, outputs=None):
  sg = schema.SubGraphT()
  sg.tensors = tensors or []
  sg.operators = operators or []
  if inputs is not None:
    sg.inputs = inputs
  if outputs is not None:
    sg.outputs = outputs
  return sg


def _make_model(subgraphs, buffers=None, op_codes=None, signature_defs=None):
  model = schema.ModelT()
  model.version = 3
  model.buffers = buffers if buffers is not None else [_make_buffer()]
  if op_codes is not None:
    model.operatorCodes = op_codes
  if signature_defs is not None:
    model.signatureDefs = signature_defs
  model.subgraphs = subgraphs
  return model


class LiteRtLintRulesTest(googletest.TestCase):

  def test_multisig_weight_sharing_pass(self):
    # Two subgraphs sharing the exact same buffer index 1 (150 KB)
    shared_data = b"\xaa" * (150 * 1024)
    model = _make_model(
        subgraphs=[
            _make_subgraph(tensors=[_make_tensor([150, 1024], buffer_index=1)]),
            _make_subgraph(tensors=[_make_tensor([150, 1024], buffer_index=1)]),
        ],
        buffers=[_make_buffer(), _make_buffer(shared_data)],
    )

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)
    self.assertEqual(result.total_weight_bytes, 150 * 1024)
    self.assertEqual(result.shared_weight_bytes, 150 * 1024)

  def test_multisig_weight_sharing_fail(self):
    # Two subgraphs with separate buffers holding identical 150 KB payload
    duplicate_data = b"\xbb" * (150 * 1024)
    model = _make_model(
        subgraphs=[
            _make_subgraph(tensors=[_make_tensor([150, 1024], buffer_index=1)]),
            _make_subgraph(tensors=[_make_tensor([150, 1024], buffer_index=2)]),
        ],
        buffers=[
            _make_buffer(),
            _make_buffer(duplicate_data),
            _make_buffer(duplicate_data),
        ],
    )

    result = litert_lint_rules.lint_model(model)
    self.assertFalse(result.is_passing())
    self.assertEqual(result.total_weight_bytes, 2 * 150 * 1024)
    self.assertEqual(result.shared_weight_bytes, 0)
    self.assertTrue(
        any("NO_UNSHARED_MULTISIG_WEIGHTS" in err for err in result.errors)
    )

  def test_activation_arena_budget_pass(self):
    # Subgraph with peak activation of 2 MB (500k float32) when budget is 10 MB.
    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[
                    _make_tensor([500000], buffer_index=0),
                    _make_tensor([500000], buffer_index=0),
                ],
                operators=[_make_op(inputs=[0], outputs=[1])],
            )
        ],
        op_codes=[_make_op_code(schema.BuiltinOperator.FULLY_CONNECTED)],
    )

    result = litert_lint_rules.lint_model(model, max_activation_mb=10.0)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)

  def test_activation_arena_budget_fail(self):
    # Subgraph with activation tensor requiring 40 MB when budget is 30 MB
    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[
                    _make_tensor([10 * 1024 * 1024], buffer_index=0),
                    _make_tensor([10 * 1024 * 1024], buffer_index=0),
                ],
                operators=[_make_op(inputs=[0], outputs=[1])],
            )
        ],
        op_codes=[_make_op_code(schema.BuiltinOperator.FULLY_CONNECTED)],
    )

    result = litert_lint_rules.lint_model(model, max_activation_mb=30.0)
    self.assertFalse(result.is_passing())
    self.assertTrue(
        any("MAX_ACTIVATION_PEAK_BUDGET" in err for err in result.errors)
    )

  def test_partition_fragmentation_pass(self):
    # Subgraph with homogeneous supported ops
    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[_make_tensor([10], buffer_index=0) for _ in range(5)],
                operators=[
                    _make_op(inputs=[i], outputs=[i + 1]) for i in range(4)
                ],
            )
        ],
        op_codes=[_make_op_code(schema.BuiltinOperator.FULLY_CONNECTED)],
    )

    result = litert_lint_rules.lint_model(model, max_partitions=3)
    self.assertEqual(result.partition_count, 1)
    self.assertEmpty(result.warnings)

  def test_partition_fragmentation_warn(self):
    # Subgraph alternating supported and unsupported ops: 5 partitions
    op_indices = [0, 1, 0, 1, 0]
    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[_make_tensor([10], buffer_index=0) for _ in range(6)],
                operators=[
                    _make_op(opcode_index=code_idx, inputs=[i], outputs=[i + 1])
                    for i, code_idx in enumerate(op_indices)
                ],
            )
        ],
        op_codes=[
            _make_op_code(schema.BuiltinOperator.FULLY_CONNECTED),
            _make_op_code(schema.BuiltinOperator.CUSTOM, "MY_CUSTOM_OP"),
        ],
    )

    result = litert_lint_rules.lint_model(model, max_partitions=3)
    self.assertGreater(result.partition_count, 3)
    self.assertTrue(
        any("PARTITION_FRAGMENTATION_CHECK" in w for w in result.warnings)
    )

  def test_end_to_end_model_file(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    model_path = temp_dir / "test_model.tflite"

    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[
                    _make_tensor([10, 16], buffer_index=0) for _ in range(3)
                ],
                operators=[_make_op(inputs=[0, 1], outputs=[2])],
            )
        ],
        op_codes=[_make_op_code(schema.BuiltinOperator.ADD)],
    )

    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    model_path.write_bytes(bytes(builder.Output()))

    result = litert_lint_rules.lint_model_file(str(model_path))
    self.assertTrue(result.is_passing())
    self.assertEqual(result.subgraphs_count, 1)
    self.assertGreater(result.peak_activation_bytes, 0)

  def test_model_file_not_found(self):
    result = litert_lint_rules.lint_model_file(
        "/nonexistent/path/to/model.tflite"
    )
    self.assertFalse(result.is_passing())
    self.assertTrue(any("not found" in err for err in result.errors))

  def test_scalar_and_empty_tensors(self):
    model = _make_model(
        subgraphs=[_make_subgraph(tensors=[_make_tensor([], buffer_index=0)])]
    )

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)

  def test_single_subgraph_weight_sharing_ignored(self):
    # In a single-subgraph model, duplicate buffers are not flagged by
    # multisig rule.
    duplicate_data = b"\xbb" * (150 * 1024)
    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[
                    _make_tensor([150, 1024], buffer_index=1),
                    _make_tensor([150, 1024], buffer_index=2),
                ]
            )
        ],
        buffers=[
            _make_buffer(),
            _make_buffer(duplicate_data),
            _make_buffer(duplicate_data),
        ],
    )

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)

  def test_orphan_subgraphs_warn_and_call_once_reachability(self):
    # Subgraph 0 (signature "serving_default") -> CALL_ONCE(Subgraph 1);
    # Subgraph 2 is an unreferenced orphan subgraph.
    call_once_opts = schema.CallOnceOptionsT()
    call_once_opts.initSubgraphIndex = 1
    sig = schema.SignatureDefT()
    sig.signatureKey = "serving_default"
    sig.subgraphIndex = 0

    model = _make_model(
        subgraphs=[
            _make_subgraph(
                operators=[
                    _make_op(
                        builtin_options=call_once_opts,
                        builtin_options_type=schema.BuiltinOptions.CallOnceOptions,
                    )
                ]
            ),
            _make_subgraph(),
            _make_subgraph(),
        ],
        op_codes=[_make_op_code(schema.BuiltinOperator.CALL_ONCE)],
        signature_defs=[sig],
    )

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEqual(result.orphan_subgraphs_count, 1)
    self.assertTrue(
        any("ORPHAN_SUBGRAPH_CHECK" in w and "#2" in w for w in result.warnings)
    )

  def test_activation_tensors_with_nonzero_empty_buffers_and_dead_tensors(self):
    # Modern converters often assign each activation tensor a distinct empty
    # BufferT entry (buffer_index > 0 with empty data). Those must still be
    # counted as activations, while dead unreferenced tensors must be ignored.
    # Tensor 0 and 1 are 2 MB each (live on op 0 -> 4 MB peak).
    # Tensor 2 is a 100 MB dead tensor not referenced by any op or subgraph IO.
    model = _make_model(
        subgraphs=[
            _make_subgraph(
                tensors=[
                    _make_tensor([500000], buffer_index=1),
                    _make_tensor([500000], buffer_index=2),
                    _make_tensor([25 * 1024 * 1024], buffer_index=3),
                ],
                operators=[_make_op(inputs=[0], outputs=[1])],
                inputs=[0],
                outputs=[1],
            )
        ],
        buffers=[_make_buffer() for _ in range(4)],
        op_codes=[_make_op_code(schema.BuiltinOperator.FULLY_CONNECTED)],
    )

    # With a 10 MB budget, the 4 MB live activations pass (dead 100 MB ignored),
    # and peak_activation_bytes reflects the 4 MB non-zero-empty-buffer tensors.
    result = litert_lint_rules.lint_model(model, max_activation_mb=10.0)
    self.assertTrue(result.is_passing())
    self.assertEqual(result.peak_activation_bytes, 4 * 1000000)


if __name__ == "__main__":
  googletest.main()
