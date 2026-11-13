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

from absl.testing import parameterized
import flatbuffers

from absl.testing import absltest as googletest
from litert.tools import litert_lint_rules
from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import

_KB = 1024
_MB = 1024 * 1024
_FC = schema.BuiltinOperator.FULLY_CONNECTED
_LintConfig = litert_lint_rules.LintConfig


def _tensor(
    shape,
    buffer_index=0,
    external_buffer=0,
    tensor_type=schema.TensorType.FLOAT32,
):
  tensor = schema.TensorT()
  tensor.shape = shape
  tensor.type = tensor_type
  tensor.buffer = buffer_index
  tensor.externalBuffer = external_buffer
  return tensor


def _buffer(data=None):
  buf = schema.BufferT()
  buf.data = data
  return buf


def _op_code(builtin_code):
  op_code = schema.OperatorCodeT()
  op_code.builtinCode = builtin_code
  return op_code


def _op(inputs=None, outputs=None):
  op = schema.OperatorT()
  op.opcodeIndex = 0
  op.inputs = inputs or []
  op.outputs = outputs or []
  return op


def _options(cls, **fields):
  opts = cls()
  for name, value in fields.items():
    setattr(opts, name, value)
  return opts


def _subgraph(tensors=None, operators=None, inputs=None, outputs=None):
  sg = schema.SubGraphT()
  sg.tensors = tensors or []
  sg.operators = operators or []
  sg.inputs = inputs or []
  sg.outputs = outputs or []
  return sg


def _chain_subgraph(num_tensors, shape=(10,)):
  """Tensors t0..tN-1 and ops t[i] -> t[i+1]."""
  return _subgraph(
      tensors=[_tensor(list(shape)) for _ in range(num_tensors)],
      operators=[
          _op(inputs=[i], outputs=[i + 1]) for i in range(num_tensors - 1)
      ],
  )


def _model(subgraphs, buffers=None, op_codes=None, signature_roots=None):
  model = schema.ModelT()
  model.version = 3
  model.buffers = buffers if buffers is not None else [_buffer()]
  model.operatorCodes = op_codes or [_op_code(_FC)]
  model.subgraphs = subgraphs
  if signature_roots is not None:
    model.signatureDefs = []
    for i, root in enumerate(signature_roots):
      sig = schema.SignatureDefT()
      sig.signatureKey = f"sig{i}"
      sig.subgraphIndex = root
      model.signatureDefs.append(sig)
  return model


def _pack(model) -> bytes:
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
  return bytes(builder.Output())


def _write_model(test_case, data) -> str:
  path = pathlib.Path(test_case.create_tempdir().full_path) / "model.tflite"
  path.write_bytes(data)
  return str(path)


class WeightSharingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # (name, buffer indices used per subgraph, distinct payload per buffer,
      #  duplicate threshold KB, expect error)
      ("shared_index", [[1], [1]], [b"a"], 100, False),
      ("cross_subgraph_duplicate", [[1], [2]], [b"a", b"a"], 100, True),
      ("below_threshold", [[1], [2]], [b"a", b"a"], 200, False),
      ("intra_subgraph_duplicate", [[1, 2], [1]], [b"a", b"a"], 100, False),
      (
          "intra_subgraph_duplicate_mirrored",
          [[1], [1, 2]],
          [b"a", b"a"],
          100,
          False,
      ),
      (
          "duplicate_shared_by_later_subgraph",
          [[1, 2], [2]],
          [b"a", b"a"],
          100,
          False,
      ),
      ("single_subgraph_duplicate", [[1, 2]], [b"a", b"a"], 100, False),
      ("distinct_payloads", [[1], [2]], [b"a", b"b"], 100, False),
  )
  def test_inline_buffers(
      self, sg_buffers, payloads, threshold_kb, expect_error
  ):
    model = _model(
        subgraphs=[
            _subgraph(tensors=[_tensor([150, 256], b) for b in bufs])
            for bufs in sg_buffers
        ],
        buffers=[_buffer()] + [_buffer(p * (150 * _KB)) for p in payloads],
    )

    result = litert_lint_rules.lint_model(
        model, _LintConfig(large_buffer_threshold=threshold_kb * _KB)
    )

    self.assertEqual(result.is_passing(), not expect_error)
    self.assertEqual(result.total_weight_bytes, len(payloads) * 150 * _KB)
    if expect_error:
      self.assertLen(result.errors, 1)
      self.assertIn("Subgraph 1 buffer #2", result.errors[0])
      self.assertIn("buffer #1 used by subgraph 0", result.errors[0])

  def test_externalized_weights_are_shared_constants(self):
    # Two subgraphs reference the same external region through distinct
    # external buffer ids; tensor.buffer is 0 as written by the externalizer.
    ext_ids = [(1 << 31) | 0, (1 << 31) | 1]
    model = _model(
        subgraphs=[
            _subgraph(
                tensors=[
                    _tensor([10]),
                    _tensor([10 * _MB], external_buffer=ext_id),
                    _tensor([10]),
                ],
                operators=[_op(inputs=[0, 1], outputs=[2])],
                inputs=[0],
                outputs=[2],
            )
            for ext_id in ext_ids
        ],
        signature_roots=[0, 1],
    )
    model.externalBuffers = []
    for ext_id in ext_ids:
      ext = schema.ExternalBufferT()
      ext.id, ext.group, ext.offset, ext.length = ext_id, 1, 0, 40 * _MB
      model.externalBuffers.append(ext)

    result = litert_lint_rules.lint_model(
        model, _LintConfig(max_activation_mb=1.0)
    )

    self.assertTrue(result.is_passing())
    self.assertEqual(result.total_weight_bytes, 40 * _MB)
    self.assertEqual(result.peak_activation_bytes, 2 * 10 * 4)
    self.assertEmpty(result.warnings)


class ActivationBudgetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("within_budget", [500000], 10.0, 4 * 1000000, False),
      ("over_budget", [10 * _MB], 30.0, 80 * _MB, True),
      ("dynamic_dim_ignored", [-1, 10 * _MB], 1.0, 0, False),
  )
  def test_budget_is_a_warning(self, shape, budget_mb, peak, expect_warning):
    model = _model(subgraphs=[_chain_subgraph(2, shape)])

    result = litert_lint_rules.lint_model(
        model, _LintConfig(max_activation_mb=budget_mb)
    )

    self.assertTrue(result.is_passing())
    self.assertEqual(result.peak_activation_bytes, peak)
    if expect_warning:
      self.assertLen(result.warnings, 1)
      self.assertIn("MAX_ACTIVATION_PEAK_BUDGET", result.warnings[0])
      self.assertIn("Subgraph 0:", result.warnings[0])
    else:
      self.assertEmpty(result.warnings)

  def test_empty_buffers_count_as_activations_and_dead_tensors_ignored(self):
    # Converters often give each activation its own empty BufferT (index > 0).
    # Tensors 0 and 1 are live (2 MB each); tensor 2 (100 MB) is dead.
    model = _model(
        subgraphs=[
            _subgraph(
                tensors=[
                    _tensor([500000], buffer_index=1),
                    _tensor([500000], buffer_index=2),
                    _tensor([25 * _MB], buffer_index=3),
                ],
                operators=[_op(inputs=[0], outputs=[1])],
                inputs=[0],
                outputs=[1],
            )
        ],
        buffers=[_buffer() for _ in range(4)],
    )

    result = litert_lint_rules.lint_model(
        model, _LintConfig(max_activation_mb=10.0)
    )

    self.assertEmpty(result.warnings)
    self.assertEqual(result.peak_activation_bytes, 4 * 1000000)

  def test_variable_tensors_live_for_whole_subgraph(self):
    # The variable t3 is only written by the last op but persists across
    # invocations, so it overlaps t0 and t1 at op 0.
    sg = _chain_subgraph(4, shape=(1000,))
    sg.tensors[3].isVariable = True

    result = litert_lint_rules.lint_model(_model(subgraphs=[sg]))

    self.assertEqual(result.peak_activation_bytes, 3 * 4000)

  @parameterized.named_parameters(
      ("bfloat16", schema.TensorType.BFLOAT16, 2),
      ("float8_e4m3fn", schema.TensorType.FLOAT8_E4M3FN, 1),
      ("float8_e5m2", schema.TensorType.FLOAT8_E5M2, 1),
      ("uint4", schema.TensorType.UINT4, 1),
      ("int2", schema.TensorType.INT2, 1),
      ("resource", schema.TensorType.RESOURCE, 0),
      ("variant", schema.TensorType.VARIANT, 0),
  )
  def test_tensor_type_byte_sizes(self, tensor_type, elem_bytes):
    sg = _subgraph(
        tensors=[
            _tensor([100], tensor_type=tensor_type),
            _tensor([100], tensor_type=tensor_type),
        ],
        operators=[_op(inputs=[0], outputs=[1])],
        inputs=[0],
        outputs=[1],
    )

    result = litert_lint_rules.lint_model(_model(subgraphs=[sg]))

    self.assertEqual(result.peak_activation_bytes, 2 * 100 * elem_bytes)

  def test_subgraph_input_overwritten_later_stays_live(self):
    # Subgraph input t0 is read at op 0 and mutated in-place at op 2, so its
    # buffer spans [0, 2] and overlaps both t1 and t2 at op 1.
    sg = _subgraph(
        tensors=[_tensor([1000]) for _ in range(3)],
        operators=[
            _op(inputs=[0], outputs=[1]),
            _op(inputs=[1], outputs=[2]),
            _op(inputs=[2], outputs=[0]),
        ],
        inputs=[0],
        outputs=[2],
    )

    result = litert_lint_rules.lint_model(_model(subgraphs=[sg]))

    self.assertEqual(result.peak_activation_bytes, 3 * 4000)

  @parameterized.named_parameters(
      (
          "unread_second_write",
          [
              _op(inputs=[0], outputs=[1]),
              _op(inputs=[1], outputs=[2]),
              _op(inputs=[2], outputs=[3]),
              _op(inputs=[3], outputs=[1]),
          ],
          3,
      ),
      (
          "read_after_second_write",
          [
              _op(inputs=[0], outputs=[1]),
              _op(inputs=[0], outputs=[2]),
              _op(inputs=[2], outputs=[3]),
              _op(inputs=[3], outputs=[1]),
              _op(inputs=[1], outputs=[4]),
          ],
          4,
      ),
  )
  def test_multi_write_intermediate_tensor_spans_all_writes(
      self, operators, output_idx
  ):
    sg = _subgraph(
        tensors=[_tensor([1000]) for _ in range(output_idx + 1)],
        operators=operators,
        inputs=[0],
        outputs=[output_idx],
    )

    result = litert_lint_rules.lint_model(_model(subgraphs=[sg]))

    self.assertEqual(result.peak_activation_bytes, 3 * 4000)

  def test_orphan_subgraphs_are_skipped(self):
    model = _model(
        subgraphs=[
            _chain_subgraph(2),
            _chain_subgraph(8, shape=(100 * _MB,)),
        ]
    )

    result = litert_lint_rules.lint_model(model)

    self.assertEqual(result.peak_activation_bytes, 2 * 10 * 4)
    self.assertLen(result.warnings, 1)
    self.assertIn("ORPHAN_SUBGRAPH_CHECK", result.warnings[0])
    self.assertIn("(#1)", result.warnings[0])


class ReachabilityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # (name, builtin, (options slot, options type), options, orphans)
      (
          "call_once",
          schema.BuiltinOperator.CALL_ONCE,
          ("builtinOptions", schema.BuiltinOptions.CallOnceOptions),
          _options(schema.CallOnceOptionsT, initSubgraphIndex=1),
          [2],
      ),
      (
          "if",
          schema.BuiltinOperator.IF,
          ("builtinOptions", schema.BuiltinOptions.IfOptions),
          _options(schema.IfOptionsT, thenSubgraphIndex=1, elseSubgraphIndex=2),
          [],
      ),
      (
          "while",
          schema.BuiltinOperator.WHILE,
          ("builtinOptions", schema.BuiltinOptions.WhileOptions),
          _options(
              schema.WhileOptionsT, condSubgraphIndex=1, bodySubgraphIndex=2
          ),
          [],
      ),
      (
          "stablehlo_case",
          schema.BuiltinOperator.STABLEHLO_CASE,
          ("builtinOptions2", schema.BuiltinOptions2.StablehloCaseOptions),
          _options(schema.StablehloCaseOptionsT, branchSubgraphIndices=[1, 2]),
          [],
      ),
      (
          "stablehlo_composite",
          schema.BuiltinOperator.STABLEHLO_COMPOSITE,
          ("builtinOptions2", schema.BuiltinOptions2.StableHLOCompositeOptions),
          _options(
              schema.StableHLOCompositeOptionsT,
              name="c",
              decompositionSubgraphIndex=2,
          ),
          [1],
      ),
      (
          "stablehlo_custom_call",
          schema.BuiltinOperator.STABLEHLO_CUSTOM_CALL,
          (
              "builtinOptions2",
              schema.BuiltinOptions2.StablehloCustomCallOptions,
          ),
          _options(
              schema.StablehloCustomCallOptionsT,
              callTargetName="c",
              calledComputations=[2],
          ),
          [1],
      ),
  )
  def test_control_flow_callees_via_file(
      self, builtin, slot_type, options, expected_orphans
  ):
    op = _op()
    slot, options_type = slot_type
    setattr(op, slot, options)
    setattr(op, slot + "Type", options_type)
    model = _model(
        subgraphs=[_subgraph(operators=[op]), _subgraph(), _subgraph()],
        op_codes=[_op_code(builtin)],
        signature_roots=[0],
    )

    # Round-trip through a file so vector fields unpack as numpy arrays.
    result = litert_lint_rules.lint_model_file(_write_model(self, _pack(model)))

    self.assertTrue(result.is_passing())
    if expected_orphans:
      orphan_str = ", ".join(f"#{idx}" for idx in expected_orphans)
      self.assertLen(result.warnings, 1)
      self.assertIn(f"({orphan_str})", result.warnings[0])
    else:
      self.assertEmpty(result.warnings)

  def test_signature_roots_and_primary_subgraph(self):
    # Subgraph 0 is always a root, even when no signature references it.
    model = _model(
        subgraphs=[_subgraph(), _subgraph(), _subgraph()],
        signature_roots=[2],
    )

    result = litert_lint_rules.lint_model(model)

    self.assertLen(result.warnings, 1)
    self.assertIn("(#1)", result.warnings[0])


class ModelFileTest(googletest.TestCase):

  def _offset_model(self, payload_size):
    """Two subgraphs, each using one appended buffer of `payload_size`."""
    model = _model(
        subgraphs=[
            _subgraph(tensors=[_tensor([150, 256], 1)]),
            _subgraph(tensors=[_tensor([150, 256], 2)]),
        ],
        buffers=[_buffer(), _buffer(), _buffer()],
        signature_roots=[0, 1],
    )
    for buf in model.buffers[1:]:
      buf.offset, buf.size = 1, payload_size
    return model

  def test_offset_buffers_are_viewed(self):
    # >2 GB models store buffers after the FlatBuffer (Buffer.offset/size).
    # Two subgraphs with identical appended payloads must be flagged.
    payload = b"\xcc" * (150 * _KB)
    model = self._offset_model(len(payload))
    size = len(_pack(model))
    model.buffers[1].offset = size
    model.buffers[2].offset = size + len(payload)
    packed = _pack(model)
    self.assertLen(packed, size)

    result = litert_lint_rules.lint_model_file(
        _write_model(self, packed + payload + payload)
    )

    self.assertFalse(result.is_passing())
    self.assertEqual(result.total_weight_bytes, 2 * len(payload))
    self.assertIn("NO_DUPLICATE_CROSS_SUBGRAPH_WEIGHTS", result.errors[0])

  def test_sentinel_offsets_are_not_payloads(self):
    # Offset 1 means "no appended data"; reading view[1:1+size] would make
    # both buffers look like identical duplicates.
    model = self._offset_model(150 * _KB)
    packed = _pack(model)

    result = litert_lint_rules.lint_model_file(
        _write_model(self, packed + bytes(len(packed) + 150 * _KB))
    )

    self.assertTrue(result.is_passing())
    self.assertEqual(result.total_weight_bytes, 0)

  def test_offset_past_end_of_file_is_an_error(self):
    model = self._offset_model(150 * _KB)
    model.buffers[1].offset = 1 << 20
    model.buffers[2].offset = 0

    result = litert_lint_rules.lint_model_file(_write_model(self, _pack(model)))

    self.assertLen(result.errors, 1)
    self.assertIn("Buffer #1", result.errors[0])
    self.assertIn("past the end of the file", result.errors[0])

  def test_model_file_not_found(self):
    result = litert_lint_rules.lint_model_file("/nonexistent/model.tflite")

    self.assertFalse(result.is_passing())
    self.assertIn("not found or empty", result.errors[0])

  def test_main_exit_code(self):
    path = _write_model(self, _pack(_model([_chain_subgraph(2)])))

    self.assertEqual(litert_lint_rules.main(["--model", path]), 0)
    self.assertEqual(litert_lint_rules.main(["--model", "/nonexistent"]), 1)


if __name__ == "__main__":
  googletest.main()
