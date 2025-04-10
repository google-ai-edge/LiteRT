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

import logging
import unittest

import numpy as np

from litert.python.litert_wrapper.compiled_model_wrapper.compiled_model import CompiledModel
from litert.python.litert_wrapper.tensor_buffer_wrapper.tensor_buffer import TensorBuffer
from google3.third_party.tensorflow.python.platform import resource_loader

# Paths to test model files
MODEL_FLOAT_FILE_NAME = "testdata/simple_model_float.tflite"
MODEL_INT8_FILE_NAME = "testdata/simple_model_int.tflite"

# Test data for float model
TEST_INPUT0_FLOAT = [1.0, 2.0, 3.0, 4.0]
TEST_INPUT1_FLOAT = [10.0, 20.0, 30.0, 40.0]
EXPECTED_OUTPUT_FLOAT = [11.0, 22.0, 33.0, 44.0]

# Test data for int8 model
TEST_INPUT_INT8 = [0, 10, 20, 30]
EXPECTED_OUTPUT_INT8 = [1, 11, 21, 31]


def get_model_path(model_filename):
  """Returns the absolute path to a test model file.

  Args:
    model_filename: Name of the model file in the testdata directory.

  Returns:
    String containing the absolute path to the model file.
  """
  return resource_loader.get_path_to_datafile(model_filename)


def aligned_array(shape, dtype, alignment=64):
  """Allocates a NumPy array with the specified memory alignment.

  Creates a NumPy array where the underlying memory is aligned to the specified
  byte boundary, which is useful for hardware-specific optimizations.

  Args:
    shape: The shape of the array.
    dtype: The data type of the array.
    alignment: The memory alignment in bytes (default: 64).

  Returns:
    A NumPy array with the specified alignment.
  """
  size = np.prod(shape)
  itemsize = np.dtype(dtype).itemsize
  n_extra = alignment // itemsize

  # Create a 1D array with extra space to find aligned memory
  raw = np.zeros(size + n_extra, dtype=dtype)

  # Find the first aligned position
  start_index = 0
  while (raw[start_index:].ctypes.data % alignment) != 0:
    start_index += 1

  # Return the aligned slice reshaped to the requested dimensions
  aligned = raw[start_index : start_index + size].reshape(shape)
  return aligned


class CompiledModelBasicTest(unittest.TestCase):

  def test_basic(self):
    """Tests basic functionality of the CompiledModel."""
    cm = CompiledModel.from_file(
        resource_loader.get_path_to_datafile(MODEL_FLOAT_FILE_NAME)
    )
    num_signatures = cm.get_num_signatures()
    self.assertEqual(num_signatures, 1)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    print(sig_idx)
    if sig_idx < 0:
      sig_idx = 0

    # Create input and output buffers
    input_bufs = cm.create_input_buffers(sig_idx)
    output_bufs = cm.create_output_buffers(sig_idx)

    req = cm.get_input_buffer_requirements(input_index=0)
    print("Required buffer size:", req["buffer_size"])

    self.assertEqual(len(input_bufs), 2)
    self.assertEqual(len(output_bufs), 1)

    # Populate input buffers with test data
    input_bufs[0].write(np.array(TEST_INPUT0_FLOAT, dtype=np.float32))
    input_bufs[1].write(np.array(TEST_INPUT1_FLOAT, dtype=np.float32))

    # Execute the model
    cm.run_by_index(sig_idx, input_bufs, output_bufs)

    # Retrieve and verify output values
    out_values = output_bufs[0].read(len(EXPECTED_OUTPUT_FLOAT), np.float32)
    logging.info("Output = %s", out_values)

    expected_output = np.array(EXPECTED_OUTPUT_FLOAT, dtype=np.float32)
    np.testing.assert_array_almost_equal(out_values, expected_output, decimal=5)

  def test_from_file_and_signatures(self):
    """Tests loading a model from file and accessing its signatures."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = CompiledModel.from_file(float_model_path)

    # Verify signature count
    num_sigs = cm.get_num_signatures()
    self.assertGreaterEqual(num_sigs, 1)

    # Retrieve and validate signature metadata
    sig_list = cm.get_signature_list()
    self.assertIn("<placeholder signature>", sig_list)
    serving_default_info = sig_list["<placeholder signature>"]
    self.assertIn("inputs", serving_default_info)
    self.assertIn("outputs", serving_default_info)

    # Verify signature lookup works
    sig_idx = cm.get_signature_index("<placeholder signature>")
    self.assertNotEqual(sig_idx, -1)

  def test_run_by_index_float(self):
    """Tests running inference on a float model using index-based API."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = CompiledModel.from_file(float_model_path)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx < 0:
      sig_idx = 0  # Fall back to first signature if name doesn't match

    # Create input & output buffers
    input_bufs = cm.create_input_buffers(sig_idx)
    output_bufs = cm.create_output_buffers(sig_idx)

    self.assertEqual(len(input_bufs), 2)
    self.assertEqual(len(output_bufs), 1)

    # Populate input buffers
    input_bufs[0].write(np.array(TEST_INPUT0_FLOAT, dtype=np.float32))
    input_bufs[1].write(np.array(TEST_INPUT1_FLOAT, dtype=np.float32))

    # Execute the model
    cm.run_by_index(sig_idx, input_bufs, output_bufs)

    # Verify inference results
    out_data = output_bufs[0].read(len(EXPECTED_OUTPUT_FLOAT), np.float32)
    expected_output = np.array(EXPECTED_OUTPUT_FLOAT, dtype=np.float32)
    self.assertEqual(len(out_data), len(EXPECTED_OUTPUT_FLOAT))
    np.testing.assert_array_almost_equal(out_data, expected_output, decimal=5)

    # Release tensor buffer resources
    for buf in input_bufs:
      buf.destroy()
    for buf in output_bufs:
      buf.destroy()

  def test_run_by_name_float(self):
    """Tests running inference on a float model using name-based API."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = CompiledModel.from_file(float_model_path)

    # Create buffers by tensor name
    in0_buf = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    in1_buf = cm.create_input_buffer_by_name("<placeholder signature>", "arg1")
    out_buf = cm.create_output_buffer_by_name(
        "<placeholder signature>", "tfl.add"
    )

    # Populate input buffers
    in0_buf.write(np.array(TEST_INPUT0_FLOAT, dtype=np.float32))
    in1_buf.write(np.array(TEST_INPUT1_FLOAT, dtype=np.float32))

    # Execute the model using name-based API
    input_map = {"arg0": in0_buf, "arg1": in1_buf}
    output_map = {"tfl.add": out_buf}
    cm.run_by_name("<placeholder signature>", input_map, output_map)

    # Verify inference results
    out_data = out_buf.read(len(EXPECTED_OUTPUT_FLOAT), np.float32)
    expected_output = np.array(EXPECTED_OUTPUT_FLOAT, dtype=np.float32)
    np.testing.assert_array_almost_equal(out_data, expected_output, decimal=5)

    # Release tensor buffer resources
    in0_buf.destroy()
    in1_buf.destroy()
    out_buf.destroy()

  def test_from_buffer(self):
    """Tests loading a model from an in-memory buffer."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    with open(float_model_path, "rb") as f:
      model_data = f.read()

    cm = CompiledModel.from_buffer(model_data)
    self.assertGreaterEqual(cm.get_num_signatures(), 1)

  def test_int32_model_inference(self):
    """Tests inference on an int8 quantized model."""
    int8_model_path = get_model_path(MODEL_INT8_FILE_NAME)
    cm = CompiledModel.from_file(int8_model_path)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    self.assertNotEqual(
        sig_idx, -1, "Model must have '<placeholder signature>' signature."
    )

    # Create tensor buffers
    input_bufs = cm.create_input_buffers(sig_idx)
    output_bufs = cm.create_output_buffers(sig_idx)
    self.assertEqual(len(input_bufs), 1)
    self.assertEqual(len(output_bufs), 1)

    # Populate input buffer
    input_bufs[0].write(np.array(TEST_INPUT_INT8, dtype=np.int32))

    # Execute the model
    cm.run_by_index(sig_idx, input_bufs, output_bufs)

    # Verify inference results
    out_data = output_bufs[0].read(len(TEST_INPUT_INT8), np.int32)
    expected_output = np.array(EXPECTED_OUTPUT_INT8, dtype=np.int32)
    np.testing.assert_array_equal(out_data, expected_output)

    # Release tensor buffer resources
    input_bufs[0].destroy()
    output_bufs[0].destroy()

  def test_zero_copy_input(self):
    """Tests creating an input buffer from existing memory (zero-copy)."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = CompiledModel.from_file(float_model_path)

    # Create aligned numpy array for zero-copy input
    arr = aligned_array((4,), np.float32)
    arr[:] = TEST_INPUT0_FLOAT

    # Create a TensorBuffer that references the existing memory without copying
    zero_copy_buf = TensorBuffer.create_from_host_memory(arr)

    # Create regular input buffer for second input
    input1_buf = cm.create_input_buffer_by_name(
        "<placeholder signature>", "arg1"
    )
    input1_buf.write(np.array(TEST_INPUT1_FLOAT, dtype=np.float32))

    # Create output buffer
    out_buf = cm.create_output_buffer_by_name(
        "<placeholder signature>", "tfl.add"
    )

    # Execute the model
    input_map = {"arg0": zero_copy_buf, "arg1": input1_buf}
    output_map = {"tfl.add": out_buf}
    cm.run_by_name("<placeholder signature>", input_map, output_map)

    # Verify inference results
    out_data = out_buf.read(len(EXPECTED_OUTPUT_FLOAT), np.float32)
    expected_output = np.array(EXPECTED_OUTPUT_FLOAT, dtype=np.float32)
    np.testing.assert_array_almost_equal(out_data, expected_output, decimal=5)

    # Release tensor buffer resources
    zero_copy_buf.destroy()
    input1_buf.destroy()
    out_buf.destroy()

  def test_destroy_buffer_twice(self):
    """Tests that destroying a tensor buffer multiple times is safe."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = CompiledModel.from_file(float_model_path)

    in0_buf = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    in0_buf.write(np.array(TEST_INPUT0_FLOAT, dtype=np.float32))

    # First destroy call
    in0_buf.destroy()

    # Second destroy call should not crash
    in0_buf.destroy()


if __name__ == "__main__":
  unittest.main()
