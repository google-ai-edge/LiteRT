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

"""Unit tests for compiled_model.py API."""

import unittest
from unittest import mock

from litert.python.litert_wrapper.compiled_model_wrapper.compiled_model import CompiledModel


class CompiledModelApiTest(unittest.TestCase):
  """Unit tests for the CompiledModel API.

  These tests focus on verifying API behavior of the CompiledModel class.
  """

  def test_constructor(self):
    """Tests the basic constructor."""
    # Create a mock C++ model wrapper
    mock_model = mock.MagicMock()

    # Create a CompiledModel with the mock
    model = CompiledModel(mock_model)

    # Verify the model wrapper is stored properly
    self.assertEqual(model._model, mock_model)

  @mock.patch(
      "litert.python.litert_wrapper.compiled_model_wrapper._pywrap_litert_compiled_model_wrapper.CreateCompiledModelFromFile"
  )
  def test_from_file_api(self, mock_create_from_file):
    """Tests the from_file class method API."""
    # Configure mock
    mock_cpp_model = mock.MagicMock()
    mock_create_from_file.return_value = mock_cpp_model

    # Test with default parameters
    model_path = "/path/to/model.tflite"
    model = CompiledModel.from_file(model_path)
    mock_create_from_file.assert_called_once_with(model_path, "", "", 0)
    self.assertIsInstance(model, CompiledModel)

    # Reset mock for next test
    mock_create_from_file.reset_mock()

    # Test with all parameters specified
    model = CompiledModel.from_file(
        model_path,
        compiler_plugin="test_plugin",
        dispatch_library="test_lib",
        hardware_accel=1,
    )
    mock_create_from_file.assert_called_once_with(
        model_path, "test_plugin", "test_lib", 1
    )
    self.assertIsInstance(model, CompiledModel)

  @mock.patch(
      "litert.python.litert_wrapper.compiled_model_wrapper._pywrap_litert_compiled_model_wrapper.CreateCompiledModelFromBuffer"
  )
  def test_from_buffer_api(self, mock_create_from_buffer):
    """Tests the from_buffer class method API."""
    # Configure mock
    mock_cpp_model = mock.MagicMock()
    mock_create_from_buffer.return_value = mock_cpp_model

    # Sample model data
    model_data = b"mock model data"

    # Test with default parameters
    model = CompiledModel.from_buffer(model_data)
    mock_create_from_buffer.assert_called_once_with(model_data, "", "", 0)
    self.assertIsInstance(model, CompiledModel)

    # Reset mock for next test
    mock_create_from_buffer.reset_mock()

    # Test with all parameters specified
    model = CompiledModel.from_buffer(
        model_data,
        compiler_plugin="test_plugin",
        dispatch_library="test_lib",
        hardware_accel=1,
    )
    mock_create_from_buffer.assert_called_once_with(
        model_data, "test_plugin", "test_lib", 1
    )
    self.assertIsInstance(model, CompiledModel)

  def test_signature_methods_api(self):
    """Tests the signature-related method APIs."""
    # Create model with mock C++ wrapper
    mock_model = mock.MagicMock()
    model = CompiledModel(mock_model)

    # Configure mocks for signature methods
    mock_model.GetNumSignatures.return_value = 2
    mock_model.GetSignatureList.return_value = {
        "serving_default": {
            "inputs": ["input1", "input2"],
            "outputs": ["output"],
        }
    }
    mock_model.GetSignatureIndex.return_value = 0
    mock_model.GetSignatureByIndex.return_value = {
        "name": "serving_default",
        "inputs": ["input1", "input2"],
        "outputs": ["output"],
    }

    # Test get_num_signatures API
    num_signatures = model.get_num_signatures()
    mock_model.GetNumSignatures.assert_called_once()
    self.assertEqual(num_signatures, 2)

    # Test get_signature_list API
    sig_list = model.get_signature_list()
    mock_model.GetSignatureList.assert_called_once()
    self.assertEqual(
        sig_list,
        {
            "serving_default": {
                "inputs": ["input1", "input2"],
                "outputs": ["output"],
            }
        },
    )

    # Test get_signature_index API
    sig_idx = model.get_signature_index("serving_default")
    mock_model.GetSignatureIndex.assert_called_once_with("serving_default")
    self.assertEqual(sig_idx, 0)

    # Test get_signature_by_index API
    sig_info = model.get_signature_by_index(0)
    mock_model.GetSignatureByIndex.assert_called_once_with(0)
    self.assertEqual(
        sig_info,
        {
            "name": "serving_default",
            "inputs": ["input1", "input2"],
            "outputs": ["output"],
        },
    )

    # Setup for nonexistent signature test
    mock_model.GetSignatureIndex.reset_mock()
    mock_model.GetSignatureIndex.return_value = -1

    # Verify invalid signature name returns -1
    invalid_idx = model.get_signature_index("nonexistent_signature")
    mock_model.GetSignatureIndex.assert_called_once_with(
        "nonexistent_signature"
    )
    self.assertEqual(invalid_idx, -1)

  def test_buffer_requirements_api(self):
    """Tests buffer requirement method APIs."""
    # Create model with mock C++ wrapper
    mock_model = mock.MagicMock()
    model = CompiledModel(mock_model)

    # Configure mock returns
    mock_model.GetInputBufferRequirements.return_value = {
        "buffer_size": 16,
        "alignment": 64,
        "dtype": "float32",
    }
    mock_model.GetOutputBufferRequirements.return_value = {
        "buffer_size": 16,
        "alignment": 64,
        "dtype": "float32",
    }

    # Test get_input_buffer_requirements API
    input_req = model.get_input_buffer_requirements(0)
    mock_model.GetInputBufferRequirements.assert_called_once_with(0, 0)
    self.assertEqual(
        input_req, {"buffer_size": 16, "alignment": 64, "dtype": "float32"}
    )

    # Test get_output_buffer_requirements API
    output_req = model.get_output_buffer_requirements(0)
    mock_model.GetOutputBufferRequirements.assert_called_once_with(0, 0)
    self.assertEqual(
        output_req, {"buffer_size": 16, "alignment": 64, "dtype": "float32"}
    )

  @mock.patch(
      "litert.python.litert_wrapper.tensor_buffer_wrapper.tensor_buffer.TensorBuffer"
  )
  def test_buffer_creation_api(self, mock_tensor_buffer_class):
    """Tests buffer creation method APIs."""
    # Create model with mock C++ wrapper
    mock_model = mock.MagicMock()
    model = CompiledModel(mock_model)

    # Configure mocks
    mock_input_capsule = mock.MagicMock()
    mock_output_capsule = mock.MagicMock()
    mock_model.CreateInputBufferByName.return_value = mock_input_capsule
    mock_model.CreateOutputBufferByName.return_value = mock_output_capsule
    mock_model.CreateInputBuffers.return_value = [
        mock_input_capsule,
        mock_input_capsule,
    ]
    mock_model.CreateOutputBuffers.return_value = [mock_output_capsule]

    # Configure tensor buffer mock
    mock_buffer = mock.MagicMock()
    mock_tensor_buffer_class.return_value = mock_buffer

    # Test create_input_buffers API
    input_bufs = model.create_input_buffers(0)
    mock_model.CreateInputBuffers.assert_called_once_with(0)
    self.assertEqual(len(input_bufs), 2)
    # Just verify they're TensorBuffer instances with capsule attribute
    for buf in input_bufs:
      self.assertTrue(hasattr(buf, "capsule"))

    # Test create_output_buffers API
    output_bufs = model.create_output_buffers(0)
    mock_model.CreateOutputBuffers.assert_called_once_with(0)
    self.assertEqual(len(output_bufs), 1)
    # Just verify they're TensorBuffer instances with capsule attribute
    for buf in output_bufs:
      self.assertTrue(hasattr(buf, "capsule"))

    # Test create_input_buffer_by_name API
    input_buf = model.create_input_buffer_by_name("serving_default", "input1")
    mock_model.CreateInputBufferByName.assert_called_once_with(
        "serving_default", "input1"
    )
    self.assertTrue(hasattr(input_buf, "capsule"))

    # Test create_output_buffer_by_name API
    output_buf = model.create_output_buffer_by_name("serving_default", "output")
    mock_model.CreateOutputBufferByName.assert_called_once_with(
        "serving_default", "output"
    )
    self.assertTrue(hasattr(output_buf, "capsule"))

  def test_run_methods_api(self):
    """Tests the run method APIs."""
    # Create model with mock C++ wrapper
    mock_model = mock.MagicMock()
    model = CompiledModel(mock_model)

    # Create mock buffers and capsules
    mock_buffer = mock.MagicMock()
    mock_buffer.capsule = mock.MagicMock()

    # Setup for run_by_index
    input_buffers = [mock_buffer, mock_buffer]
    output_buffers = [mock_buffer]

    # Test run_by_index API
    model.run_by_index(0, input_buffers, output_buffers)
    expected_input_capsules = [mock_buffer.capsule, mock_buffer.capsule]
    expected_output_capsules = [mock_buffer.capsule]
    mock_model.RunByIndex.assert_called_once_with(
        0, expected_input_capsules, expected_output_capsules
    )

    # Setup for run_by_name
    input_map = {"input1": mock_buffer, "input2": mock_buffer}
    output_map = {"output": mock_buffer}

    # Test run_by_name API
    model.run_by_name("serving_default", input_map, output_map)
    expected_input_map = {
        "input1": mock_buffer.capsule,
        "input2": mock_buffer.capsule,
    }
    expected_output_map = {"output": mock_buffer.capsule}
    mock_model.RunByName.assert_called_once_with(
        "serving_default", expected_input_map, expected_output_map
    )


if __name__ == "__main__":
  unittest.main()
