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

"""Python wrapper for LiteRT compiled models."""

import os
from typing import Any, Dict, List

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "compiled_model")
):
  # This file is part of litert package.
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      _pywrap_litert_compiled_model_wrapper as _cm,
  )
  from litert.python.litert_wrapper.compiled_model_wrapper import hardware_accelerator
  from litert.python.litert_wrapper.tensor_buffer_wrapper import tensor_buffer
  HardwareAccelerator = hardware_accelerator.HardwareAccelerator
  TensorBuffer = tensor_buffer.TensorBuffer
else:
  # This file is part of ai_edge_litert package.
  from ai_edge_litert import _pywrap_litert_compiled_model_wrapper as _cm
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator
  from ai_edge_litert.tensor_buffer import TensorBuffer
# pylint: enable=g-import-not-at-top


class CompiledModel:
  """Python wrapper for the C++ CompiledModelWrapper.

  This class provides methods to load, inspect, and execute machine learning
  models using the LiteRT runtime.
  """

  def __init__(self, c_model_ptr):
    """Initializes the CompiledModel with a C++ model pointer.

    Args:
      c_model_ptr: Pointer to the underlying C++ CompiledModelWrapper.
    """
    self._model = c_model_ptr  # Pointer to C++ CompiledModelWrapper

  @classmethod
  def from_file(
      cls,
      model_path: str,
      hardware_accel: HardwareAccelerator = HardwareAccelerator.CPU,
  ) -> "CompiledModel":
    """Creates a CompiledModel from a model file.

    Args:
      model_path: Path to the model file.
      hardware_accel: Hardware acceleration option. Use constants from
        HardwareAccelerator class (e.g., HardwareAccelerator.CPU,
        HardwareAccelerator.GPU). Defaults to CPU.

    Returns:
      A new CompiledModel instance.
    """
    ptr = _cm.CreateCompiledModelFromFile(
        model_path,
        runtime_path=os.path.dirname(os.path.abspath(__file__)),
        compiler_plugin_path="",
        dispatch_library_path="",
        hardware_accel=hardware_accel,
    )
    return cls(ptr)

  @classmethod
  def from_buffer(
      cls,
      model_data: bytes,
      hardware_accel: HardwareAccelerator = HardwareAccelerator.CPU,
  ) -> "CompiledModel":
    """Creates a CompiledModel from an in-memory buffer.

    Args:
      model_data: Model data as bytes.
      hardware_accel: Hardware acceleration option. Use constants from
        HardwareAccelerator class (e.g., HardwareAccelerator.CPU,
        HardwareAccelerator.GPU). Defaults to CPU.

    Returns:
      A new CompiledModel instance.
    """
    ptr = _cm.CreateCompiledModelFromBuffer(
        model_data,
        runtime_path=os.path.dirname(os.path.abspath(__file__)),
        compiler_plugin_path="",
        dispatch_library_path="",
        hardware_accel=hardware_accel,
    )
    return cls(ptr)

  def get_signature_list(self) -> Dict[str, Dict[str, List[str]]]:
    """Returns a dictionary of all available model signatures.

    Returns:
      Dictionary mapping signature names to their input/output specifications.
    """
    return self._model.GetSignatureList()

  def get_signature_by_index(self, index: int) -> Dict[str, Any]:
    """Returns signature information for the given index.

    Args:
      index: Index of the signature to retrieve.

    Returns:
      Dictionary containing signature information.
    """
    return self._model.GetSignatureByIndex(index)

  def get_num_signatures(self) -> int:
    """Returns the number of signatures in the model.

    Returns:
      Number of signatures.
    """
    return self._model.GetNumSignatures()

  def get_signature_index(self, key: str) -> int:
    """Returns the index for a signature name.

    Args:
      key: Name of the signature.

    Returns:
      Index of the signature, or -1 if not found.
    """
    return self._model.GetSignatureIndex(key)

  def get_input_tensor_details(self, signature_key: str) -> Dict[str, Any]:
    """Returns details of input tensors for a given signature.

    Args:
      signature_key: Name of the signature.

    Returns:
      Dictionary mapping input tensor names to their details (name, index,
      dtype, shape, etc.).
    """
    return self._model.GetInputTensorDetails(signature_key)

  def get_input_buffer_requirements(
      self, input_index: int, signature_index: int = 0
  ) -> Dict[str, Any]:
    """Returns memory requirements for an input tensor.

    Args:
      input_index: Index of the input tensor.
      signature_index: Index of the signature. Default is 0 (first signature).

    Returns:
      Dictionary with buffer requirements (size, alignment, etc.).
    """
    return self._model.GetInputBufferRequirements(signature_index, input_index)

  def get_output_buffer_requirements(
      self, output_index: int, signature_index: int = 0
  ) -> Dict[str, Any]:
    """Returns memory requirements for an output tensor.

    Args:
      output_index: Index of the output tensor.
      signature_index: Index of the signature. Default is 0 (first signature).

    Returns:
      Dictionary with buffer requirements (size, alignment, etc.).
    """
    return self._model.GetOutputBufferRequirements(
        signature_index, output_index
    )

  def create_input_buffer_by_name(
      self, signature_key: str, input_name: str
  ) -> TensorBuffer:
    """Creates an input TensorBuffer for the specified signature and input name.

    Args:
      signature_key: Name of the signature.
      input_name: Name of the input tensor.

    Returns:
      A TensorBuffer object for the specified input.
    """
    capsule = self._model.CreateInputBufferByName(signature_key, input_name)
    return TensorBuffer(capsule)

  def create_output_buffer_by_name(
      self, signature_key: str, output_name: str
  ) -> TensorBuffer:
    """Creates an output TensorBuffer for the specified signature and output name.

    Args:
      signature_key: Name of the signature.
      output_name: Name of the output tensor.

    Returns:
      A TensorBuffer object for the specified output.
    """
    capsule = self._model.CreateOutputBufferByName(signature_key, output_name)
    return TensorBuffer(capsule)

  def create_input_buffers(self, signature_index: int) -> List[TensorBuffer]:
    """Creates TensorBuffers for all inputs of the specified signature.

    Args:
      signature_index: Index of the signature.

    Returns:
      List of TensorBuffer objects for all inputs.
    """
    capsule_list = self._model.CreateInputBuffers(signature_index)
    return [TensorBuffer(c) for c in capsule_list]

  def create_output_buffers(self, signature_index: int) -> List[TensorBuffer]:
    """Creates TensorBuffers for all outputs of the specified signature.

    Args:
      signature_index: Index of the signature.

    Returns:
      List of TensorBuffer objects for all outputs.
    """
    capsule_list = self._model.CreateOutputBuffers(signature_index)
    return [TensorBuffer(c) for c in capsule_list]

  def run_by_name(
      self,
      signature_key: str,
      input_map: Dict[str, TensorBuffer],
      output_map: Dict[str, TensorBuffer],
  ) -> None:
    """Runs inference using the named signature and tensor maps.

    Args:
      signature_key: Name of the signature to execute.
      input_map: Dictionary mapping input names to TensorBuffer objects.
      output_map: Dictionary mapping output names to TensorBuffer objects.
    """
    # Convert TensorBuffer objects to raw capsules
    capsule_input_map = {k: v.capsule for k, v in input_map.items()}
    capsule_output_map = {k: v.capsule for k, v in output_map.items()}
    self._model.RunByName(signature_key, capsule_input_map, capsule_output_map)

  def run_by_index(
      self,
      signature_index: int,
      input_buffers: List[TensorBuffer],
      output_buffers: List[TensorBuffer],
  ) -> None:
    """Runs inference using the indexed signature and tensor lists.

    Args:
      signature_index: Index of the signature to execute.
      input_buffers: List of input TensorBuffer objects.
      output_buffers: List of output TensorBuffer objects.
    """
    input_capsules = [tb.capsule for tb in input_buffers]
    output_capsules = [tb.capsule for tb in output_buffers]
    self._model.RunByIndex(signature_index, input_capsules, output_capsules)
