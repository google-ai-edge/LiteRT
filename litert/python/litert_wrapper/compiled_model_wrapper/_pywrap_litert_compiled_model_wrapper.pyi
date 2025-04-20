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

"""Python type stubs for the LiteRT compiled model wrapper."""

from typing import Any, Dict, List

class CompiledModelWrapper:
    """Wrapper for a compiled LiteRT model."""

    def GetSignatureList(self) -> Dict[str, Dict[str, List[str]]]:
        """Returns a dictionary of all signatures in the model."""
        ...

    def GetSignatureByIndex(self, index: int) -> Dict[str, Any]:
        """Returns the signature at the specified index.

        Args:
          index: The index of the signature to retrieve.

        Returns:
          A dictionary containing the signature details.
        """
        ...

    def GetNumSignatures(self) -> int:
        """Returns the number of signatures in the model."""
        ...

    def GetSignatureIndex(self, key: str) -> int:
        """Returns the index of the signature with the specified key.

        Args:
          key: The signature key to look up.

        Returns:
          The index of the signature.
        """
        ...

    def GetInputBufferRequirements(self, sig_idx: int, in_idx: int) -> Dict[str, Any]:
        """Returns the requirements for an input buffer.

        Args:
          sig_idx: The signature index.
          in_idx: The input index within the signature.

        Returns:
          A dictionary containing the buffer requirements.
        """
        ...

    def GetOutputBufferRequirements(self, sig_idx: int, out_idx: int) -> Dict[str, Any]:
        """Returns the requirements for an output buffer.

        Args:
          sig_idx: The signature index.
          out_idx: The output index within the signature.

        Returns:
          A dictionary containing the buffer requirements.
        """
        ...

    def CreateInputBufferByName(self, sig_key: str, in_name: str) -> object:
        """Creates an input buffer for the specified signature and input name.

        Args:
          sig_key: The signature key.
          in_name: The input tensor name.

        Returns:
          A capsule object representing the tensor buffer.
        """
        ...

    def CreateOutputBufferByName(self, sig_key: str, out_name: str) -> object:
        """Creates an output buffer for the specified signature and output name.

        Args:
          sig_key: The signature key.
          out_name: The output tensor name.

        Returns:
          A capsule object representing the tensor buffer.
        """
        ...

    def CreateInputBuffers(self, sig_idx: int) -> List[object]:
        """Creates all input buffers for the specified signature.

        Args:
          sig_idx: The signature index.

        Returns:
          A list of capsule objects representing the tensor buffers.
        """
        ...

    def CreateOutputBuffers(self, sig_idx: int) -> List[object]:
        """Creates all output buffers for the specified signature.

        Args:
          sig_idx: The signature index.

        Returns:
          A list of capsule objects representing the tensor buffers.
        """
        ...

    def RunByName(self, sig_key: str, input_map: Dict[str, object], output_map: Dict[str, object]) -> None:
        """Runs inference using the specified signature and named tensors.

        Args:
          sig_key: The signature key.
          input_map: A dictionary mapping input names to tensor buffer capsules.
          output_map: A dictionary mapping output names to tensor buffer capsules.
        """
        ...

    def RunByIndex(self, sig_idx: int, input_caps_list: List[object], output_caps_list: List[object]) -> None:
        """Runs inference using the specified signature index and tensor lists.

        Args:
          sig_idx: The signature index.
          input_caps_list: A list of input tensor buffer capsules.
          output_caps_list: A list of output tensor buffer capsules.
        """
        ...

    def Run(self, input_caps_list: List[object], output_caps_list: List[object]) -> None:
        """Runs inference using the first signature (index 0) and tensor lists.

        Args:
          input_caps_list: A list of input tensor buffer capsules.
          output_caps_list: A list of output tensor buffer capsules.
        """
        ...

def CreateCompiledModelFromFile(
        model_path: str,
        compiler_plugin_path: str = ...,
        dispatch_library_path: str = ...,
        hardware_accel: int = ...
) -> CompiledModelWrapper:
    """Creates a compiled model from a model file.

    Args:
      model_path: Path to the model file.
      compiler_plugin_path: Optional path to the compiler plugin.
      dispatch_library_path: Optional path to the dispatch library.
      hardware_accel: Optional hardware acceleration flag.

    Returns:
      A CompiledModelWrapper instance.
    """
    ...


def CreateCompiledModelFromBuffer(
        model_data: bytes,
        compiler_plugin_path: str = ...,
        dispatch_library_path: str = ...,
        hardware_accel: int = ...
) -> CompiledModelWrapper:
    """Creates a compiled model from a model buffer.

    Args:
      model_data: The model data as bytes.
      compiler_plugin_path: Optional path to the compiler plugin.
      dispatch_library_path: Optional path to the dispatch library.
      hardware_accel: Optional hardware acceleration flag.

    Returns:
      A CompiledModelWrapper instance.
    """
    ...