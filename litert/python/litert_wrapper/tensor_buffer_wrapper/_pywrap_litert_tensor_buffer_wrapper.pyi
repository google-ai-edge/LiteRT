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

"""Python type stubs for the LiteRT TensorBuffer wrapper."""

from typing import Any, List, Union
import numpy as np


def CreateTensorBufferFromHostMemory(
        py_data: List,
        dtype: str,
        num_elements: int
) -> object:
    """Creates a TensorBuffer from existing host memory.

    Args:
      py_data: Python data to be used as the source for the tensor buffer.
        Can be a NumPy array (e.g., np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)).
      dtype: Data type of the tensor elements as a string (e.g., 'float32') or
        numpy dtype (e.g., np.float32).
      num_elements: Number of elements in the tensor.

    Returns:
      A PyCapsule object containing the LiteRT TensorBuffer.
    """
    ...


def WriteTensor(
        capsule: object,
        data_list: List,
        dtype: str
) -> None:
    """Writes data into the tensor buffer.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer.
      data_list: Data to write to the tensor buffer. Can be a list or NumPy array
        (e.g., np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)).
      dtype: Data type of the tensor elements as a string (e.g., 'float32') or
        numpy dtype (e.g., np.float32).
    """
    ...


def ReadTensor(
        capsule: object,
        num_elements: int,
        dtype: str
) -> List:
    """Reads data from the tensor buffer.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer.
      num_elements: Number of elements to read from the buffer.
      dtype: Data type of the tensor elements as a string (e.g., 'float32') or
        numpy dtype (e.g., np.float32).

    Returns:
      A list containing the tensor data. Can be converted to NumPy array if needed:
      e.g., np.array(output_data, dtype=np.float32).reshape((1, 4))
    """
    ...


def DestroyTensorBuffer(
        capsule: object
) -> None:
    """Destroys the tensor buffer and releases associated resources.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer to destroy.
    """
    ...