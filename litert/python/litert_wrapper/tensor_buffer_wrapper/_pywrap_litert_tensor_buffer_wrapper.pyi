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

from typing import Any, List


def CreateTensorBufferFromHostMemory(
        py_data: Any,
        dtype: str,
        num_elements: int
) -> object:
    """Creates a TensorBuffer from existing host memory.

    Args:
      py_data: Python object exporting a contiguous readable buffer, such as a
        NumPy array or memoryview.
      dtype: Data type of the tensor elements as a string (e.g., 'float32',
        'float16').
      num_elements: Number of elements in the tensor.

    Returns:
      A PyCapsule object containing the LiteRT TensorBuffer.
    """
    ...


def WriteTensor(
        capsule: object,
        data_list: Any,
        dtype: str
) -> None:
    """Writes data into the tensor buffer.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer.
      data_list: Data to write to the tensor buffer. Can be a Python list of
        values or a contiguous buffer object such as a NumPy array.
      dtype: Data type of the tensor elements as a string (e.g., 'float32',
        'float16').
    """
    ...


def WriteTensorBuffer(
        capsule: object,
        py_data: Any,
        dtype: str
) -> None:
    """Writes raw contiguous tensor storage from a Python buffer object.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer.
      py_data: Python object exporting a contiguous readable buffer whose
        storage matches `dtype`.
      dtype: Data type of the tensor elements as a string.
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
      dtype: Data type of the tensor elements as a string (e.g., 'float32',
        'float16').

    Returns:
      A list containing the tensor data as Python scalar values.
    """
    ...


def ReadTensorToBuffer(
        capsule: object,
        py_data: Any,
        dtype: str
) -> None:
    """Reads raw contiguous tensor storage into a Python buffer object.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer.
      py_data: Python object exporting a contiguous writable buffer whose
        storage matches `dtype`.
      dtype: Data type of the tensor elements as a string.
    """
    ...


def GetTensorDetails(
        capsule: object
) -> Any:
    """Returns tensor details for the given tensor buffer.

    Args:
      capsule: PyCapsule object containing the LiteRT TensorBuffer.

    Returns:
      A dictionary containing at least the tensor `dtype` and `shape`.
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
