// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"
#include "pybind11/pybind11.h"  // from @pybind11

PYBIND11_MODULE(_litert_wrapper_utils_test_helper, m) {
  m.def("make_capsule", [](pybind11::object model_wrapper) {
    std::vector<int32_t> dims = {1};
    litert::RankedTensorType tensor_type =
        litert::MakeRankedTensorType<float>(dims);
    size_t buffer_size = sizeof(float);
    auto buffer_or =
        litert::TensorBuffer::CreateManagedHostMemory(tensor_type, buffer_size);
    if (!buffer_or) {
      throw std::runtime_error("Failed to create buffer");
    }
    litert::TensorBuffer buffer = std::move(buffer_or.Value());

    PyObject* model_ptr =
        model_wrapper.is_none() ? nullptr : model_wrapper.ptr();
    PyObject* capsule = litert::litert_wrapper_utils::MakeTensorBufferCapsule(
        buffer, model_ptr);
    return pybind11::reinterpret_steal<pybind11::object>(capsule);
  });
}
