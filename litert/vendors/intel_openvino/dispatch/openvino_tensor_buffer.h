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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_

#include <cstddef>

#include "openvino/runtime/tensor.hpp"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

class OpenVinoTensorBuffer {
 public:
  OpenVinoTensorBuffer(const OpenVinoTensorBuffer&) = delete;
  OpenVinoTensorBuffer& operator=(const OpenVinoTensorBuffer&) = delete;
  OpenVinoTensorBuffer(OpenVinoTensorBuffer&&) = default;
  OpenVinoTensorBuffer& operator=(OpenVinoTensorBuffer&&) = default;

  OpenVinoTensorBuffer() : host_tensor_(), allocated_(false) {};
  ~OpenVinoTensorBuffer() = default;

  litert::Expected<void> Alloc(const LiteRtRankedTensorType& tensor_type,
                               size_t size);

  litert::Expected<void*> GetTensorData();

  litert::Expected<ov::Tensor> GetOVTensor();

 private:
  ov::Tensor host_tensor_;
  bool allocated_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_
