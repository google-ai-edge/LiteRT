// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_UNOWNED_TENSOR_DESC_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_UNOWNED_TENSOR_DESC_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace litert::ml_drift {

// A TensorDescriptor that does not own the data. This is used to avoid
// copying data unnecessarily when creating a tensor from a descriptor.
// The original data must outlive the descriptor and the original allocator
// is responsible for ensuring the data is eventually freed.
//
// This is typically used to point to the mmap'd tflite file during model
// loading.
class UnownedDataTensorDescriptor : public ::ml_drift::TensorDescriptor {
 public:
  UnownedDataTensorDescriptor(const ::ml_drift::TensorDescriptor& desc,
                              absl::Span<const uint8_t> data)
      : ::ml_drift::TensorDescriptor(desc), unowned_data_(data) {}

  UnownedDataTensorDescriptor() = default;

  absl::Span<const uint8_t> GetData() const override { return unowned_data_; }

 private:
  absl::Span<const uint8_t> unowned_data_;
};

// A callback that can optionally be used to release the data that is owned by a
// UnownedDataTensorDescriptor.
using ReleaseDataCallback = std::unique_ptr<std::function<void()>>;

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_UNOWNED_TENSOR_DESC_H_
