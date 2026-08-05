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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_KERNELS_AUDIO_FRONTEND_OVERLAP_ADD_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_KERNELS_AUDIO_FRONTEND_OVERLAP_ADD_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/c/c_api_types.h"

namespace litert {
namespace audio_frontend {

class OverlapAddKernel : public CustomOpKernel {
 public:
  Expected<void> Init(const void* init_data, size_t init_data_size) override;
  Expected<void> Destroy() override;

  // lite::CustomOpKernel methods.
  const std::string& OpName() const override { return kOpName; }
  int OpVersion() const override { return 1; };

  Expected<void> GetOutputLayouts(const std::vector<Layout>& input_layouts,
                                  std::vector<Layout>& output_layouts) override;

  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override;

 private:
  const std::string kOpName = "OverlapAdd";
  int32_t frame_step_ = 0;
  TfLiteType fft_type_ = kTfLiteNoType;

  // Cached values from GetOutputLayouts
  int32_t frame_size_ = 0;
  int32_t n_frames_ = 0;
  int32_t outer_dims_ = 0;

  uint8_t** state_buffers_ = nullptr;
};

}  // namespace audio_frontend
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_KERNELS_AUDIO_FRONTEND_OVERLAP_ADD_KERNEL_H_
