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

#ifndef ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_LITERT_CUSTOM_OP_H_
#define ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_LITERT_CUSTOM_OP_H_

#include <cstddef>
#include <string>
#include <vector>

#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace gated_delta_net {

class TrilInvCustomOpKernel : public CustomOpKernel {
 public:
  const std::string& OpName() const override;
  int OpVersion() const override { return 1; }
  Expected<void> Init(const void* init_data, size_t init_data_size) override {
    return {};
  }
  Expected<void> GetOutputLayouts(const std::vector<Layout>& input_layouts,
                                  std::vector<Layout>& output_layouts) override;
  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override;
  Expected<void> Destroy() override { return {}; }

 private:
  const std::string kOpName = "gdn_tril_inv";
};

class GatedDeltaUpdateCustomOpKernel : public CustomOpKernel {
 public:
  const std::string& OpName() const override;
  int OpVersion() const override { return 1; }
  Expected<void> Init(const void* init_data, size_t init_data_size) override;
  Expected<void> GetOutputLayouts(const std::vector<Layout>& input_layouts,
                                  std::vector<Layout>& output_layouts) override;
  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override;
  Expected<void> Destroy() override { return {}; }

 private:
  const std::string kOpName = "gated_delta_update";
  int mode_ = 0;  // 0: recurrent, 1: chunked
};

// Registers Gated Delta Net custom op kernels (gated_delta_update and
// gdn_tril_inv) to the LiteRT compilation options.
Expected<void> RegisterGatedDeltaNetCustomOps(Options& options);

}  // namespace gated_delta_net
}  // namespace litert

#endif  // ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_LITERT_CUSTOM_OP_H_
