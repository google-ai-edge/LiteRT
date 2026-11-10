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

#ifndef ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_BIDIRECTIONAL_LSTM_BIDIRECTIONAL_LSTM_CUSTOM_OP_H_
#define ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_BIDIRECTIONAL_LSTM_BIDIRECTIONAL_LSTM_CUSTOM_OP_H_

#include <cstddef>
#include <string>
#include <vector>

#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace custom_ops {

// CPU kernel for a whole bidirectional LSTM layer emitted as a single
// `tfl.custom` node, instead of the thousands of nodes torch.export produces by
// unrolling the time loop.
//
// Nine inputs, in the order the exporter marks them:
//   0 x           [B, T, D] f32     5 bias_fwd   [4H]     f32
//   1 seq_lengths [B]       i32     6 w_ih_bwd   [4H, D]  f32
//   2 mask        [B, T]    f32     7 w_hh_bwd   [4H, H]  f32
//   3 w_ih_fwd    [4H, D]   f32     8 bias_bwd   [4H]     f32
//   4 w_hh_fwd    [4H, H]   f32
// One output:
//   0 y           [B, T, 2H] f32
//
// `mask` is 1.0 on padding positions. Callers with no mask pass zeros rather
// than dropping the operand, so the arity is fixed.
//
// The kernel is stateless on purpose. `CustomOpDispatcher` registers one kernel
// object for the whole resolver and calls `Init` once per node, so anything
// stored here would be shared by every node of every compiled model using these
// options -- a stale `hidden_size` from a sibling layer, or a scratch buffer
// raced between two interpreters. Shapes carry everything the math needs, so
// nothing has to be remembered.
class BidirectionalLstmCustomOpKernel : public CustomOpKernel {
 public:
  const std::string& OpName() const override;
  int OpVersion() const override;
  Expected<void> Init(const void* init_data, size_t init_data_size) override;
  Expected<void> GetOutputLayouts(const std::vector<Layout>& input_layouts,
                                  std::vector<Layout>& output_layouts) override;
  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override;
  Expected<void> Destroy() override;

 private:
  // The `custom_code` written by LegalizeCompositeToCustomOpPass is the
  // composite name verbatim, dotted prefix included, and the TFLite op resolver
  // matches on exactly this string.
  const std::string kOpName = "litert_custom_op.bidirectional_lstm";
};

// Registers the bidirectional LSTM kernel on `options`. Must be called before
// CompiledModel::Create; the kernel is a function-local static because
// LiteRtOptions stores a raw pointer to it that must outlive the model.
Expected<void> RegisterBidirectionalLstmCustomOp(Options& options);

}  // namespace custom_ops
}  // namespace litert

#endif  // ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_BIDIRECTIONAL_LSTM_BIDIRECTIONAL_LSTM_CUSTOM_OP_H_
