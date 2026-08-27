/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"
#include "tensor/internal/type_id.h"

struct xnn_subgraph;

namespace litert::tensor {

// Tag to identify the XNNPACK mixin.
struct XnnpackMixinTag {};

struct XnnpackValue {
  graph::TensorInformation info;
  uint32_t id = XNN_INVALID_VALUE_ID;
  uint32_t flags = 0;
  LockedBufferSpan<const std::byte> data =
      LockedBufferSpan<const std::byte>::Empty();
};

class XnnpackGraph;
class TensorHandle;

// Context for building an XNNPACK subgraph.
class XnnpackBuildContext {
 public:
  explicit XnnpackBuildContext(
      std::vector<TensorHandle> outputs,
      absl::flat_hash_map<graph::Tensor, uint32_t> external_ids = {});
  ~XnnpackBuildContext();
  absl::Status Init();
  absl::StatusOr<std::unique_ptr<XnnpackGraph>> Finalize();
  // Defines a tensor in the XNNPACK subgraph.
  absl::StatusOr<uint32_t> DefineValue(const graph::Tensor& tensor);
  // Aliases `source` to `target`. Any call to `DefineValue(source)` will return
  // the XNNPACK ID of `target`.
  absl::Status AliasValue(const graph::Tensor& source,
                          const graph::Tensor& target);
  // Removes a tensor from the context.
  void RemoveTensor(const graph::Tensor& tensor);
  // Defines a constant tensor in the XNNPACK subgraph.
  // The data will be copied and its lifetime will be managed by the
  // graph/runner.
  absl::StatusOr<uint32_t> DefineConstant(const void* data, size_t bytes,
                                          ::xnn_datatype datatype,
                                          std::vector<size_t> shape);
  // Returns the XNNPACK subgraph.
  ::xnn_subgraph* subgraph();
  // Keeps a buffer alive for the duration of the graph's lifetime.
  void KeepAlive(std::shared_ptr<Buffer> buffer) {
    keep_alive_buffers_.push_back(std::move(buffer));
  }

 private:
  xnn_subgraph* subgraph_ = nullptr;
  std::vector<graph::Tensor> outputs_;
  std::vector<XnnpackValue> values_;
  absl::flat_hash_map<graph::Tensor, size_t> tensor_index_;
  absl::flat_hash_set<graph::Tensor> external_outputs_;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids_;
  std::vector<std::vector<float>> dequantized_buffers_;
  std::vector<std::vector<fp16_t>> fp16_buffers_;
  std::vector<std::vector<char>> constant_buffers_;
  std::vector<std::shared_ptr<Buffer>> keep_alive_buffers_;

  friend absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
      std::vector<TensorHandle> outputs);
};

// Base class for XNNPACK operations.
class XnnpackOperation : public graph::BackendExtension {
 public:
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<XnnpackOperation>();
  }
  // Converts the operation to XNNPACK.
  virtual absl::Status ToXnnpack(const graph::Operation& op,
                                 XnnpackBuildContext& ctx) const = 0;
};

namespace graph {

// XNNPACK mixin for the Add operation.
template <>
class OpMixin<AddOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MulOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SubOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DivOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaximumOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MinimumOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PowOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AbsOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SquareOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RsqrtOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqrtOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CeilOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SignOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RoundOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<NegOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TanhOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogisticOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CosOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CastOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DequantizeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Relu6Operation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LeakyReluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<EluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<HardSwishOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PReluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<L2NormalizationOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SinOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GeluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SoftmaxOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AveragePool2DOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaxPool2DOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Conv2DOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthwiseConv2DOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FullyConnectedOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<BatchMatMulOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MeanOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SliceOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ConcatenationOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReshapeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqueezeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpandDimsOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TileOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeBilinearOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConvOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConv2DOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GatherOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SpaceToDepthOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthToSpaceOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SplitOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RopeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};
}  // namespace graph

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_
