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

#ifndef ODML_LITERT_LITERT_PYTHON_LITERT_MODEL_LITERT_MODEL_LIB_H_
#define ODML_LITERT_LITERT_PYTHON_LITERT_MODEL_LITERT_MODEL_LIB_H_

// Graph editing helpers on top of the C++ LiteRT IR (`LiteRtModelT`) used by
// the `litert_model` Python bindings.
//
// Everything here operates on the in-memory IR and reports failures through
// `litert::Expected` (no exceptions), so the logic can be unit tested in C++
// independently of pybind11. The helpers keep the doubly linked graph (op
// inputs/outputs <-> tensor users/defining op) consistent and refuse edits that
// would put `LiteRtSubgraphT::Ops()` out of topological order.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"

namespace litert::litert_model {

// Custom code of the ops produced by `OutlineSubgraphToDispatchOp`.
inline constexpr absl::string_view kDispatchOpCustomCode =
    internal::kLiteRtDispatchOpCustomName;

//===----------------------------------------------------------------------===//
// Element types.
//===----------------------------------------------------------------------===//

// Number of bits used to store one element of `element_type`, or 0 for types
// without a fixed width (none, string, resource, variant).
int ElementTypeBitWidth(LiteRtElementType element_type);

// Byte size of the dense (packed for sub-byte types) buffer holding a tensor of
// the given type and fully static `shape`. Returns std::nullopt if the shape is
// dynamic or the element type has no fixed width, and
// kLiteRtStatusErrorInvalidArgument if the size overflows size_t.
Expected<std::optional<size_t>> DenseByteSize(LiteRtElementType element_type,
                                              absl::Span<const int32_t> shape);

//===----------------------------------------------------------------------===//
// Queries.
//===----------------------------------------------------------------------===//

// Position of `op` in `subgraph.Ops()`.
Expected<size_t> FindOpIndex(const LiteRtSubgraphT& subgraph,
                             const LiteRtOpT& op);

// Position of `tensor` in `subgraph.Tensors()`.
Expected<size_t> FindTensorIndex(const LiteRtSubgraphT& subgraph,
                                 const LiteRtTensorT& tensor);

// Position of `subgraph` in `model.Subgraphs()`.
Expected<size_t> FindSubgraphIndex(const LiteRtModelT& model,
                                   const LiteRtSubgraphT& subgraph);

// Returns the subgraph of `model` owning `op`, or nullptr.
LiteRtSubgraphT* FindOwningSubgraph(const LiteRtModelT& model,
                                    const LiteRtOpT& op);

// True if `tensor` holds constant data (it has a non-empty weights buffer, or
// it is a zero-sized rank-1 tensor not produced by an op). Unlike
// `litert::internal::IsConstant` this never asserts.
bool IsConstantTensor(const LiteRtTensorT& tensor);

//===----------------------------------------------------------------------===//
// Op codes.
//===----------------------------------------------------------------------===//

// Returns the index of a TFLite operator code entry for `code` (and
// `custom_code` for custom ops) in the model's operator code table, appending a
// version 1 entry if none exists.
Expected<int32_t> GetOrAddTflOpCodeIndex(LiteRtModelT& model, LiteRtOpCode code,
                                         absl::string_view custom_code);

//===----------------------------------------------------------------------===//
// Builtin options.
//===----------------------------------------------------------------------===//

// Returns the (BuiltinOptions, BuiltinOptions2) union type tags of `op`.
std::pair<int32_t, int32_t> GetBuiltinOptionsTypes(const LiteRtOpT& op);

// Serializes the builtin options unions of `op` into a standalone flatbuffer
// whose root table is a `tflite::Operator` with only `builtin_options(_type)`
// and `builtin_options_2(_type)` set. The op is not modified.
OwningBufferRef<uint8_t> PackBuiltinOptions(LiteRtOpT& op);

// Inverse of `PackBuiltinOptions`: verifies `operator_flatbuffer` (a
// `tflite::Operator` root table) and replaces both builtin options unions of
// `op` with the ones it contains (a NONE union clears the corresponding
// options).
Expected<void> UnpackBuiltinOptions(BufferRef<uint8_t> operator_flatbuffer,
                                    LiteRtOpT& op);

//===----------------------------------------------------------------------===//
// Mutations. All of them keep def-use links consistent and keep
// `subgraph.Ops()` in topological order (when a tensor involved has a defining
// op). Transient states such as a tensor that is used but not (yet) defined are
// allowed; `ValidateModel` reports them.
//===----------------------------------------------------------------------===//

// Options for `AddOp`.
struct AddOpOptions {
  // Position in `subgraph.Ops()`; std::nullopt appends.
  std::optional<size_t> index;
  // Custom code, required for (and only allowed on) `kLiteRtOpCodeTflCustom`.
  std::optional<std::string> custom_code;
  // Opaque custom options.
  std::optional<std::vector<uint8_t>> custom_options;
};

// Creates a new op in `subgraph`. Output tensors must not be defined by another
// op, must not be constant and must not be subgraph inputs.
Expected<LiteRtOpT*> AddOp(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                           LiteRtOpCode code,
                           absl::Span<LiteRtTensorT* const> inputs,
                           absl::Span<LiteRtTensorT* const> outputs,
                           const AddOpOptions& options);

// Replaces input `input_index` of `op` with `tensor`.
Expected<void> SetOpInput(LiteRtSubgraphT& subgraph, LiteRtOpT& op,
                          size_t input_index, LiteRtTensorT& tensor);

// Replaces output `output_index` of `op` with `tensor`. The previous output
// loses its defining op.
Expected<void> SetOpOutput(LiteRtSubgraphT& subgraph, LiteRtOpT& op,
                           size_t output_index, LiteRtTensorT& tensor);

// Disconnects `op` from all its tensors and deallocates it. Tensors are kept.
// Any asset attached to the op is detached.
Expected<void> RemoveOp(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                        LiteRtOpT& op);

// Deallocates `tensor`, which must be unused: no uses, no defining op, not a
// subgraph input/output and not referenced by block-wise quantization.
Expected<void> RemoveTensor(LiteRtSubgraphT& subgraph, LiteRtTensorT& tensor);

// Redirects every use of `old_tensor` (except uses by ops in `exclude`) to
// `new_tensor`. Occurrences of `old_tensor` in the subgraph outputs and in the
// outputs of the signatures of `subgraph` are replaced as well.
Expected<void> ReplaceAllUses(
    LiteRtModelT& model, LiteRtSubgraphT& subgraph, LiteRtTensorT& old_tensor,
    LiteRtTensorT& new_tensor,
    const absl::flat_hash_set<const LiteRtOpT*>& exclude);

// Replaces the subgraph inputs and keeps the signatures of `subgraph` in sync.
Expected<void> SetSubgraphInputs(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                                 absl::Span<LiteRtTensorT* const> inputs);

// Replaces the subgraph outputs and keeps the signatures of `subgraph` in sync.
Expected<void> SetSubgraphOutputs(LiteRtModelT& model,
                                  LiteRtSubgraphT& subgraph,
                                  absl::Span<LiteRtTensorT* const> outputs);

// Dead code elimination to a fixpoint: removes ops whose outputs all have no
// uses and are not subgraph outputs (ops without outputs are kept), then the
// tensors left without defining op and uses (subgraph I/O and tensors
// referenced by block-wise quantization are kept). Returns whether anything was
// removed.
bool EliminateDeadCode(LiteRtModelT& model, LiteRtSubgraphT& subgraph);

// Re-derives the input/output names and tensors of every signature of
// `subgraph` from the current subgraph I/O (in place, pointer stable).
void SyncSignatures(LiteRtModelT& model, LiteRtSubgraphT& subgraph);

// A named signature tensor.
using NamedTensor = std::pair<std::string, LiteRtTensorT*>;

// Adds a signature. If the subgraph inputs (outputs) are empty they are set
// from the signature, otherwise the signature tensors must be exactly the
// subgraph inputs (outputs), in any order.
Expected<LiteRtSignatureT*> AddSignature(LiteRtModelT& model,
                                         absl::string_view key,
                                         LiteRtSubgraphT& subgraph,
                                         absl::Span<const NamedTensor> inputs,
                                         absl::Span<const NamedTensor> outputs);

// Removes the signature with `key`. Returns the removed (deallocated) pointer.
Expected<const LiteRtSignatureT*> RemoveSignature(LiteRtModelT& model,
                                                  absl::string_view key);

// Removes subgraphs not reachable from any signature (or from subgraph 0 when
// the model has no signatures) via StableHLO composite decomposition subgraph
// references, and remaps `decomposition_subgraph_index` on remaining
// composites. Returns the number of removed subgraphs, or 0 if control-flow
// operators (`WHILE`, `IF`, ...) prevent pruning
// (`kLiteRtStatusErrorUnsupported` from `PruneModelToSignatures` is tolerated
// without mutating the model).
Expected<size_t> PruneUnreachableSubgraphs(LiteRtModelT& model);

//===----------------------------------------------------------------------===//
// Validation.
//===----------------------------------------------------------------------===//

enum class ValidationLevel {
  // Only what `SerializeModel` needs to produce a faithful flatbuffer: op and
  // tensor membership, def-use consistency (`graph_validation`), operator code
  // table entries, block-wise quantization tensors and signatures matching
  // their subgraph I/O by position. Everything the C++ loader produces passes.
  kSerializable,
  // Additionally: topological order, every used tensor/output is constant, a
  // subgraph input or produced by an op, constants are not produced by ops,
  // unique subgraph inputs that are not produced by ops, ranks, unique
  // signature keys.
  kFull,
};

// Checks invariants of the whole model. Signature names are not required to be
// unique (the loader's placeholder signature may repeat output names). On
// failure returns kLiteRtStatusErrorInvalidGraphInvariant with all violations
// in the message.
Expected<void> ValidateModel(const LiteRtModelT& model,
                             ValidationLevel level = ValidationLevel::kFull);

//===----------------------------------------------------------------------===//
// Dispatch ops and assets.
//===----------------------------------------------------------------------===//

// Replaces all ops of `subgraph` by a single dispatch custom op whose inputs
// are the subgraph inputs and whose outputs are the subgraph outputs, then
// removes the now dead tensors (including constants left unused).
//
// Each subgraph output position becomes a distinct dispatch op output:
//   * the first occurrence of a tensor produced by an op is kept as is;
//   * an output not produced by an op (a constant or a forwarded subgraph
//     input), and every repeated occurrence of a tensor, is replaced by a fresh
//     tensor with the same name, type and quantization produced by the dispatch
//     op. The subgraph outputs and, positionally, the outputs of the signatures
//     of `subgraph` are remapped to it (signature names are kept, except that
//     repeated output names of the loader's placeholder signature
//     (`kDefaultSignatureKey`) become "<name>_<position>"). Forwarded inputs
//     stay subgraph inputs (and dispatch op inputs).
//
// Fails without modifying the model if the outputs of a signature of
// `subgraph` are not the subgraph outputs in order (see `ValidateModel`).
Expected<LiteRtOpT*> OutlineSubgraphToDispatchOp(LiteRtModelT& model,
                                                 LiteRtSubgraphT& subgraph);

// True if `op` is a dispatch op (custom code "DISPATCH_OP" or created by
// `MakeDispatchOp`).
bool IsDispatchOp(const LiteRtOpT& op);

// Attaches the model-owned buffer `buffer_id` to dispatch op `op` under the
// entry point `name` (see `LiteRtModelT::AttachAssetToOp`).
Expected<void> AttachAsset(LiteRtModelT& model, LiteRtOpT& op,
                           uint32_t buffer_id, absl::string_view name);

// Parses the dispatch options stored in the custom options of a dispatch op of
// a serialized-and-reloaded model. Returns kLiteRtStatusErrorNotFound if `op`
// has no dispatch options.
Expected<internal::DispatchOpOptions> ReadDispatchOpOptions(
    const LiteRtOpT& op);

}  // namespace litert::litert_model

#endif  // ODML_LITERT_LITERT_PYTHON_LITERT_MODEL_LITERT_MODEL_LIB_H_
