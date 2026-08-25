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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_GRAPH_ADAPTER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_GRAPH_ADAPTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift

namespace ml_drift {

// A graph-agnostic view over the small set of graph operations that the
// SharedMemoryManager needs to register shared constant weights.
//
// The SharedMemoryManager historically operated directly on a GraphFloat32.
// This interface abstracts the ~handful of graph reads/mutations it performs so
// that the same orchestration logic can run on either a GraphFloat32 (via
// GraphFloat32Adapter) or an ir::IrModel (via IrModelAdapter).
//
// Ids passed across this interface are the underlying graph's native value/op
// ids. GraphFloat32 uses uint32_t ids directly; ir::IrModel uses size_t ids
// which are narrowed to uint32_t here (safe in practice, ids are small).
//
// Adapter methods are called at graph-construction time (not on any hot path),
// so virtual dispatch overhead is irrelevant.
class GraphAdapter {
 public:
  virtual ~GraphAdapter() = default;

  // ---- Shared-constant value descriptor access ----

  // Returns the BHWC shape currently recorded for `value_id`.
  virtual BHWC GetValueShape(uint32_t value_id) const = 0;

  // Sets the data type of `value_id`, leaving its shape (and any other
  // descriptor state the underlying graph tracks) unchanged.
  virtual void SetValueType(uint32_t value_id, DataType type) = 0;

  // Sets the shape and data type of `value_id` together, leaving any other
  // descriptor state the underlying graph tracks unchanged. The value id is
  // stable (the value is mutated in place, not replaced).
  virtual void SetValueShapeAndType(uint32_t value_id, const BHWC& shape,
                                    DataType type) = 0;

  // ---- Consumer op discovery / inspection ----

  // Returns the ids of the ops that consume `value_id`.
  virtual std::vector<uint32_t> FindConsumerOps(uint32_t value_id) const = 0;

  // Returns the operation-type name of `op_id` (comparable against
  // ToString(OperationType::...)).
  virtual std::string GetOpTypeName(uint32_t op_id) const = 0;

  // Returns whether `op_id` has at least one input value.
  virtual bool OpHasInputs(uint32_t op_id) const = 0;

  // Returns the shape of `op_id`'s first input value.
  // Precondition: OpHasInputs(op_id) is true.
  virtual BHWC GetOpFirstInputShape(uint32_t op_id) const = 0;

  // Returns the data type of `op_id`'s first input value.
  // Precondition: OpHasInputs(op_id) is true.
  virtual DataType GetOpFirstInputType(uint32_t op_id) const = 0;

  // ---- Graph mutation ----

  // Creates a new constant-input value carrying (`shape`, `type`), wires it as
  // an input of `consumer_op_id`, and returns its id. `global_tensor_id` is
  // recorded as the value's global reference where the underlying graph
  // supports it (GraphFloat32); implementations that do not track a global
  // reference (ir::IrModel) ignore it.
  virtual uint32_t AddConstantInput(uint32_t global_tensor_id,
                                    const BHWC& shape, DataType type,
                                    uint32_t consumer_op_id) = 0;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_GRAPH_ADAPTER_H_
