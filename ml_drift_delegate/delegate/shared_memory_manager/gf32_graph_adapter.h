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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_GF32_GRAPH_ADAPTER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_GF32_GRAPH_ADAPTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"

namespace ml_drift {

// GraphAdapter implementation backed by a GraphFloat32. Holds a reference to a
// caller-owned graph; the graph must outlive this adapter.
class GraphFloat32Adapter : public GraphAdapter {
 public:
  explicit GraphFloat32Adapter(GraphFloat32& graph) : graph_(graph) {}

  BHWC GetValueShape(uint32_t value_id) const override;
  void SetValueType(uint32_t value_id, DataType type) override;
  void SetValueShapeAndType(uint32_t value_id, const BHWC& shape,
                            DataType type) override;

  std::vector<uint32_t> FindConsumerOps(uint32_t value_id) const override;
  std::string GetOpTypeName(uint32_t op_id) const override;
  bool OpHasInputs(uint32_t op_id) const override;
  BHWC GetOpFirstInputShape(uint32_t op_id) const override;
  DataType GetOpFirstInputType(uint32_t op_id) const override;

  uint32_t AddConstantInput(uint32_t global_tensor_id, const BHWC& shape,
                            DataType type, uint32_t consumer_op_id) override;

 private:
  GraphFloat32& graph_;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_GF32_GRAPH_ADAPTER_H_
