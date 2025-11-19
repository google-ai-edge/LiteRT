// Copyright 2024 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_EXTRA_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_EXTRA_MODEL_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model_load.h"
#include "litert/test/generators/common.h"
#include "litert/test/simple_buffer.h"

namespace litert::testing {

// Container for test graphs that originate from raw .tflite passed to the test.
class ExtraModel : public TestGraph {
 public:
  using Ptr = std::unique_ptr<ExtraModel>;

  static constexpr absl::string_view Name() { return "ExtraModel"; }

  bool HasReference() const override { return false; }

  static Expected<ExtraModel::Ptr> Create(absl::string_view model_path) {
    LITERT_ASSIGN_OR_RETURN(auto model,
                            internal::LoadModelFromFile(model_path));
    return std::make_unique<ExtraModel>(std::move(model));
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    for (const auto& input : Graph().Subgraph(0).Inputs()) {
      LITERT_ASSIGN_OR_RETURN(auto t, (input->Ranked()));
      LITERT_ASSIGN_OR_RETURN(auto b,
                              SimpleBuffer::Create(RankedTensorType(t)));
      LITERT_RETURN_IF_ERROR((b.WriteRandom(data_builder, device)));
      inputs.push_back(std::move(b));
    }
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return Error(kLiteRtStatusErrorUnsupported);
  }

  explicit ExtraModel(LiteRtModelT::Ptr model) : TestGraph(std::move(model)) {}
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_EXTRA_MODEL_H_
