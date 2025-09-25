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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_H_

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/configure.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_rng.h"
#include "litert/core/model/model_load.h"
#include "litert/test/generators/generators.h"
#include "litert/test/simple_buffer.h"

namespace litert::testing {

/// REGISTER GENERATED TESTS ///////////////////////////////////////////////////

// Utility to register a test logic a given number of times with a common
// random device.
template <typename Fixture>
class RegisterFunctor {
 public:
  template <typename Logic>
  void operator()() {
    DefaultDevice device(options_.GetSeedForParams(Logic::Name()));
    for (size_t i = 0; i < iters_; ++i) {
      auto test_graph = Logic::Create(device);
      if (!test_graph) {
        LITERT_LOG(LITERT_WARNING, "Failed to create ATS test %lu, %s: %s",
                   test_id_, Logic::Name().data(),
                   test_graph.Error().Message().c_str());
        continue;
      }
      Fixture::Register(test_id_++, std::move(*test_graph), Logic::Name(),
                        options_);
    }
  }

  RegisterFunctor(size_t iters, size_t& test_id, const AtsConf& options)
      : iters_(iters), test_id_(test_id), options_(options) {}

 private:
  const size_t iters_;
  size_t& test_id_;
  const AtsConf& options_;
};

// Specializes the given test logic template with the cartesian product of
// the given type lists and registers each specialization a given number
// of times. Each of these registrations will yield a single test case with a
// a different set of random parameters.
template <typename Fixture, template <typename...> typename Logic,
          typename... Lists>
void RegisterCombinations(size_t iters, size_t& test_id,
                          const AtsConf& options) {
  RegisterFunctor<Fixture> f(iters, test_id, options);
  ExpandProduct<Logic, Lists...>(f);
}

/// REGISTER EXTRA MODELS TESTS ////////////////////////////////////////////////

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

// Registers a test for each extra model passed in the options.
template <typename Fixture>
void RegisterExtraModels(size_t& test_id, const AtsConf& options) {
  DefaultDevice device(options.GetSeedForParams(ExtraModel::Name()));
  const auto extra_models = options.ExtraModels();
  LITERT_LOG(LITERT_INFO, "Registering %zu extra models", extra_models.size());
  for (const auto& file : extra_models) {
    auto model = ExtraModel::Create(file);
    if (!model) {
      LITERT_LOG(LITERT_WARNING, "Failed to create extra model %s: %s",
                 file.c_str(), model.Error().Message().c_str());
      continue;
    }
    Fixture::Register(test_id++, std::move(*model), ExtraModel::Name(), options,
                      /*always_reg=*/true);
  }
}

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_H_
