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
#include <optional>
#include <utility>

#include "litert/ats/common.h"
#include "litert/ats/configure.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/filesystem.h"
#include "litert/test/generators/generators.h"

namespace litert::testing {

// Gets the names of a potential future test case after consulting the options.
// Only increments test_id if a name is returned.
template <typename... Args>
std::optional<TestNames> NamesForNextTest(size_t& test_id,
                                          const AtsConf& options,
                                          Args&&... args) {
  auto names = TestNames::Create(test_id, std::forward<Args>(args)...);
  if (!options.ShouldRegister(absl::StrCat(names.suite, names.test))) {
    return std::nullopt;
  }
  test_id++;
  return names;
}

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
      if (options_.AtLimit(test_id_)) {
        return;
      }
      auto test_graph = Logic::Create(device);
      if (!test_graph) {
        LITERT_LOG(LITERT_WARNING, "Failed to create ATS test %lu, %s: %s",
                   test_id_, Logic::Name().data(),
                   test_graph.Error().Message().c_str());
        continue;
      }
      auto names = NamesForNextTest(
          test_id_, options_,
          absl::StrFormat("%s_%s", "SingleOp", Fixture::Name()), Logic::Name(),
          test_graph.Value()->Graph());
      if (!names) {
        continue;
      }
      Fixture::Register(std::move(*test_graph), options_, std::move(*names),
                        cap_.NewEntry());
    }
  }

  RegisterFunctor(size_t iters, size_t& test_id, const AtsConf& options,
                  Fixture::Capture& cap)
      : iters_(iters), test_id_(test_id), options_(options), cap_(cap) {}

 private:
  const size_t iters_;
  size_t& test_id_;
  const AtsConf& options_;
  Fixture::Capture& cap_;
};

// Specializes the given test logic template with the cartesian product of
// the given type lists and registers each specialization a given number
// of times. Each of these registrations will yield a single test case with a
// different set of random parameters.
template <typename Fixture, template <typename...> typename Logic,
          typename... Lists>
void RegisterCombinations(size_t iters, size_t& test_id, const AtsConf& options,
                          typename Fixture::Capture& cap) {
  RegisterFunctor<Fixture> f(iters, test_id, options, cap);
  ExpandProduct<Logic, Lists...>(f);
}

/// REGISTER EXTRA MODELS TESTS ////////////////////////////////////////////////

// Registers a test for each extra model passed in the options.
template <typename Fixture>
void RegisterExtraModels(size_t& test_id, const AtsConf& options,
                         typename Fixture::Capture& cap) {
  DefaultDevice device(options.GetSeedForParams(ExtraModel::Name()));
  const auto extra_models = options.ExtraModels();
  LITERT_LOG(LITERT_INFO, "Registering %zu extra models", extra_models.size());
  for (const auto& file : extra_models) {
    if (options.AtLimit(test_id)) {
      return;
    }
    auto model = ExtraModel::Create(file);
    if (!model) {
      LITERT_LOG(LITERT_WARNING, "Failed to create extra model %s: %s",
                 file.c_str(), model.Error().Message().c_str());
      continue;
    }
    auto test_name = internal::Filename(file);
    auto names =
        NamesForNextTest(test_id, options, Fixture::Name(), ExtraModel::Name(),
                         *test_name, file, "user provided tflite");
    if (!names) {
      continue;
    }
    Fixture::Register(std::move(*model), options, std::move(*names),
                      cap.NewEntry());
  }
}

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_H_
