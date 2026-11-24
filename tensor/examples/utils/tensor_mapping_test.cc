/* Copyright 2026 Google LLC.

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

#include "tensor/examples/utils/tensor_mapping.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::IsEmpty;

// Writes a safetensors file holding two small FP32 tensors named
// `checkpoint.a.weight` and `checkpoint.b.weight`.
SafetensorFileGuard CreateMappingTestSafetensor() {
  return CreateTempSafetensor(
      {
          {.name = "checkpoint.a.weight",
           .type = Type::kFP32,
           .shape = {2},
           .buffer = std::vector<float>({1, 2})},
          {.name = "checkpoint.b.weight",
           .type = Type::kFP32,
           .shape = {3},
           .buffer = std::vector<float>({3, 4, 5})},
      },
      /*quant_config_json=*/"");
}

// Maps an existing checkpoint tensor and a missing one.
absl::flat_hash_map<std::string, std::string> MappingTestNames() {
  return {{"checkpoint.a.weight", "model.a"},
          {"checkpoint.missing.weight", "model.missing"}};
}

// Creates a mapping over the checkpoint written in `file`.
absl::StatusOr<LazyTensorMapping> CreateMappingTestMapping(
    const SafetensorFileGuard& file) {
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(file.GetPath()));
  return LazyTensorMapping(MappingTestNames(), std::move(loader));
}

TensorHandle MappingTestTensor() {
  return TensorHandle({.name = "tensor",
                       .type = Type::kFP32,
                       .shape = {2},
                       .buffer = std::vector<float>({1, 2})});
}

// Records the hook calls in `log` as "<id>.<hook>(<model_name>)".
class RecordingHooks : public TensorMappingHooks {
 public:
  RecordingHooks(std::vector<std::string>& log, std::string id)
      : log_(log), id_(std::move(id)) {}

  absl::Status OnLoaded(absl::string_view model_name,
                        TensorHandle& tensor) override {
    log_.push_back(absl::StrCat(id_, ".OnLoaded(", model_name, ")"));
    return absl::OkStatus();
  }

  absl::StatusOr<TensorHandle> OnNotFound(
      TensorMapping& mapping, absl::string_view model_name) override {
    log_.push_back(absl::StrCat(id_, ".OnNotFound(", model_name, ")"));
    return TensorMappingHooks::OnNotFound(mapping, model_name);
  }

 private:
  std::vector<std::string>& log_;
  std::string id_;
};

// Provides "model.alias" from "model.a".
class AliasHooks : public TensorMappingHooks {
 public:
  absl::StatusOr<TensorHandle> OnNotFound(
      TensorMapping& mapping, absl::string_view model_name) override {
    if (model_name == "model.alias") {
      return mapping.Get("model.a");
    }
    return TensorMappingHooks::OnNotFound(mapping, model_name);
  }
};

// Fails every hook call.
class FailingHooks : public TensorMappingHooks {
 public:
  absl::Status OnLoaded(absl::string_view model_name,
                        TensorHandle& tensor) override {
    return absl::InternalError("OnLoaded failure");
  }

  absl::StatusOr<TensorHandle> OnNotFound(
      TensorMapping& mapping, absl::string_view model_name) override {
    return absl::InternalError("OnNotFound failure");
  }
};

TEST(LazyTensorMappingTest, GetsTensorsProvidedAtConstruction) {
  LazyTensorMapping mapping({{"model.a", MappingTestTensor()}});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle result, mapping.Get("model.a"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, result.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(),
              ElementsAre(FloatEq(1), FloatEq(2)));
  EXPECT_THAT(mapping.Get("model.b"), StatusIs(absl::StatusCode::kNotFound));
}

TEST(LazyTensorMappingTest, LoadsTensorsOnDemand) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  // Tensors that are missing from the checkpoint don't fail the creation.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor, mapping.Get("model.a"));
  EXPECT_EQ(tensor.GetName(), "model.a");
  EXPECT_THAT(tensor.GetShape(), ElementsAre(2));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(),
              ElementsAre(FloatEq(1), FloatEq(2)));

  // Subsequent calls return the tensor that was already loaded.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle again, mapping.Get("model.a"));
  EXPECT_EQ(again.GetBufferPtr(), tensor.GetBufferPtr());
}

TEST(LazyTensorMappingTest, FailsForMissingTensors) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));

  // Mapped but not in the checkpoint.
  EXPECT_THAT(mapping.Get("model.missing"),
              StatusIs(absl::StatusCode::kNotFound));
  // In the checkpoint but not mapped.
  EXPECT_THAT(mapping.Get("checkpoint.b.weight"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(mapping.Get("model.b"), StatusIs(absl::StatusCode::kNotFound));
}

TEST(LazyTensorMappingTest, CallsOnLoadedOncePerTensor) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));
  std::vector<std::string> log;
  mapping.Register<RecordingHooks>(log, "hooks");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor, mapping.Get("model.a"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle again, mapping.Get("model.a"));
  EXPECT_EQ(again.GetBufferPtr(), tensor.GetBufferPtr());
  EXPECT_THAT(log, ElementsAre("hooks.OnLoaded(model.a)"));
}

TEST(LazyTensorMappingTest, ChainsHooksInRegistrationOrder) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));
  std::vector<std::string> log;
  mapping.Register<RecordingHooks>(log, "first")
      .Register<AliasHooks>()
      .Register<RecordingHooks>(log, "last");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle alias,
                                  mapping.Get("model.alias"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor, mapping.Get("model.a"));
  EXPECT_EQ(alias.GetBufferPtr(), tensor.GetBufferPtr());
  // Every `OnLoaded()` is called. `OnNotFound()` stops at the hooks that
  // provide the tensor.
  EXPECT_THAT(
      log,
      ElementsAre("first.OnNotFound(model.alias)", "first.OnLoaded(model.a)",
                  "last.OnLoaded(model.a)", "first.OnLoaded(model.alias)",
                  "last.OnLoaded(model.alias)"));

  log.clear();
  EXPECT_THAT(mapping.Get("model.missing"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(log, ElementsAre("first.OnNotFound(model.missing)",
                               "last.OnNotFound(model.missing)"));
}

TEST(LazyTensorMappingTest, RegisterAppendsHooks) {
  std::vector<std::string> log;
  LazyTensorMapping mapping =
      LazyTensorMapping({{"model.a", MappingTestTensor()}})
          .Register<AliasHooks>()
          .Register<RecordingHooks>(log, "hooks");

  // Tensors provided at construction don't go through the hooks.
  ASSERT_THAT(mapping.Get("model.a"), IsOk());
  EXPECT_THAT(log, IsEmpty());

  ASSERT_THAT(mapping.Get("model.alias"), IsOk());
  EXPECT_THAT(log, ElementsAre("hooks.OnLoaded(model.alias)"));
}

TEST(LazyTensorMappingTest, SetBypassesHooks) {
  std::vector<std::string> log;
  LazyTensorMapping mapping;
  mapping.Register<RecordingHooks>(log, "hooks");

  ASSERT_THAT(mapping.Set("model.a", MappingTestTensor()), IsOk());
  ASSERT_THAT(mapping.Get("model.a"), IsOk());
  EXPECT_THAT(log, IsEmpty());
}

TEST(LazyTensorMappingTest, FailsWhenOnLoadedFails) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));
  mapping.Register<FailingHooks>();

  EXPECT_THAT(mapping.Get("model.a"), StatusIs(absl::StatusCode::kInternal));
}

TEST(LazyTensorMappingTest, OnNotFoundErrorStopsTheChain) {
  std::vector<std::string> log;
  LazyTensorMapping mapping;
  mapping.Register<FailingHooks>().Register<RecordingHooks>(log, "hooks");

  EXPECT_THAT(mapping.Get("model.a"), StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(log, IsEmpty());
}

TEST(LazyTensorMappingTest, ReleaseTensorsNotPinnedSkipsPinnedAndInMemory) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping loaded,
                                  CreateMappingTestMapping(file));
  ASSERT_THAT(loaded.Get("model.a"), IsOk());
  ASSERT_THAT(loaded.Set("model.pinned", MappingTestTensor()), IsOk());
  EXPECT_EQ(loaded.ReleaseTensorsNotPinned(), 1);
  // Loader-backed tensor is reloaded on demand; pinned tensor stays intact.
  ASSERT_THAT(loaded.Get("model.a"), IsOk());
  ASSERT_THAT(loaded.Get("model.pinned"), IsOk());

  // In-memory mappings without a loader never drop preloaded tensors.
  LazyTensorMapping in_memory({{"model.mem", MappingTestTensor()}});
  EXPECT_EQ(in_memory.ReleaseTensorsNotPinned(), 0);
  ASSERT_THAT(in_memory.Get("model.mem"), IsOk());
}

// With a loader function, `Get()` calls it again for a released tensor.
TEST(LazyTensorMappingTest, ReleaseTensorsNotPinnedReloadsWithLoaderFn) {
  int num_loads = 0;
  LazyTensorMapping mapping(
      {{"checkpoint.x", "model.x"}},
      [&num_loads](absl::string_view) -> absl::StatusOr<TensorHandle> {
        ++num_loads;
        return MappingTestTensor();
      });
  ASSERT_THAT(mapping.Get("model.x"), IsOk());
  ASSERT_THAT(mapping.Get("model.x"), IsOk());
  EXPECT_EQ(num_loads, 1);
  EXPECT_EQ(mapping.ReleaseTensorsNotPinned(), 1);
  ASSERT_THAT(mapping.Get("model.x"), IsOk());
  EXPECT_EQ(num_loads, 2);
}

// Without a loader (a null loader function), nothing is released: the cached
// tensors (here one provided by a hook) couldn't be loaded again.
TEST(LazyTensorMappingTest, ReleaseTensorsNotPinnedIsANoOpWithoutLoader) {
  LazyTensorMapping mapping(
      MappingTestNames(),
      absl::AnyInvocable<absl::StatusOr<TensorHandle>(absl::string_view)>());
  mapping.Register<AliasHooks>();
  ASSERT_THAT(mapping.Set("model.a", MappingTestTensor()), IsOk());
  ASSERT_THAT(mapping.Get("model.alias"), IsOk());
  EXPECT_EQ(mapping.ReleaseTensorsNotPinned(), 0);
  ASSERT_THAT(mapping.Get("model.alias"), IsOk());
}

// A tensor that a hook provided (`OnNotFound`) is released like the loaded
// ones, and provided again by the next `Get()`.
TEST(LazyTensorMappingTest, ReleaseTensorsNotPinnedReloadsHookTensors) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));
  mapping.Register<AliasHooks>();
  ASSERT_THAT(mapping.Get("model.alias"), IsOk());
  // "model.alias" and the "model.a" it was made from.
  EXPECT_EQ(mapping.ReleaseTensorsNotPinned(), 2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle alias,
                                  mapping.Get("model.alias"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, alias.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(),
              ElementsAre(FloatEq(1), FloatEq(2)));
}

TEST(LazyTensorMappingTest, EvictReloadableCacheDropsCheckpointTensorsOnly) {
  SafetensorFileGuard file = CreateMappingTestSafetensor();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LazyTensorMapping mapping,
                                  CreateMappingTestMapping(file));
  std::vector<std::string> log;
  mapping.Register<RecordingHooks>(log, "hooks");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle loaded, mapping.Get("model.a"));
  std::weak_ptr<Buffer> weak_loaded = loaded.GetBufferPtr();
  loaded = TensorHandle::Invalid();
  EXPECT_FALSE(weak_loaded.expired());

  // Override a tensor via `Set` and ensure eviction preserves it.
  TensorHandle pinned = MappingTestTensor();
  std::weak_ptr<Buffer> weak_pinned = pinned.GetBufferPtr();
  ASSERT_THAT(mapping.Set("model.pinned", std::move(pinned)), IsOk());

  mapping.EvictReloadableCache();
  EXPECT_TRUE(weak_loaded.expired());
  EXPECT_FALSE(weak_pinned.expired());

  // Subsequent `Get` transparently reloads the evicted checkpoint tensor.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle reloaded,
                                  mapping.Get("model.a"));
  EXPECT_EQ(reloaded.GetName(), "model.a");
  EXPECT_THAT(log, ElementsAre("hooks.OnLoaded(model.a)",
                               "hooks.OnLoaded(model.a)"));
}
}  // namespace
}  // namespace litert::tensor::examples
