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

// Focused functional/ownership smoke test; no model or performance measurement.
#include "tensor/examples/gemma4/native/stage_runner.h"
#include <dirent.h>
#include <array>
#include <iostream>
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"

using namespace litert::tensor;
using XnnTensor = Tensor<XnnpackMixinTag>;

static size_t Threads() {
  size_t count = 0;
  DIR* directory = opendir("/proc/self/task");
  if (directory == nullptr) return 0;
  while (dirent* entry = readdir(directory)) {
    if (entry->d_name[0] != '.') ++count;
  }
  closedir(directory);
  return count;
}

#define REQUIRE(condition)                                                   \
  do {                                                                       \
    if (!(condition))                                                        \
      return absl::InternalError("Smoke check failed: " #condition);         \
  } while (false)

struct PoolOwner {
  pthreadpool_t value = pthreadpool_create(2);
  ~PoolOwner() { if (value != nullptr) pthreadpool_destroy(value); }
};
struct CacheOwner {
  xnn_weights_cache_t value = nullptr;
  ~CacheOwner() { if (value != nullptr) xnn_delete_weights_cache(value); }
};

static absl::Status Check() {
  REQUIRE(xnn_initialize(nullptr) == xnn_status_success);
  PoolOwner pool;
  REQUIRE(pool.value != nullptr);
  REQUIRE(pthreadpool_get_threads_count(pool.value) == 2);
  CacheOwner cache;
  REQUIRE(xnn_create_weights_cache(&cache.value) == xnn_status_success);
  const size_t pool_threads = Threads();
  REQUIRE(pool_threads >= 2);

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 3}});
  XnnTensor weights({.name = "weight", .type = Type::kFP32, .shape = {2, 3},
                     .buffer = std::vector<float>{1, 2, 3, 4, 5, 6}});
  XnnTensor bias({.name = "bias", .type = Type::kFP32, .shape = {2},
                  .buffer = std::vector<float>{0.5f, 1.5f}});
  XnnTensor output = FullyConnected(input, weights, bias);
  std::vector<std::unique_ptr<StageRunner>> stages;
  for (size_t i = 0; i < 100; ++i) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto stage, StageRunner::Create(
        {output}, pool.value, cache.value, XNN_FLAG_BASIC_PROFILING));
    REQUIRE(stage->runtime() == nullptr);
    REQUIRE(stage->threadpool() == pool.value);
    REQUIRE(stage->weights_cache() == cache.value);
    LRT_TENSOR_RETURN_IF_ERROR(stage->PrepareRuntime());
    REQUIRE(stage->runtime() != nullptr);
    stages.push_back(std::move(stage));
  }
  REQUIRE(Threads() == pool_threads);
  REQUIRE(xnn_finalize_weights_cache(cache.value,
             xnn_weights_cache_finalization_kind_soft) == xnn_status_success);
  const std::vector<float> data{1, 2, 3};
  for (auto& stage : stages) {
    LRT_TENSOR_RETURN_IF_ERROR(stage->SetInput(input, data));
    LRT_TENSOR_RETURN_IF_ERROR(stage->Run());
    LRT_TENSOR_ASSIGN_OR_RETURN(auto result, stage->ReadOutputAs<float>(output));
    REQUIRE(result.size() == 2 && result.data()[0] == 14.5f &&
            result.data()[1] == 33.5f);
    size_t count = 0, required = 0;
    REQUIRE(xnn_get_runtime_profiling_info(stage->runtime(),
                xnn_profile_info_num_operators, sizeof(count), &count,
                &required) == xnn_status_success);
    REQUIRE(count > 0);
  }
  REQUIRE(Threads() == pool_threads);

  auto& survivor = *stages.back();
  const xnn_runtime_t runtime = survivor.runtime();
  // Destroying 99 stages must leave the pool/cache and final runtime usable.
  stages.erase(stages.begin(), stages.end() - 1);
  REQUIRE(Threads() == pool_threads);
  REQUIRE(xnn_weights_cache_is_finalized(cache.value));
  const std::array<int32_t, 2> shape{2, 3};
  LRT_TENSOR_RETURN_IF_ERROR(survivor.ReshapeInput(input, shape));
  LRT_TENSOR_RETURN_IF_ERROR(survivor.SetInputAsCopy(
      input, std::vector<float>{1, 2, 3, 4, 5, 6}));
  LRT_TENSOR_RETURN_IF_ERROR(survivor.Run());
  REQUIRE(survivor.runtime() == runtime);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto result, survivor.ReadOutputAs<float>(output));
  REQUIRE(result.size() == 4 && result.data()[0] == 14.5f &&
          result.data()[1] == 33.5f && result.data()[2] == 32.5f &&
          result.data()[3] == 78.5f);

  survivor.SetNumThreads(1);
  REQUIRE(absl::IsFailedPrecondition(survivor.Run()));
  survivor.SetNumThreads(2);
  LRT_TENSOR_RETURN_IF_ERROR(survivor.Run());
  stages.clear();
  // An output lock must retain its buffer after its runner is destroyed.
  REQUIRE(result.data()[3] == 78.5f);
  REQUIRE(Threads() == pool_threads);

  // Null pool/cache is a valid serial runner and also owns no shared resource.
  LRT_TENSOR_ASSIGN_OR_RETURN(auto serial,
      StageRunner::Create({output}, nullptr, nullptr));
  LRT_TENSOR_RETURN_IF_ERROR(serial->SetInput(input, data));
  LRT_TENSOR_RETURN_IF_ERROR(serial->Run());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto serial_result,
                              serial->ReadOutputAs<float>(output));
  REQUIRE(serial_result.size() == 2 && serial_result.data()[1] == 33.5f);
  return absl::OkStatus();
}

int main() {
  const absl::Status status = Check();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  std::cout << "PASS: 100 shared-pool/cache stages; FC outputs, flags, reuse, "
               "reshape, copied input, output lifetime, borrowed ownership, "
               "thread-count guard, and null-resource serial execution\n";
  return 0;
}
