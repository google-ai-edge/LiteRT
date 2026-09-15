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

// Shared scratch correctness/ownership test. No model or performance timings.
#include "tensor/examples/gemma4/native/stage_runner.h"

#include <array>
#include <cstring>
#include <iostream>
#include <vector>

#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"

using namespace litert::tensor;
using XnnTensor = Tensor<XnnpackMixinTag>;

namespace {
size_t checks = 0;
size_t compared_floats = 0;

#define REQUIRE(condition)                                                    \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(condition))                                                         \
      return absl::InternalError("Workspace check failed: " #condition);      \
  } while (false)

struct WorkspaceOwner {
  xnn_workspace_t value = nullptr;
  ~WorkspaceOwner() { if (value != nullptr) xnn_release_workspace(value); }
};

bool Same(absl::Span<const float> actual, absl::Span<const float> expected) {
  compared_floats += actual.size();
  return actual.size() == expected.size() &&
      std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
}

absl::Status SetRows(StageRunner& stage, const XnnTensor& input, int32_t rows,
                     absl::Span<const float> data) {
  const std::array<int32_t, 2> shape{rows, 64};
  LRT_TENSOR_RETURN_IF_ERROR(stage.ReshapeInput(input, shape));
  return stage.SetInput(input, data);
}

absl::Status RunPair(StageRunner& shared, StageRunner& isolated,
                     const XnnTensor& input, const XnnTensor& output,
                     int32_t rows, absl::Span<const float> data) {
  LRT_TENSOR_RETURN_IF_ERROR(SetRows(shared, input, rows, data));
  LRT_TENSOR_RETURN_IF_ERROR(SetRows(isolated, input, rows, data));
  LRT_TENSOR_RETURN_IF_ERROR(shared.Run());
  LRT_TENSOR_RETURN_IF_ERROR(isolated.Run());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto got, shared.ReadOutputAs<float>(output));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto want, isolated.ReadOutputAs<float>(output));
  // ReadOutput returns the retained allocation, which can exceed the live shape
  // after shrinking. Compare only the logical output, as the model driver does.
  REQUIRE(Same({got.data(), size_t(rows) * 64},
               {want.data(), size_t(rows) * 64}));
  return absl::OkStatus();
}

std::vector<float> Input(size_t rows) {
  std::vector<float> data(rows * 64);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = float(int(i % 31) - 15) / 32.0f;
  }
  return data;
}

absl::Status Check() {
  REQUIRE(xnn_initialize(nullptr) == xnn_status_success);
  WorkspaceOwner workspace;
  REQUIRE(xnn_create_workspace(&workspace.value) == xnn_status_success);
  REQUIRE(workspace.value->ref_count == 1);

  XnnTensor in_a({.name = "producer", .type = Type::kFP32, .shape = {1, 64}});
  XnnTensor out_a = Add(Square(Add(in_a, 0.5f)), in_a);
  XnnTensor in_b({.name = "consumer", .type = Type::kFP32, .shape = {1, 64}});
  XnnTensor out_b = Square(Add(Square(in_b), 0.25f));
  XnnTensor in_c({.name = "grower", .type = Type::kFP32, .shape = {1, 64}});
  XnnTensor out_c = Add(Square(Add(in_c, 0.125f)), Square(in_c));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto a, StageRunner::Create(
      {out_a}, nullptr, nullptr, 0, workspace.value));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto b, StageRunner::Create(
      {out_b}, nullptr, nullptr, 0, workspace.value));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto c, StageRunner::Create(
      {out_c}, nullptr, nullptr, 0, workspace.value));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto private_a,
                              StageRunner::Create({out_a}, nullptr, nullptr));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto private_b,
                              StageRunner::Create({out_b}, nullptr, nullptr));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto private_c,
                              StageRunner::Create({out_c}, nullptr, nullptr));
  REQUIRE(a->workspace() == workspace.value && a->workspace_bytes() == 0);
  REQUIRE(private_a->workspace() == nullptr);
  LRT_TENSOR_RETURN_IF_ERROR(a->PrepareRuntime());
  LRT_TENSOR_RETURN_IF_ERROR(b->PrepareRuntime());
  LRT_TENSOR_RETURN_IF_ERROR(c->PrepareRuntime());
  REQUIRE(workspace.value->ref_count == 4);
  REQUIRE(a->runtime()->workspace == b->runtime()->workspace &&
          b->runtime()->workspace == c->runtime()->workspace);

  const auto initial = Input(1);
  LRT_TENSOR_RETURN_IF_ERROR(RunPair(*a, *private_a, in_a, out_a, 1, initial));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto retained_a, a->ReadOutputAs<float>(out_a));
  const std::vector<float> saved_a(retained_a.data(), retained_a.data() + 64);
  LRT_TENSOR_RETURN_IF_ERROR(RunPair(*b, *private_b, in_b, out_b, 1,
                                   {retained_a.data(), 64}));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto retained_b, b->ReadOutputAs<float>(out_b));
  const std::vector<float> saved_b(retained_b.data(), retained_b.data() + 64);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto memory_a, a->MemorySnapshot());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto memory_b, b->MemorySnapshot());
  REQUIRE(memory_a.workspace == workspace.value);
  REQUIRE(memory_a.workspace_bytes > 0);
  REQUIRE(memory_a.owned_external_bytes == 64 * sizeof(float));
  REQUIRE(memory_a.owned_output_bytes == 64 * sizeof(float));
  REQUIRE(memory_a.viewed_external_bytes == 64 * sizeof(float));
  REQUIRE(memory_b.owned_output_bytes == 64 * sizeof(float));
  const uintptr_t arena_begin = reinterpret_cast<uintptr_t>(workspace.value->data);
  for (const auto& buffer : memory_a.external_buffers) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(buffer.data);
    REQUIRE(address < arena_begin || address >= arena_begin + workspace.value->size);
  }
  REQUIRE(private_a->workspace() != workspace.value);
  REQUIRE(private_a->workspace() != private_b->workspace());
  const size_t small_bytes = workspace.value->size;
  const void* small_data = workspace.value->data;
  const auto grower = Input(4096);
  LRT_TENSOR_RETURN_IF_ERROR(RunPair(*c, *private_c, in_c, out_c, 4096, grower));
  REQUIRE(workspace.value->size > small_bytes);
  REQUIRE(workspace.value->data != small_data);
  REQUIRE(Same({retained_a.data(), 64}, saved_a));
  REQUIRE(Same({retained_b.data(), 64}, saved_b));

  // A prior user's operators must remain invocable immediately after another
  // stage relocates the arena, even without a fresh reshape/setup of that user.
  REQUIRE(xnn_invoke_runtime(a->runtime()) == xnn_status_success);
  REQUIRE(Same({retained_a.data(), 64}, saved_a));
  REQUIRE(xnn_invoke_runtime(b->runtime()) == xnn_status_success);
  REQUIRE(Same({retained_b.data(), 64}, saved_b));

  size_t arena_high_water = workspace.value->size;
  const std::array<int32_t, 9> schedule{1, 17, 4, 511, 3, 2048, 2, 8192, 1};
  for (int32_t rows : schedule) {
    const auto data = Input(rows);
    LRT_TENSOR_RETURN_IF_ERROR(RunPair(*a, *private_a, in_a, out_a, rows, data));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto result_a, a->ReadOutputAs<float>(out_a));
    const std::vector<float> before(result_a.data(), result_a.data() + rows * 64);
    LRT_TENSOR_RETURN_IF_ERROR(RunPair(*b, *private_b, in_b, out_b, rows,
                                     {result_a.data(), size_t(rows) * 64}));
    // Exercise a third graph between a producer/consumer pair without copying
    // the producer output or allowing its storage to alias the shared scratch.
    const int32_t c_rows = rows == 8192 ? 16384 : 1;
    const auto data_c = Input(c_rows);
    LRT_TENSOR_RETURN_IF_ERROR(RunPair(*c, *private_c, in_c, out_c, c_rows, data_c));
    REQUIRE(Same({result_a.data(), size_t(rows) * 64}, before));
    REQUIRE(xnn_invoke_runtime(b->runtime()) == xnn_status_success);
    LRT_TENSOR_ASSIGN_OR_RETURN(auto result_b, b->ReadOutputAs<float>(out_b));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto expected_b, private_b->ReadOutputAs<float>(out_b));
    REQUIRE(Same({result_b.data(), size_t(rows) * 64},
                 {expected_b.data(), size_t(rows) * 64}));
    REQUIRE(workspace.value->size >= arena_high_water);
    arena_high_water = workspace.value->size;
  }
  // Growing a producer replaces its owning output buffer. Locks of the earlier
  // allocation must remain live, including after all runtimes are destroyed.
  REQUIRE(Same({retained_a.data(), 64}, saved_a));
  REQUIRE(Same({retained_b.data(), 64}, saved_b));
  c.reset();
  REQUIRE(workspace.value->ref_count == 3);
  LRT_TENSOR_RETURN_IF_ERROR(b->Run());
  a.reset();
  REQUIRE(workspace.value->ref_count == 2);
  b.reset();
  REQUIRE(workspace.value->ref_count == 1);
  REQUIRE(Same({retained_a.data(), 64}, saved_a));
  REQUIRE(Same({retained_b.data(), 64}, saved_b));

  // Sharing scratch never turns a caller-provided external output into owning
  // memory; it survives destruction under the caller's existing lifetime rule.
  LRT_TENSOR_ASSIGN_OR_RETURN(auto external, StageRunner::Create(
      {out_a}, nullptr, nullptr, 0, workspace.value));
  auto provided = OwningCpuBuffer::Allocate<Type::kFP32>(64);
  LRT_TENSOR_RETURN_IF_ERROR(external->SetOutput(out_a,
      {provided->data(), 64 * sizeof(float)}));
  LRT_TENSOR_RETURN_IF_ERROR(external->SetInput(in_a, initial));
  LRT_TENSOR_RETURN_IF_ERROR(external->Run());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto external_memory, external->MemorySnapshot());
  REQUIRE(external_memory.owned_output_bytes == 0);
  REQUIRE(external_memory.owned_external_bytes == 0);
  REQUIRE(external_memory.viewed_external_bytes == 128 * sizeof(float));
  external.reset();
  REQUIRE(Same({reinterpret_cast<const float*>(provided->data()), 64}, saved_a));
  REQUIRE(workspace.value->ref_count == 1);
  return absl::OkStatus();
}
}  // namespace

int main() {
  const auto result = Check();
  if (!result.ok()) {
    std::cerr << result << '\n';
    return 1;
  }
  std::cout << "PASS: " << checks << " checks, " << compared_floats
            << " floats compared bitwise; shared/private parity, borrowed outputs, "
               "grow/shrink/reallocation, direct reinvocation after relocation, "
               "workspace reference ownership, retained output locks, caller "
               "outputs, and allocation accounting\n";
}
