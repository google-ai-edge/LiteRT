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

#include <filesystem>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/examples/experimental/gemma4/bundle_loader.h"
#include "tensor/examples/experimental/gemma4/model_builder.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/utils/macros.h"

ABSL_FLAG(std::string, bundle_dir, "",
          "Verified published Gemma4 E2B weight bundle.");
ABSL_FLAG(std::string, model_path, "", "New output .tflite model path.");
ABSL_FLAG(int, prefill_chunk_rows, 128,
          "Prefill row limit, a power of two in [32,1024].");
ABSL_FLAG(bool, experimental_compact_int2, false,
          "Read original compact INT2 storage.");
ABSL_FLAG(bool, experimental_dynamic_prefill, false,
          "Allow variable prefill query rows.");
ABSL_FLAG(bool, experimental_explicit_dynamic_shapes, false,
          "Declare dynamic KV extents and optional prefill rows for strict "
          "resizing.");

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {
absl::Status Export() {
  const std::string path = absl::GetFlag(FLAGS_model_path);
  const std::string directory = absl::GetFlag(FLAGS_bundle_dir);
  if (directory.empty() || path.empty() || std::filesystem::exists(path)) {
    return absl::InvalidArgumentError(
        "Require bundle_dir and a new model_path");
  }
  const auto config = examples::gemma4::Config::E2B();
  const bool compact = absl::GetFlag(FLAGS_experimental_compact_int2);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto weights, LoadPublishedBundle(directory, config, false, compact));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto specs,
                              LoadPublishedKvSpecs(directory, config, compact));
  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(AddGemma4Signatures(
      factory, weights, specs,
      {.prefill_rows = absl::GetFlag(FLAGS_prefill_chunk_rows),
       .dynamic_prefill_rows =
           absl::GetFlag(FLAGS_experimental_dynamic_prefill),
       .explicit_dynamic_shapes =
           absl::GetFlag(FLAGS_experimental_explicit_dynamic_shapes)}));
  return factory.Save(path);
}
}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const auto status =
      litert::tensor::examples::experimental::gemma4::cpu::Export();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
