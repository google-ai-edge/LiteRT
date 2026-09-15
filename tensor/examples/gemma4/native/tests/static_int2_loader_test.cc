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

// Validates all actual compact weights and unmapped widened files; no inference.
#include "tensor/examples/gemma4/native/driver_support.h"
#include "tensor/examples/gemma4/native/matched_bundle_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
namespace litert::tensor::examples::gemma4::native {
static absl::Status Check(const std::string& directory) {
  SetPreserveStaticInt2Weights(true);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto loaded, LoadPublishedBundle(directory, Config::E2B()));
  const auto audit = GetStaticInt2WeightAudit(loaded.weights_handle);
  if (audit.tensor_count != 60 || audit.compact_bytes != 283115520 ||
      audit.widened_bytes != 566231040 || audit.lowering_count != 0 ||
      audit.allocation_bytes != audit.compact_bytes + 60 * XNN_EXTRA_BYTES)
    return absl::InternalError("Full INT2 loader accounting mismatch");
  std::ifstream maps_file("/proc/self/maps");
  std::stringstream maps; maps << maps_file.rdbuf();
  for (const auto& [name, tensor] : loaded.weights_handle) {
    if (tensor.GetType() != Type::kI2) continue;
    if (maps.str().find(name + ".i4") != std::string::npos)
      return absl::InternalError("Widened mapping still retained: " + name);
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& compact, GetStaticInt2Weight(tensor));
    if (tensor.GetShape() != compact.provenance.shape ||
        tensor.GetBufferPtr()->ByteSize().value() * 4 !=
            static_cast<size_t>(tensor.GetShape()[0]) * tensor.GetShape()[1])
      return absl::InternalError("Tensor frontend shape/type/byte mismatch");
  }
  std::cout << "PASS: " << audit.tensor_count << " original static INT2 MLP weights; compact_bytes="
            << audit.compact_bytes << "; widened_bytes_replaced=" << audit.widened_bytes
            << "; allocated_including_padding=" << audit.allocation_bytes
            << "; all widened MLP mappings released; no inference or timings\n";
  return absl::OkStatus();
}
}
int main(int argc,char** argv) {
  if(argc != 2) return 2;
  auto status = litert::tensor::examples::gemma4::native::Check(argv[1]);
  if(!status.ok()){std::cerr << status << '\n';return 1;}
}
