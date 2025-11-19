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

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/test/matchers.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::test {
namespace {

TEST(MediaTekSmokeTest, LoadLibsFromEnvPath) {
  constexpr auto kLibNeuronAdapterLib = "libneuron_adapter.so";

  const std::vector<std::string> so_paths = {
      // The following preinstalled library is for system partition
      // applications.
      "libneuronusdk_adapter.mtk.so", "libneuron_adapter_mgvi.so",
      kLibNeuronAdapterLib};

  for (const auto& so_path : so_paths) {
    auto dlib = SharedLibrary::Load(so_path, RtldFlags::Default());
    if (dlib.HasValue()) {
      return;
    }
  }
  ADD_FAILURE() << "Failed to load any of the expected libraries.";
}

TEST(MediaTekSmokeTest, NeuronAdapterApiCreate) {
  auto mediatek_options = mediatek::MediatekOptions::Create();
  auto neuron_adapter_api = litert::mediatek::NeuronAdapterApi::Create(
      std::nullopt, mediatek_options);
  ASSERT_TRUE(neuron_adapter_api.HasValue());
}

}  // namespace
}  // namespace litert::test
