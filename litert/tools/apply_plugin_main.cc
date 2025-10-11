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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expruns or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define INCLUDE_QUALCOMM_COMPILE_FLAGS
#define INCLUDE_MEDIATEK_COMPILE_FLAGS
#define INCLUDE_INTEL_OPENVINO_COMPILE_FLAGS
#define INCLUDE_GOOGLE_TENSOR_COMPILE_FLAGS

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"
#include "litert/tools/apply_plugin.h"
#include "litert/tools/flags/apply_plugin_flags.h"
#include "litert/tools/flags/common_flags.h"
#include "litert/tools/flags/flag_types.h"
#include "litert/tools/flags/vendors/google_tensor_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/intel_openvino_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/mediatek_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/qualcomm_flags.h"  // IWYU pragma: keep
#include "litert/tools/outstream.h"

using ::litert::tools::ApplyPlugin;
using ::litert::tools::ApplyPluginRun;
using ::litert::tools::IntList;
using ::litert::tools::UserStream;

ApplyPluginRun::Ptr ParseFlags() {
  auto res = std::make_unique<ApplyPluginRun>();

  const auto model = absl::GetFlag(FLAGS_model);
  if (!model.empty()) {
    res->model = model;
  }

  const auto soc_manufacturer_absl = absl::GetFlag(FLAGS_soc_manufacturer);
  res->soc_manufacturer = soc_manufacturer_absl;
  const auto soc_model_absl = absl::GetFlag(FLAGS_soc_model);
  res->soc_models.push_back(soc_model_absl);

  const auto libs = absl::GetFlag(FLAGS_libs);
  res->lib_search_paths.assign(libs.begin(), libs.end());

  const auto cmd = absl::GetFlag(FLAGS_cmd);
  if (cmd == "apply") {
    res->cmd = ApplyPluginRun::Cmd::APPLY;
  } else if (cmd == "partition") {
    res->cmd = ApplyPluginRun::Cmd::PARTITION;
  } else if (cmd == "compile") {
    res->cmd = ApplyPluginRun::Cmd::COMPILE;
  } else if (cmd == "info") {
    res->cmd = ApplyPluginRun::Cmd::INFO;
  } else if (cmd == "noop") {
    res->cmd = ApplyPluginRun::Cmd::NOOP;
  } else {
    return nullptr;
  }

  const auto subgraphs = absl::GetFlag(FLAGS_subgraphs);
  for (auto subgraph_idx : subgraphs.elements) {
    res->subgraphs.insert(subgraph_idx);
  }

  return res;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  auto run = ParseFlags();
  if (run == nullptr) {
    return 1;
  }

  run->outs.clear();
  std::vector<std::unique_ptr<litert::tools::UserStream>> oss;
  const auto outs = absl::GetFlag(FLAGS_o);
  for (const auto& out : outs) {
    oss.push_back(std::make_unique<litert::tools::UserStream>(
        UserStream::MakeFromFlag(out)));
    run->outs.push_back(oss.back()->Get());
  }

  run->dump_out = UserStream::MakeFromFlag(absl::GetFlag(FLAGS_err));

  run->dump_out.Get() << absl::StreamFormat(
      "CMD: %s\nMODEL: %s\nSOC_MANUFACTURER: %s\nSOC_MODEL: %s\n",
      absl::GetFlag(FLAGS_cmd), absl::GetFlag(FLAGS_model),
      absl::GetFlag(FLAGS_soc_manufacturer), absl::GetFlag(FLAGS_soc_model));

  auto opts = litert::Options::Create();
  if (!opts) {
    run->dump_out.Get().get() << "Failed to create Litert options\n";
    return 1;
  }

  {
    auto qnn_opts = litert::qualcomm::QualcommOptionsFromFlags();
    if (!qnn_opts) {
      run->dump_out.Get().get() << "Failed to create Qualcomm options\n";
      return 1;
    }

    if (!opts->AddOpaqueOptions(std::move(*qnn_opts))) {
      run->dump_out.Get().get() << "Failed to add Qualcomm options to list\n";
      return 1;
    }
  }

  {
    auto google_tensor_opts =
        litert::google_tensor::GoogleTensorOptionsFromFlags();
    if (!google_tensor_opts) {
      run->dump_out.Get().get() << "Failed to create Google Tensor options\n";
      return 1;
    }

    if (!opts->AddOpaqueOptions(std::move(*google_tensor_opts))) {
      run->dump_out.Get().get()
          << "Failed to add google tensor options to list\n";
      return 1;
    }
  }

  {
    auto intel_openvino_opts =
        litert::intel_openvino::IntelOpenVinoOptionsFromFlags();
    if (!intel_openvino_opts) {
      run->dump_out.Get().get() << "Failed to create Intel OpenVINO options\n";
      return 1;
    }

    if (!opts->AddOpaqueOptions(std::move(*intel_openvino_opts))) {
      run->dump_out.Get().get()
          << "Failed to add Intel OpenVINO options to list\n";
      return 1;
    }
  }

  {
    auto mediatek_opts =
        litert::mediatek::MediatekOptionsFromFlags();
    if (!mediatek_opts) {
      run->dump_out.Get().get() << "Failed to create Mediatek options\n";
      return 1;
    }

    if (!opts->AddOpaqueOptions(std::move(*mediatek_opts))) {
      run->dump_out.Get().get()
          << "Failed to add Mediatek options to list\n";
      return 1;
    }
  }

  run->options = std::move(*opts);

  return ApplyPlugin(std::move(run));
}
