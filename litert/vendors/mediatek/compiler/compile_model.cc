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

#include "litert/vendors/mediatek/compiler/compile_model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "neuron/api/NeuronAdapter.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

constexpr int kDecodePartitionIndex = 0;
constexpr int kPrefillPartitionIndex = 1;

namespace {

// Returns the Neuron compiler `--option-bundle` value to use for
// `subgraph_index`, or an empty view if no bundle should be passed.
absl::string_view ResolveOptionBundle(LrtMediatekOptions* mediatek_opts,
                                      int subgraph_index) {
  const char* bundle = nullptr;

  // An explicit `option_bundle` applies to every subgraph, so it short-circuits
  // the per-partition selection below. This is the knob to use for models with
  // more than the two decode/prefill signatures.
  LrtGetMediatekOptionsOptionBundle(mediatek_opts, &bundle);
  if (bundle != nullptr && *bundle != '\0') {
    return bundle;
  }

  if (subgraph_index == kDecodePartitionIndex) {
    LrtGetMediatekOptionsOptionBundleDecode(mediatek_opts, &bundle);
  } else if (subgraph_index == kPrefillPartitionIndex) {
    LrtGetMediatekOptionsOptionBundlePrefill(mediatek_opts, &bundle);
  } else {
    // Any other subgraph is compiled like the decode partition.
    LrtGetMediatekOptionsOptionBundleDecode(mediatek_opts, &bundle);
  }
  return bundle == nullptr ? absl::string_view() : absl::string_view(bundle);
}

}  // namespace

Expected<NeuronCompilationPtr> CompileModel(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model, LrtMediatekOptions* mediatek_opts,
    const int subgraph_index, bool get_supported_mode,
    std::optional<std::string> extract_static_data_path) {
  // LITERT_USE_JIT is automatically defined based on the build target.
  // It is defined on devices with MediaTek hardwares.
#if LITERT_USE_JIT
  if (soc_model) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "JIT compilation for a specific SoC is not supported");
  }
#endif

  // Per MediaTek recommendation, Compilation_create,
  // Compilation_createWithOptions, and Compilation_setOptimizationString
  // should be used as follow:
  // - AOT Compilation: Compilation_createWithOptions only
  // - JIT Compilation: Compilation_create and Compilation_setOptimizationString
  // The code below takes care of those conditions.

  // NOLINTBEGIN
  auto compile_options =
#if LITERT_USE_JIT
      std::string(neuron_adapter_api.JitCompileOptions());
#else
      std::string(neuron_adapter_api.AotCompileOptions());
#endif
  // NOLINTEND

  bool gemma_compiler_optimizations;
  LrtGetMediatekOptionsGemmaCompilerOptimizations(
      mediatek_opts, &gemma_compiler_optimizations);
  if (gemma_compiler_optimizations) {
    const absl::string_view option_bundle =
        ResolveOptionBundle(mediatek_opts, subgraph_index);
    if (!option_bundle.empty()) {
      compile_options = absl::StrCat(" --option-bundle=", option_bundle);
    }
  }

  if (!get_supported_mode && extract_static_data_path.has_value() &&
      !extract_static_data_path->empty()) {
    absl::StrAppend(&compile_options,
                    " --opt-static-sharing --extract-static-data=",
                    *extract_static_data_path);
  }

  // This is needed in order to support FP32 acativations since TFLite doesn't
  // contain support for FP16 activations currently.
  if (auto status = neuron_adapter_api.api().relax_fp32_to_fp16(model, true);
      status != NEURON_NO_ERROR) {
    LITERT_LOG(
        LITERT_INFO,
        "NeuronModel_relaxComputationFloat32toFloat16 failed with error %d",
        status);
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to relaxFp32ToFp16");
  }

  auto compilation =
#if LITERT_USE_JIT
      neuron_adapter_api.CreateCompilation(model);
#else
      neuron_adapter_api.CreateCompilation(model, compile_options);
#endif
  if (!compilation) {
    return compilation.Error();
  }

  if (neuron_adapter_api.api().compilation_set_priority(
          compilation->get(), NEURON_PRIORITY_HIGH) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation priority");
  }

  LiteRtMediatekNeuronAdapterPerformanceMode performance_mode;
  LrtGetMediatekOptionsPerformanceMode(mediatek_opts, &performance_mode);

  LITERT_LOG(LITERT_INFO,
             "NeuronCompilation_setPreference being set with value: %d",
             performance_mode);

  if (neuron_adapter_api.api().compilation_set_preference(
          compilation->get(), performance_mode) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation preference");
  }

  bool l1_cache_optimizations;
  LrtGetMediatekOptionsL1CacheOptimizations(mediatek_opts,
                                            &l1_cache_optimizations);
  if (l1_cache_optimizations) {
    uint32_t apu_mem_size = 0;
    if (neuron_adapter_api.api().get_l1_memory_size_kb(&apu_mem_size) ==
            NEURON_NO_ERROR &&
        apu_mem_size > 0) {
      if (neuron_adapter_api.api().compilation_set_l1_memory_size_kb(
              compilation->get(), apu_mem_size) != NEURON_NO_ERROR) {
        LITERT_LOG(LITERT_INFO,
                   "NeuronCompilation_setL1MemorySizeKb failed with error %d",
                   neuron_adapter_api.api().compilation_set_l1_memory_size_kb(
                       compilation->get(), apu_mem_size));
      }
    }
  }

  LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint;
  LrtGetMediatekOptionsOptimizationHint(mediatek_opts, &optimization_hint);
  if (auto status = neuron_adapter_api.api().compilation_set_optimization_hint(
          compilation->get(), optimization_hint);
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationHint failed with error %d",
               status);
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationHint failed attempting to set "
               "optimization hint enum value to %d",
               optimization_hint);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set optimization hint");
  }

  if (auto status =
          neuron_adapter_api.api().compilation_set_optimization_string(
              compilation->get(), compile_options.c_str());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationString failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set optimization string");
  }

  // NeuronCompilation_getSupportedOperations must be called before
  // calling NeuronCompilation_finish
  if (get_supported_mode) {
    return compilation;
  }

  if (auto status =
          neuron_adapter_api.api().compilation_finish(compilation->get());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO, "NeuronCompilation_finish failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to finish compilation");
  }

  return compilation;
}

Expected<void> GetSupportedOperations(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model, LrtMediatekOptions* mediatek_opts,
    const int subgraph_index, bool* support_flags, int num_ops) {
  auto compilation =
      CompileModel(neuron_adapter_api, model, soc_model, mediatek_opts,
                   subgraph_index, /*get_supported_mode*/ true);
  if (!compilation) {
    return compilation.Error();
  }
  if (auto status =
          neuron_adapter_api.api().compilation_get_supported_opertations(
              compilation->get(), num_ops, support_flags);
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_getSupportedOperations failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get supported op of compilation");
  }
  return {};
}

namespace {

constexpr uint32_t kExtractedWeightsMagic = 0xbbced1ec;

struct ExtractedWeightEntry {
  uint64_t order = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

bool ParseQuotedUint64(absl::string_view text, size_t* pos, uint64_t* out) {
  const size_t open_quote = text.find('"', *pos);
  if (open_quote == absl::string_view::npos) {
    return false;
  }
  const size_t close_quote = text.find('"', open_quote + 1);
  if (close_quote == absl::string_view::npos) {
    return false;
  }
  uint64_t value = 0;
  for (size_t i = open_quote + 1; i < close_quote; ++i) {
    const char c = text[i];
    if (c < '0' || c > '9') {
      return false;
    }
    value = value * 10 + static_cast<uint64_t>(c - '0');
  }
  *out = value;
  *pos = close_quote + 1;
  return true;
}

}  // namespace

Expected<std::vector<std::vector<int8_t>>> ParseExtractedStaticWeights(
    const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return std::vector<std::vector<int8_t>>{};
  }
  const std::streamoff total_size_signed = file.tellg();
  if (total_size_signed < 12) {
    return std::vector<std::vector<int8_t>>{};
  }
  const uint64_t total_size = static_cast<uint64_t>(total_size_signed);
  const uint64_t footer_offset = total_size - 12;

  file.seekg(static_cast<std::streamoff>(footer_offset), std::ios::beg);
  uint8_t footer[12];
  if (!file.read(reinterpret_cast<char*>(footer), sizeof(footer))) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to read extracted weights footer");
  }

  uint64_t json_offset = 0;
  uint32_t magic = 0;
  std::memcpy(&json_offset, footer, sizeof(uint64_t));
  std::memcpy(&magic, footer + sizeof(uint64_t), sizeof(uint32_t));
  if (magic != kExtractedWeightsMagic || json_offset > footer_offset) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid extracted weights footer");
  }

  const uint64_t json_size = footer_offset - json_offset;
  std::string json_text(json_size, '\0');
  file.seekg(static_cast<std::streamoff>(json_offset), std::ios::beg);
  if (json_size > 0 && !file.read(json_text.data(), json_size)) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to read extracted weights metadata JSON");
  }

  std::vector<ExtractedWeightEntry> entries;
  size_t pos = 0;
  while (pos < json_text.size()) {
    if (json_text.find('"', pos) == std::string::npos) {
      break;
    }
    ExtractedWeightEntry entry;
    if (!ParseQuotedUint64(json_text, &pos, &entry.order)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Failed to parse weight order in extracted weights JSON");
    }
    const size_t open_bracket = json_text.find('[', pos);
    if (open_bracket == std::string::npos) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Malformed entry in extracted weights JSON");
    }
    pos = open_bracket + 1;
    if (!ParseQuotedUint64(json_text, &pos, &entry.offset) ||
        !ParseQuotedUint64(json_text, &pos, &entry.size)) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          "Failed to parse weight offset/size in extracted weights JSON");
    }
    const size_t close_bracket = json_text.find(']', pos);
    if (close_bracket == std::string::npos) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Missing closing bracket in extracted weights JSON");
    }
    pos = close_bracket + 1;
    if (entry.offset > json_offset || entry.size > json_offset - entry.offset) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Extracted weight range out of bounds");
    }
    entries.push_back(entry);
  }

  std::sort(entries.begin(), entries.end(),
            [](const ExtractedWeightEntry& a, const ExtractedWeightEntry& b) {
              return a.order < b.order;
            });

  const uint64_t base_order = entries.empty() ? 0 : entries.front().order;
  std::vector<std::vector<int8_t>> weight_buffers;
  weight_buffers.reserve(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].order != base_order + i) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrFormat("Non-contiguous weight order in extracted weights "
                          "JSON: expected %lu, got %lu",
                          base_order + i, entries[i].order));
    }
    std::vector<int8_t> buf(entries[i].size);
    if (entries[i].size > 0) {
      file.seekg(static_cast<std::streamoff>(entries[i].offset), std::ios::beg);
      if (!file.read(reinterpret_cast<char*>(buf.data()), entries[i].size)) {
        return Error(kLiteRtStatusErrorFileIO,
                     "Failed to read extracted weight buffer payload");
      }
    }
    weight_buffers.push_back(std::move(buf));
  }

  return weight_buffers;
}

}  // namespace litert::mediatek
