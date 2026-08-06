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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_RUNNER_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_RUNNER_OPTIONS_H_

#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_options.h"

namespace litert::tensor {

class RunnerOptions {
 public:
  static absl::StatusOr<RunnerOptions> Create() {
    auto options_or = litert::Options::Create();
    if (!options_or.HasValue()) {
      return absl::InternalError(options_or.Error().Message());
    }
    return RunnerOptions(std::move(*options_or));
  }

  RunnerOptions(RunnerOptions&&) = default;
  RunnerOptions& operator=(RunnerOptions&&) = default;
  RunnerOptions(const RunnerOptions&) = delete;
  RunnerOptions& operator=(const RunnerOptions&) = delete;

  enum class Accelerator {
    kCpu,
    kGpu,
  };

  absl::Status SetAccelerator(Accelerator accelerator) {
    litert::HwAcceleratorSet hw_set(accelerator == Accelerator::kGpu
                                        ? litert::HwAccelerators::kGpu
                                        : litert::HwAccelerators::kCpu);
    auto status = options_.SetHardwareAccelerators(hw_set);
    if (!status.HasValue()) {
      return absl::InternalError(status.Error().Message());
    }
    return absl::OkStatus();
  }

  absl::Status SetCacheDirectory(const std::string& cache_dir,
                                 const std::string& model_key) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    if (ec) {
      return absl::InternalError(
          absl::StrCat("Failed to create cache directory: ", ec.message()));
    }

    fs::path program_cache_path =
        fs::path(cache_dir) / absl::StrCat(model_key, ".program.bin");
    fs::path weight_cache_path =
        fs::path(cache_dir) / absl::StrCat(model_key, ".weights.bin");

    if (!fs::exists(program_cache_path)) {
      std::ofstream(program_cache_path).close();
    }
    if (!fs::exists(weight_cache_path)) {
      std::ofstream(weight_cache_path).close();
    }

    auto program_file_or =
        litert::ScopedFile::OpenWritable(program_cache_path.string());
    if (!program_file_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to open program cache file: ",
                       program_file_or.status().message()));
    }
    auto weight_file_or =
        litert::ScopedFile::OpenWritable(weight_cache_path.string());
    if (!weight_file_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to open weight cache file: ",
                       weight_file_or.status().message()));
    }

    auto gpu_options_or = options_.GetGpuOptions();
    if (!gpu_options_or.HasValue()) {
      return absl::InternalError(gpu_options_or.Error().Message());
    }

    auto status = gpu_options_or->SetProgramCacheFd(program_file_or->file());
    if (status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to set program cache FD");
    }

    status = gpu_options_or->SetWeightCacheFd(weight_file_or->file());
    if (status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to set weight cache FD");
    }

    status = gpu_options_or->SetSerializeProgramCache(true);
    if (status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to set serialize program cache");
    }

    status = gpu_options_or->SetSerializeExternalTensors(true);
    if (status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to set serialize external tensors");
    }

    status = gpu_options_or->SetModelCacheKey(model_key.c_str());
    if (status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to set model cache key");
    }

    owned_files_.push_back(std::move(*program_file_or));
    owned_files_.push_back(std::move(*weight_file_or));

    return absl::OkStatus();
  }

  // Expose underlying options for internal runner usage.
  litert::Options& litert_options() { return options_; }
  const litert::Options& litert_options() const { return options_; }

 private:
  explicit RunnerOptions(litert::Options options)
      : options_(std::move(options)) {}

  litert::Options options_;
  std::vector<litert::ScopedFile> owned_files_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_RUNNER_OPTIONS_H_
