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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/litert_expected.h"

namespace litert::google_tensor {

class GoogleTensorOptions {
 public:
  GoogleTensorOptions() = delete;

  static const char* Discriminator() { return "google_tensor"; }

  static Expected<GoogleTensorOptions> Create();

  LrtGoogleTensorOptions Get() const { return options_.get(); }

  LiteRtStatus GetOpaqueOptionsData(const char** identifier, void** payload,
                                    void (**payload_deleter)(void*)) const;

  void SetFloatTruncationType(
      LrtGoogleTensorOptionsTruncationType truncation_type);

  LrtGoogleTensorOptionsTruncationType GetFloatTruncationType() const;

  void SetInt64ToInt32Truncation(bool int64_to_int32_truncation);

  bool GetInt64ToInt32Truncation() const;

  void SetOutputDir(absl::string_view output_dir);

  absl::string_view GetOutputDir() const;

  void SetDumpOpTimings(bool dump_op_timings);

  bool GetDumpOpTimings() const;

  bool GetEnableLargeModelSupport() const;

  void SetEnableLargeModelSupport(bool enable_large_model_support);

  bool GetEnable4BitCompilation() const;

  void SetEnable4BitCompilation(bool enable_4bit_compilation);

  void SetShardingIntensity(
      LrtGoogleTensorOptionsShardingIntensity sharding_intensity);

  LrtGoogleTensorOptionsShardingIntensity GetShardingIntensity() const;

  bool GetEnableDynamicRangeQuantization() const;

  void SetEnableDynamicRangeQuantization(
      bool enable_dynamic_range_quantization);

  std::vector<std::vector<std::string>> GetTestingFlags() const;

  void SetTestingFlags(const std::string& testing_flags);

 private:
  explicit GoogleTensorOptions(LrtGoogleTensorOptions options);

  struct Deleter {
    void operator()(LrtGoogleTensorOptions options) const {
      LrtDestroyGoogleTensorOptions(options);
    }
  };
  std::unique_ptr<LrtGoogleTensorOptionsT, Deleter> options_;
};

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
