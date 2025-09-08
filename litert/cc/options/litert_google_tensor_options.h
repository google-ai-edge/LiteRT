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

#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::google_tensor {

// Wraps a LiteRtGoogleTensorOptions object for convenience.
class GoogleTensorOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  GoogleTensorOptions() = delete;

  static const char* Discriminator() {
    return LiteRtGoogleTensorOptionsGetIdentifier();
  }

  static Expected<GoogleTensorOptions> Create(OpaqueOptions& options);
  static Expected<GoogleTensorOptions> Create();

  void SetFloatTruncationType(
      LiteRtGoogleTensorOptionsTruncationType truncation_type);

  LiteRtGoogleTensorOptionsTruncationType GetFloatTruncationType() const;

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
      LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity);

  LiteRtGoogleTensorOptionsShardingIntensity GetShardingIntensity() const;

  std::vector<std::vector<std::string>> GetTestingFlags() const;

  void SetTestingFlags(const std::string& testing_flags);

 private:
  LiteRtGoogleTensorOptions Data() const;
};

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
