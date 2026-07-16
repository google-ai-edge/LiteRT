// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SAMPLE_TFLITE_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SAMPLE_TFLITE_LOADER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/samples/stable_diffusion/model_data_loader.h"  // from @ml_drift
#include "tflite/schema/schema_generated.h"

namespace litert::ml_drift {

class SampleTfliteLoader : public ::ml_drift::ModelDataLoader {
 public:
  static absl::StatusOr<std::unique_ptr<SampleTfliteLoader>> CreateFromFile(
      const std::string& tflite_model_path);

  static absl::StatusOr<std::unique_ptr<SampleTfliteLoader>> CreateFromString(
      absl::string_view tflite_model_string);

  ~SampleTfliteLoader();

  absl::StatusOr<absl::Span<const ::ml_drift::half>> GetData(
      const std::string& weights_name, int count) override;

  absl::StatusOr<std::pair<absl::Span<const ::ml_drift::half>,
                           absl::Span<const ::ml_drift::half>>>
  GetData(const std::string& weights1_name, const std::string& weights2_name,
          int count1, int count2) override;

 private:
  SampleTfliteLoader() {}

  absl::Status LoadTfliteModelFile(const std::string& tflite_model_path);
  absl::Status LoadTfliteModelString(absl::string_view tflite_model_string);
  absl::Status LoadBufferFromTfliteModel(const tflite::Model* tflite_model);

  absl::StatusOr<absl::Span<const ::ml_drift::half>> GetTFLiteWeightsInternal(
      const std::string& weights_name, int count);

  int fd_ = -1;
  void* buffer_ptr_ = nullptr;
  int64_t buffer_size_ = 0;
  absl::flat_hash_map<std::string, const tflite::Buffer*> name_to_buffer_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SAMPLE_TFLITE_LOADER_H_
