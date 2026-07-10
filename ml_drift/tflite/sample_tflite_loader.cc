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

#include "third_party/odml/litert/ml_drift/tflite/sample_tflite_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "tflite/schema/schema_generated.h"

namespace litert::ml_drift {

namespace {
absl::StatusOr<size_t> GetFileSize(std::string file_path) {
  std::ifstream in(file_path, std::ifstream::binary);
  if (!in) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open input file: ", file_path));
  }
  in.seekg(0, in.end);
  size_t file_size = in.tellg();
  in.seekg(0, in.beg);
  return file_size;
}
}  // namespace

absl::StatusOr<std::unique_ptr<SampleTfliteLoader>>
SampleTfliteLoader::CreateFromFile(const std::string& tflite_model_path) {
  auto loader = absl::WrapUnique(new SampleTfliteLoader());
  RETURN_IF_ERROR(loader->LoadTfliteModelFile(tflite_model_path));
  return loader;
}

absl::StatusOr<std::unique_ptr<SampleTfliteLoader>>
SampleTfliteLoader::CreateFromString(absl::string_view tflite_model_string) {
  auto loader = absl::WrapUnique(new SampleTfliteLoader());
  RETURN_IF_ERROR(loader->LoadTfliteModelString(tflite_model_string));
  return loader;
}

SampleTfliteLoader::~SampleTfliteLoader() {
  if (!name_to_buffer_.empty()) {
    ABSL_LOG(WARNING) << "Partial data has been loaded.";
    name_to_buffer_.clear();
  }
  if (buffer_ptr_) {
    munmap(buffer_ptr_, buffer_size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

absl::Status SampleTfliteLoader::LoadTfliteModelFile(
    const std::string& tflite_model_path) {
  fd_ = open(tflite_model_path.c_str(), O_RDONLY);
  ASSIGN_OR_RETURN(buffer_size_, GetFileSize(tflite_model_path));
  if (fd_ == -1) {
    return absl::InternalError("Failed to open file, check permissions.");
  }
  buffer_ptr_ = mmap(
      /*addr=*/nullptr, buffer_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (buffer_ptr_ == MAP_FAILED) {
    return absl::InternalError("Buffer mapping failed.");
  }
  const tflite::Model* tflite_model = tflite::GetModel(buffer_ptr_);
  return LoadBufferFromTfliteModel(tflite_model);
}

absl::Status SampleTfliteLoader::LoadTfliteModelString(
    absl::string_view tflite_model_string) {
  const tflite::Model* tflite_model =
      tflite::GetModel(tflite_model_string.data());
  return LoadBufferFromTfliteModel(tflite_model);
}

absl::Status SampleTfliteLoader::LoadBufferFromTfliteModel(
    const tflite::Model* tflite_model) {
  const flatbuffers::Vector<flatbuffers::Offset<::tflite::Buffer>>* buffers =
      (*tflite_model).buffers();
  for (const tflite::SubGraph* subgraph : *tflite_model->subgraphs()) {
    for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
      const std::string tensor_name =
          std::string(tfl_tensor->name()->data(), tfl_tensor->name()->size());
      if (tfl_tensor->type() != tflite::TensorType::TensorType_FLOAT16) {
        return absl::InvalidArgumentError(
            absl::StrCat(tensor_name, " type is not supported."));
      }
      if (tfl_tensor->buffer() >= buffers->size()) {
        return absl::OutOfRangeError(
            absl::StrCat(tensor_name, " buffer is out of range."));
      }
      const tflite::Buffer* tfl_buffer = buffers->Get(tfl_tensor->buffer());
      name_to_buffer_[tensor_name] = tfl_buffer;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Span<const ::ml_drift::half>> SampleTfliteLoader::GetData(
    const std::string& weights_name, int count) {
  return GetTFLiteWeightsInternal(weights_name, count);
}

absl::StatusOr<std::pair<absl::Span<const ::ml_drift::half>,
                         absl::Span<const ::ml_drift::half>>>
SampleTfliteLoader::GetData(const std::string& weights1_name,
                            const std::string& weights2_name, int count1,
                            int count2) {
  absl::Span<const ::ml_drift::half> data1, data2;
  ASSIGN_OR_RETURN(data1, GetTFLiteWeightsInternal(weights1_name, count1));
  ASSIGN_OR_RETURN(data2, GetTFLiteWeightsInternal(weights2_name, count2));

  return std::make_pair(data1, data2);
}

absl::StatusOr<absl::Span<const ::ml_drift::half>>
SampleTfliteLoader::GetTFLiteWeightsInternal(const std::string& weights_name,
                                             int count) {
  auto it = name_to_buffer_.find(weights_name);
  if (it == name_to_buffer_.end()) {
    return absl::NotFoundError(absl::StrCat(weights_name, " not found."));
  }
  const auto* tfl_buffer = it->second;
  const uint8_t* tfl_buffer_ptr =
      reinterpret_cast<const uint8_t*>(tfl_buffer->data()->data());
  if (tfl_buffer->size() / 2 != count) {
    return absl::InvalidArgumentError(
        absl::StrCat("model data: ", weights_name, " has wrong size."));
  }
  absl::Span<const ::ml_drift::half> ret = absl::MakeSpan(
      reinterpret_cast<const ::ml_drift::half*>(tfl_buffer_ptr), count);
  name_to_buffer_.erase(it);
  return ret;
}

}  // namespace litert::ml_drift
