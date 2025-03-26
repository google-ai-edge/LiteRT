/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "litert/tools/benchmark_stripped_litert_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_model.h"
#include "litert/tools/benchmark_litert_model.h"
#include "tensorflow/lite/c/c_api_types.h"  // from @org_tensorflow
#include "tensorflow/lite/c/common.h"  // from @org_tensorflow
#include "tensorflow/lite/model_builder.h"  // from @org_tensorflow
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"  // from @org_tensorflow
#include "tensorflow/lite/tools/logging.h"  // from @org_tensorflow
#include "tensorflow/lite/tools/strip_buffers/stripping_lib.h"  // from @org_tensorflow

namespace litert {
namespace benchmark {

InternalBenchmarkStrippedLitertModel::InternalBenchmarkStrippedLitertModel()
    : BenchmarkLiteRtModel(DefaultParams()) {}

InternalBenchmarkStrippedLitertModel::InternalBenchmarkStrippedLitertModel(
    BenchmarkParams params)
    : BenchmarkLiteRtModel(std::move(params)) {}

TfLiteStatus InternalBenchmarkStrippedLitertModel::LoadModel() {
  std::string graph = params_.Get<std::string>("graph");
  TFLITE_LOG(INFO) << "Loading model from: " << graph;
  auto input_model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!input_model) {
    TFLITE_LOG(ERROR) << "Failed to mmap model " << graph;
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Loaded original model " << graph;

  if (tflite::FlatbufferHasStrippedWeights(input_model->GetModel())) {
    // Reconstitute flatbuffer with appropriate random constant tensors
    TF_LITE_ENSURE_STATUS(tflite::ReconstituteConstantTensorsIntoFlatbuffer(
        input_model->GetModel(), &reconstituted_model_builder_));
    TFLITE_LOG(INFO) << "Finished reconstitution for model " << graph;
    litert::BufferRef<uint8_t> buffer(
        reinterpret_cast<const char*>(
            reconstituted_model_builder_.GetBufferPointer()),
        reconstituted_model_builder_.GetSize());
    auto model_result = litert::Model::CreateFromBuffer(buffer);
    model_ = std::make_unique<litert::Model>(std::move(*model_result));
  } else {
    TFLITE_LOG(INFO) << "Original model already has weights " << graph;
    auto model_result = litert::Model::CreateFromFile(graph);
    model_ = std::make_unique<litert::Model>(std::move(*model_result));
  }

  if (!model_) {
    TFLITE_LOG(ERROR) << "Failed to build model from buffer for model "
                      << graph;
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Loaded model for " << graph;
  return kTfLiteOk;
}

}  // namespace benchmark
}  // namespace litert
