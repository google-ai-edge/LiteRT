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
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/tools/benchmark_litert_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/model_builder.h"
#include "tflite/tools/benchmark/benchmark_params.h"
#include "tflite/tools/logging.h"
#include "tflite/tools/strip_buffers/stripping_lib.h"

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

  Expected<Model> model_result;
  if (tflite::FlatbufferHasStrippedWeights(input_model->GetModel())) {
    // Reconstitute flatbuffer with appropriate random constant tensors
    TF_LITE_ENSURE_STATUS(tflite::ReconstituteConstantTensorsIntoFlatbuffer(
        input_model->GetModel(), &reconstituted_model_builder_));
    TFLITE_LOG(INFO) << "Finished reconstitution for model " << graph;
    litert::BufferRef<uint8_t> buffer(
        reinterpret_cast<const char*>(
            reconstituted_model_builder_.GetBufferPointer()),
        reconstituted_model_builder_.GetSize());
    model_result = litert::Model::CreateFromBuffer(buffer);
  } else {
    TFLITE_LOG(INFO) << "Original model already has weights " << graph;
    model_result = litert::Model::CreateFromFile(graph);
  }
  if (!model_result) {
    TFLITE_LOG(ERROR) << "Failed to build model from buffer for model " << graph
                      << " (" << model_result.Error().Message() << ")";
    return kTfLiteError;
  }
  model_ = std::make_unique<litert::Model>(std::move(*model_result));

  TFLITE_LOG(INFO) << "Loaded model for " << graph;
  return kTfLiteOk;
}

}  // namespace benchmark
}  // namespace litert
