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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_BENCHMARK_STRIPPED_LITERT_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_BENCHMARK_STRIPPED_LITERT_MODEL_H_

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/tools/benchmark_litert_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/tools/benchmark/benchmark_params.h"

namespace litert {
namespace benchmark {
class InternalBenchmarkStrippedLitertModel : public BenchmarkLiteRtModel {
 public:
  InternalBenchmarkStrippedLitertModel();
  explicit InternalBenchmarkStrippedLitertModel(
      tflite::benchmark::BenchmarkParams params);

  TfLiteStatus LoadModel() override;

 private:
  flatbuffers::FlatBufferBuilder reconstituted_model_builder_;
};
}  // namespace benchmark
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_BENCHMARK_STRIPPED_LITERT_MODEL_H_
