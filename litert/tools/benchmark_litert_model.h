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
#ifndef ODML_LITERT_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_
#define ODML_LITERT_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/tools/benchmark/benchmark_model.h"
#include "tflite/tools/benchmark/benchmark_params.h"
#include "tflite/tools/command_line_flags.h"
#include "tflite/tools/utils.h"

namespace litert {
namespace benchmark {

using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Model;
using ::litert::TensorBuffer;
using ::tflite::benchmark::BenchmarkModel;
using ::tflite::benchmark::BenchmarkParam;
using ::tflite::benchmark::BenchmarkParams;
using ::tflite::utils::InputTensorData;

class BenchmarkLiteRtModel : public BenchmarkModel {
 public:
  BenchmarkLiteRtModel() = default;
  explicit BenchmarkLiteRtModel(BenchmarkParams params)
      : BenchmarkModel(std::move(params)) {}
  ~BenchmarkLiteRtModel() override = default;
  static BenchmarkParams DefaultParams() {
    BenchmarkParams default_params = BenchmarkModel::DefaultParams();
    default_params.AddParam("graph", BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("signature_to_run_for",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("use_cpu", BenchmarkParam::Create<bool>(true));
    default_params.AddParam("use_gpu", BenchmarkParam::Create<bool>(false));
    default_params.AddParam("use_npu", BenchmarkParam::Create<bool>(false));
    default_params.AddParam("qnn_dispatch_library_path",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("require_full_delegation",
                            BenchmarkParam::Create<bool>(true));

    return default_params;
  }

  TfLiteStatus Init() override;

  int64_t MayGetModelFileSize() override {
    std::string fd_or_graph_path = params_.Get<std::string>("graph");
    // Path can be one of the following:
    // 1) File descriptor path: path must be in the format of
    // "fd:%model_fd%:%model_offset%:%model_size%".
    // 2) File path: path to the model file.
    // Please see tensorflow/lite/tools/model_loader.h for more information.
    std::vector<absl::string_view> parts =
        absl::StrSplit(fd_or_graph_path, ':');
    if (!parts.empty() && parts[0] == "fd") {
      int64_t model_size = -1;
      if (parts.size() != 4 || !absl::SimpleAtoi(parts[3], &model_size)) {
        LITERT_LOG(LITERT_ERROR, "Failed to parse model file size: %s",
                   fd_or_graph_path.c_str());
      }
      return model_size;
    }
    std::ifstream in_file(fd_or_graph_path, std::ios::binary | std::ios::ate);
    return in_file.tellg();
  }

  TfLiteStatus RunImpl() override {
    if (!compiled_model_) {
      LITERT_LOG(LITERT_ERROR, "Compiled model not initialized");
      return kTfLiteError;
    }
    auto signature = params_.Get<std::string>("signature_to_run_for");
    if (signature.empty()) {
      signature = model_->GetSignature(0)->Key();
    }
    if (compiled_model_->Run(signature, *input_buffers_, *output_buffers_)) {
      return kTfLiteOk;
    } else {
      LITERT_LOG(LITERT_ERROR, "Run failed");
      return kTfLiteError;
    }
  }

  uint64_t ComputeInputBytes() override {
    uint64_t total_bytes = 0;
    for (const auto& buffer : *input_buffers_) {
      total_bytes += *buffer.Size();
    }
    return total_bytes;
  }

  InputTensorData CreateRandomTensorData(const litert::TensorBuffer& t,
                                         std::string name) {
    float low_range = 0;
    float high_range = 0;
    tflite::utils::GetDataRangesForType(
        static_cast<TfLiteType>(t.TensorType()->ElementType()), &low_range,
        &high_range);
    return tflite::utils::CreateRandomTensorData(
        name, static_cast<TfLiteType>(t.TensorType()->ElementType()), *t.Size(),
        low_range, high_range);
  }

  TfLiteStatus PrepareInputData() override {
    int index = 0;
    for (auto& buffer : *input_buffers_) {
      auto t_data =
          CreateRandomTensorData(buffer, "input_" + std::to_string(index));
      buffer.Write<char>(absl::MakeSpan(
          reinterpret_cast<char*>(t_data.data.get()), t_data.bytes));
      ++index;
    }
    return kTfLiteOk;
  }

  TfLiteStatus ResetInputsAndOutputs() override { return kTfLiteOk; }
  std::vector<tflite::Flag> GetFlags() override {
    std::vector<tflite::Flag> flags = BenchmarkModel::GetFlags();
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "graph", &params_, "The path to the model file."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "signature_to_run_for", &params_, "The signature to run for."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_cpu", &params_, "Whether to use CPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_gpu", &params_, "Whether to use GPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_npu", &params_, "Whether to use NPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "qnn_dispatch_library_path", &params_, "QNN dispatch library path."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "require_full_delegation", &params_,
        "Whether to require full delegation."));
    return flags;
  }

 protected:
  virtual TfLiteStatus LoadModel();
  std::unique_ptr<Model> model_;

 private:
  std::unique_ptr<litert::CompiledModel> compiled_model_;
  std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers_;
  std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers_;
};

}  // namespace benchmark
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_
