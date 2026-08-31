/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdarg>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/kernels/register.h"
#include "tflite/core/model_builder.h"
#include "tflite/core/tools/verifier.h"

namespace tflite {
namespace testing {
namespace {

// Avoid logging overhead with a no-op error reporter.
class NullErrorReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override { return 0; }
};

// The strict verifier guards against bogus models.
class StrictVerifier : public TfLiteVerifier {
  bool Verify(const char* data, int length, ErrorReporter* reporter) override {
    return ::tflite::Verify(data, static_cast<size_t>(length),
                            AlwaysTrueResolver{}, reporter);
  }
};

std::vector<std::tuple<std::string>> GetSeeds() {
  std::string path =
      tensorflow::GetDataDependencyFilepath("tensorflow/lite/testdata/add.bin");
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  return {std::make_tuple(content)};
}

}  // namespace

// Simple test which fuzzes the actual TFLite flatbuffer model used to create
// the interpreter.
void ModelFuzzerTest(const std::string& data) {
  // Loading while using the model verifier should never crash (though it may
  // return a null model).
  StrictVerifier verifier;
  NullErrorReporter error_reporter;
  auto model = FlatBufferModel::VerifyAndBuildFromBuffer(
      data.data(), data.size(), &verifier, &error_reporter);

  if (model) {
    // If we get a valid model, interpreter creation should never crash (though
    // it may fail with an error code).
    std::unique_ptr<Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder(model->GetModel(), resolver,
                       &error_reporter)(&interpreter);
  }
}
FUZZ_TEST(ModelFuzzer, ModelFuzzerTest).WithSeeds(GetSeeds());

}  // namespace testing
}  // namespace tflite
