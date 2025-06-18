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

#ifndef ODML_LITERT_TOOLS_CULPRIT_FINDER_INTERPRETER_HANDLER_H_
#define ODML_LITERT_TOOLS_CULPRIT_FINDER_INTERPRETER_HANDLER_H_
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/tools/culprit_finder/tflite_input_manager.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/model_builder.h"
#include "tflite/interpreter.h"
#include "tflite/tools/delegates/delegate_provider.h"
#include "tflite/tools/model_loader.h"

namespace litert::tools {

using ::tflite::tools::ModelLoader;
using ::tflite::tools::TfLiteDelegatePtr;

// A class to handle the creation and running of an interpreter.
class InterpreterHandler {
 public:
  explicit InterpreterHandler(std::unique_ptr<ModelLoader> model_loader)
      : model_loader_(std::move(model_loader)) {
    model_ = tflite::FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(
            model_loader_->GetModel()->allocation()->base()),
        model_loader_->GetModel()->allocation()->bytes());
  };
  ~InterpreterHandler() = default;

  // Creates an InterpreterHandler from a model path. If the model path is
  // invalid, returns an Expected object with an error status and no value.
  static litert::Expected<std::unique_ptr<InterpreterHandler>> Create(
      absl::string_view model_path);

  // Prepares an interpreter with the given delegate and intermediate outputs.
  litert::Expected<std::unique_ptr<tflite::Interpreter>> PrepareInterpreter(
      TfLiteDelegatePtr delegate,
      absl::Span<const int> intermediate_outputs = {},
      bool preserve_all_tensors = false);

  // Runs inference on the interpreter with the given input manager.
  TfLiteStatus RunInference(tflite::Interpreter& interpreter,
                            TfliteInputManager& input_manager);

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<ModelLoader> model_loader_;

  std::vector<TfLiteDelegatePtr> owned_delegates_;
};
}  // namespace litert::tools

#endif  // ODML_LITERT_TOOLS_CULPRIT_FINDER_INTERPRETER_HANDLER_H_
