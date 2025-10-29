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

#include "tflite/tools/culprit_finder/interpreter_handler.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tflite/c/c_api_types.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter_options.h"
#include "tflite/kernels/register.h"
#include "tflite/tools/culprit_finder/tflite_input_manager.h"
#include "tflite/tools/delegates/delegate_provider.h"
#include "tflite/tools/logging.h"
#include "tflite/tools/model_loader.h"

namespace tflite {
namespace tooling {

std::optional<std::unique_ptr<InterpreterHandler>> InterpreterHandler::Create(
    absl::string_view model_path) {
  std::unique_ptr<ModelLoader> model_loader =
      std::make_unique<::tflite::tools::PathModelLoader>(model_path);
  if (!model_loader) {
    TFLITE_LOG(ERROR) << "Failed to initialize model loader with path "
                      << model_path;
    return std::nullopt;
  }
  if (!model_loader->Init()) {
    TFLITE_LOG(ERROR) << "Failed to load model " << model_path;
    return std::nullopt;
  }

  return std::make_unique<InterpreterHandler>(std::move(model_loader));
}

std::optional<std::unique_ptr<tflite::Interpreter>>
InterpreterHandler::PrepareInterpreter(
    tflite::tools::TfLiteDelegatePtr delegate,
    std::vector<int> intermediate_outputs, bool preserve_all_tensors) {
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  tflite::InterpreterOptions options;
  if (preserve_all_tensors) {
    options.SetPreserveAllTensors(true);
  }

  std::unique_ptr<tflite::Interpreter>* interpreter = nullptr;

  tflite::InterpreterBuilder interpreter_builder(*model_, resolver, &options);
  if (interpreter_builder(interpreter) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to build interpreter";
    return std::nullopt;
  }
  if (interpreter == nullptr) {
    TFLITE_LOG(ERROR) << "Interpreter is null";
    return std::nullopt;
  }

  if (!intermediate_outputs.empty()) {
    auto outputs = (*interpreter)->outputs();
    outputs.insert(outputs.end(), intermediate_outputs.begin(),
                   intermediate_outputs.end());
    (*interpreter)->SetOutputs(outputs);
  }

  if (delegate.get() != nullptr) {
    (*interpreter)->ModifyGraphWithDelegate(delegate.get());
    owned_delegates_.push_back(std::move(delegate));
  }

  if ((*interpreter)->AllocateTensors() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to allocate tensors";
    return std::nullopt;
  }
  return std::move(*interpreter);
}

TfLiteStatus InterpreterHandler::RunInference(
    tflite::Interpreter* interpreter, TfliteInputManager* input_manager) {
  if (interpreter == nullptr) {
    TFLITE_LOG(ERROR) << "Interpreter is null";
    return kTfLiteError;
  }
  interpreter->ResetVariableTensors();
  if (input_manager->SetInputTensors(interpreter) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to set input tensors";
    return kTfLiteError;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to invoke interpreter";
    return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace tooling
}  // namespace tflite
