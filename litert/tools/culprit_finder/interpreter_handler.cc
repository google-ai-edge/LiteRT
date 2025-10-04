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

#include "litert/tools/culprit_finder/interpreter_handler.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/tools/culprit_finder/tflite_input_manager.h"
#include "tflite/c/c_api_types.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter_options.h"
#if !defined(LITERT_NO_BUILTIN_OPS)
#include "tflite/kernels/register.h"
#else
#include "tflite/mutable_op_resolver.h"
#endif  // LITERT_NO_BUILTIN_OPS
#include "tflite/tools/delegates/delegate_provider.h"
#include "tflite/tools/model_loader.h"

namespace litert::tools {

litert::Expected<std::unique_ptr<InterpreterHandler>>
InterpreterHandler::Create(absl::string_view model_path) {
  std::unique_ptr<ModelLoader> model_loader =
      std::make_unique<::tflite::tools::PathModelLoader>(model_path);
  if (!model_loader) {
    LITERT_LOG(LITERT_ERROR, "Failed to initialize model loader with path %s",
               model_path.data());
    return litert::Unexpected(LiteRtStatus::kLiteRtStatusErrorInvalidArgument,
                              "Failed to initialize model loader with path");
  }
  if (!model_loader->Init()) {
    LITERT_LOG(LITERT_ERROR, "Failed to load model %s", model_path.data());
    return litert::Unexpected(LiteRtStatus::kLiteRtStatusErrorRuntimeFailure,
                              "Failed to load model");
  }

  return std::make_unique<InterpreterHandler>(std::move(model_loader));
}

litert::Expected<std::unique_ptr<tflite::Interpreter>>
InterpreterHandler::PrepareInterpreter(
    tflite::tools::TfLiteDelegatePtr delegate,
    absl::Span<const int> intermediate_outputs, bool preserve_all_tensors) {
#if !defined(LITERT_NO_BUILTIN_OPS)
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
#else
  tflite::MutableOpResolver resolver;
#endif

  tflite::InterpreterOptions options;
  if (preserve_all_tensors) {
    options.SetPreserveAllTensors(true);
  }

  std::unique_ptr<tflite::Interpreter> interpreter;

  tflite::InterpreterBuilder interpreter_builder(*model_, resolver, &options);
  if (interpreter_builder(&interpreter) != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to build interpreter");
    return litert::Unexpected(LiteRtStatus::kLiteRtStatusErrorRuntimeFailure,
                              "Failed to build interpreter");
  }
  if (interpreter == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Interpreter is null");
    return litert::Unexpected(LiteRtStatus::kLiteRtStatusErrorRuntimeFailure,
                              "Interpreter is null");
  }

  if (!intermediate_outputs.empty()) {
    std::vector<int> outputs = interpreter->outputs();
    outputs.insert(outputs.end(), intermediate_outputs.begin(),
                   intermediate_outputs.end());
    interpreter->SetOutputs(outputs);
  }

  if (delegate.get() != nullptr) {
    interpreter->ModifyGraphWithDelegate(delegate.get());
    owned_delegates_.push_back(std::move(delegate));
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to allocate tensors");
    return litert::Unexpected(
        LiteRtStatus::kLiteRtStatusErrorMemoryAllocationFailure,
        "Failed to allocate tensors");
  }
  return interpreter;
}

TfLiteStatus InterpreterHandler::RunInference(
    tflite::Interpreter& interpreter, TfliteInputManager& input_manager) {
  interpreter.ResetVariableTensors();
  if (input_manager.SetInputTensors(interpreter) != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to set input tensors");
    return kTfLiteError;
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke interpreter");
    return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace litert::tools
