// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_INTERPRETER_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_INTERPRETER_BUILDER_H_

#include <memory>

#include "litert/core/model/model.h"
#include "tflite/converter/allocation.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_options.h"

namespace litert::internal {

// Builds a `tflite::Interpreter` directly from a `LiteRtModelT` IR graph
// without re-parsing or re-verifying the raw FlatBuffer model.
TfLiteStatus BuildInterpreterFromLiteRtModel(
    LiteRtModelT& model, const tflite::OpResolver& op_resolver,
    tflite::ErrorReporter* error_reporter,
    const tflite::InterpreterOptions& options,
    const tflite::Allocation* allocation, int num_threads,
    std::unique_ptr<tflite::Interpreter>* interpreter_out);

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_INTERPRETER_BUILDER_H_
