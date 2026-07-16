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

#ifndef ODML_LITERT_LITERT_TFLITE_SUPPORT_CUSTOM_OP_CUSTOM_OP_H_
#define ODML_LITERT_LITERT_TFLITE_SUPPORT_CUSTOM_OP_CUSTOM_OP_H_

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"

struct TfLiteRegistration;
struct TfLiteOperator;

namespace litert {
namespace tflite_support {

/// Adds a custom op to the LiteRt::Options from a `TfLiteRegistration`.
///
/// Example usage:
/// @code
/// TfLiteRegistration* reg = Register_MY_CUSTOM_OP();
/// reg->custom_name = "MyCustomOp";
/// reg->version = 1;
/// litert::tflite_support::AddCustomOp(options, reg);
/// @endcode
///
/// Note: This API is experimental and is designed to support legacy TFLite
/// custom ops. It is subject to change or removal at any time without notice.
/// This API is not ABI stable and should only be used in static runtime builds.
Expected<void> AddCustomOp(Options& options, const TfLiteRegistration* reg);

/// Adds a custom op to the LiteRt::Options from a `TfLiteOperator`.
///
/// Example usage:
/// @code
/// TfLiteOperator* op = TfLiteOperatorCreate(kTfLiteBuiltinCustom,
///                                           "MyCustomOp",
///                                           /*version=*/1, nullptr);
/// TfLiteOperatorSetPrepare(op, MyPrepare);
/// TfLiteOperatorSetInvoke(op, MyInvoke);
/// litert::tflite_support::AddCustomOp(options, op);
/// @endcode
///
/// Note: This API is experimental and is designed to support legacy TFLite
/// custom ops. It is subject to change or removal at any time without notice.
/// This API is not ABI stable and should only be used in static runtime builds.
Expected<void> AddCustomOp(Options& options, const TfLiteOperator* op);

}  // namespace tflite_support
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TFLITE_SUPPORT_CUSTOM_OP_CUSTOM_OP_H_
