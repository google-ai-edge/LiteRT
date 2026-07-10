// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"

#include <cstdlib>
#include <utility>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/custom_ir_operation_parser.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/support.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"

// The lifecycle of StubDelegate and its interaction with the TFLite runtime:
//
// 1.  User creates a TfLiteDelegate via CreateStubDelegate():
//     - Allocates a TfLiteDelegate object on the heap.
//     - Sets the .Prepare to Prepare(), but does NOT call it yet.
//
// 2.  User calls Interpreter::ModifyGraphWithDelegate():
//     - Eventually calls the .Prepare.
//
// 3.  TFLite runtime calls StubDelegate::Prepare():
//     - Registers a custom TFLite op named "StubDelegate" which has:
//       - Valid .init and .free function.
//       - No-op .prepare and .invoke functions.
//     - Calls TfLiteContext::ReplaceNodeSubsetsWithDelegateKernels(),
//       passing in this new "StubDelegate" registration. This action
//       replaces the nodes specified in ops_to_replace with instances of
//       the "StubDelegate" op.
//
// 4.  TFLite runtime calls every op's .init() incl. "StubDelegate" op's:
//     - Uses IrModelBuilder to construct and return an
//     ::ml_drift::ir::IrModel.
//
// 5.  TFLite runtime calls every op's .free() incl. "StubDelegate" op's:
//     - Deletes the IrModel.

namespace litert::ml_drift::ir {
namespace {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const auto* stub_data = static_cast<const StubDelegateData*>(delegate->data_);
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, stub_data->options, &stub_data->custom_parsers);
  if (!ops_to_replace) return kTfLiteError;
  if (!ops_to_replace->size) {
    TfLiteIntArrayFree(ops_to_replace);
    return kTfLiteOk;
  }
  const TfLiteRegistration kRegistration = {
      .init = [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* delegate_params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        const auto* stub_data = reinterpret_cast<const StubDelegateData*>(
            delegate_params->delegate->data_);
        ::ml_drift::ir::IrModel* model =
            BuildIrModel(*context, *delegate_params, stub_data->options,
                         &stub_data->custom_parsers);
        if (!model) {
          return nullptr;
        }
        const_cast<StubDelegateData*>(stub_data)->ir_model = model;
        return model;
      },
      .free = [](TfLiteContext*, void* buffer) -> void {
        delete static_cast<::ml_drift::ir::IrModel*>(buffer);
      },
      .prepare = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
        return kTfLiteOk;
      },
      .invoke = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
        return kTfLiteOk;
      },
      .profiling_string = nullptr,
      .builtin_code = kTfLiteBuiltinDelegate,
      .custom_name = "StubDelegate",
      .version = 1,
  };
  TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace

TfLiteDelegate* CreateStubDelegate(const IrModelBuilderOptions& options,
                                   CustomIrOpMap custom_parsers) {
  auto* delegate = new TfLiteDelegate();
  auto* stub_data = new StubDelegateData();
  stub_data->options = options;
  stub_data->custom_parsers = std::move(custom_parsers);
  delegate->data_ = stub_data;
  delegate->Prepare = Prepare;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->opaque_delegate_builder = nullptr;
  return delegate;
}

void DeleteStubDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete static_cast<StubDelegateData*>(delegate->data_);
  delete delegate;
}

}  // namespace litert::ml_drift::ir
