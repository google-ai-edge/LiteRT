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

#ifndef ODML_LITERT_LITERT_CC_LITERT_DISPATCH_DELEGATE_H_
#define ODML_LITERT_LITERT_CC_LITERT_DISPATCH_DELEGATE_H_

#include "litert/c/litert_common.h"
#include "tflite/delegates/utils/simple_opaque_delegate.h"

namespace litert {

// TODO LUKE

using DispatchDelegatePtr = tflite::TfLiteOpaqueDelegateUniquePtr;

DispatchDelegatePtr CreateDispatchDelegatePtr(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options);

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_DISPATCH_DELEGATE_H_
