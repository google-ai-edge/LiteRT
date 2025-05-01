// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPAQUE_OPTIONS_H_
#define ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPAQUE_OPTIONS_H_

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::internal {

class DispatchDelegateOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() { return "dispatch_delegate"; }

  // Get a non-owning view of the given opaque options if they are of the
  // correct derived type.
  static Expected<DispatchDelegateOptions> Create(OpaqueOptions& options);

  // Create a new owning view.
  static Expected<DispatchDelegateOptions> Create();

  // alloc_base ----------------------------------------------------------------

  // Alloc base is the address of the first byte of the flatbuffer model being
  // executed. This is relevant to backends with a compiled asset stored at the
  // back of the fb.

  // Set alloc base as a raw pointer.
  Expected<void> SetAllocBase(const void* alloc_base);

  // Get alloc base as a raw pointer.
  Expected<const void*> GetAllocBase();

  // alloc_base_fd -------------------------------------------------------------

  // Alloc base fd is simiilar to alloc base but it is a file descriptor to
  // assets stored externally.

  // Set alloc base fd.
  Expected<void> SetAllocBaseFd(int alloc_base_fd);

  // Get alloc base fd.
  Expected<int> GetAllocBaseFd();
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPAQUE_OPTIONS_H_
