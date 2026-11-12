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

#ifndef ODML_LITERT_LITERT_RUNTIME_MODEL_BUFFER_WRITES_H_
#define ODML_LITERT_LITERT_RUNTIME_MODEL_BUFFER_WRITES_H_

#include <cstddef>

#include "litert/cc/litert_expected.h"

namespace litert::internal {

// Makes [data, data + size) of a model buffer writable in place, without the
// write reaching the model file.
//
// Windows charges a copy-on-write file view its whole size against the
// commit limit when it is mapped, though only the pages written to ever get
// a private copy. KeepModelBufferReadOnlyUntilWritten gives that charge back
// by making the pages of the view that nothing has written to read-only. The
// pages of a range asked for here are then made copy-on-write again and given
// their private copies at once, so only those are charged. A range of a
// read-only file view is made copy-on-write too. A write to a read-only page
// of the view that was not asked for faults instead of being copied.
//
// A copy-on-write view that was not made read-only stays copy-on-write: the
// pages of the range get their private copies and nothing else changes.
//
// Memory that is not a file view, and every platform but Windows, needs
// nothing and returns at once.
Expected<void> MakeModelBufferWritable(const void* data, size_t size);

// Makes the unwritten pages of the copy-on-write file view that holds `data`
// read-only, which gives their charge back. A model is created from a buffer
// that is not changed but through MakeModelBufferWritable. Does nothing for
// memory that is not a copy-on-write view, nothing more for a view this was
// done to already, and nothing on every platform but Windows.
void KeepModelBufferReadOnlyUntilWritten(const void* data);

// While one is alive on a thread, MakeModelBufferWritable on that thread
// remembers the ranges it made writable, or found to need nothing, and returns
// at once for a range inside them. A pass that patches many values of one
// model asks for the same pages again and again (the magic numbers of a Gemma
// 4 E2B: 1,424 ranges in 326 pages), and on Windows every range that is not
// remembered costs a query of its pages, which walks the view from them to the
// next page with other attributes. The buffers written to must stay mapped
// while the scope is alive: what it remembers ends with it, so a view mapped
// later where one of them was is asked about afresh. A scope inside another on
// the same thread adds nothing.
class ScopedModelBufferWrites {
 public:
  ScopedModelBufferWrites();
  ~ScopedModelBufferWrites();
  ScopedModelBufferWrites(const ScopedModelBufferWrites&) = delete;
  ScopedModelBufferWrites& operator=(const ScopedModelBufferWrites&) = delete;

 private:
  void* ranges_ = nullptr;  // This scope's ranges, when it is the outer one.
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_MODEL_BUFFER_WRITES_H_
