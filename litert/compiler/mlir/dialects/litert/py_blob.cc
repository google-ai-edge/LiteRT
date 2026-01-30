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
#include "litert/compiler/mlir/dialects/litert/py_blob.h"

#include <Python.h>

#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {

namespace {

absl::Status GetPythonErrorStatusAndRelease(PyGILState_STATE gstate) {
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  PyObject* py_str = PyObject_Str(value);
  const char* error_msg =
      (py_str) ? PyUnicode_AsUTF8(py_str) : "Unknown Python error";

  auto status =
      absl::UnknownError(absl::StrCat("Python exception: ", error_msg));

  Py_XDECREF(py_str);
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  PyGILState_Release(gstate);
  return status;
}

}  // namespace

absl::Status PyBlob::ApplyData(ApplyDataFuncT callback) const {
  if (!bytes_getter_) {
    return absl::InternalError("PyBlob: bytes_getter is null.");
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  // Call the getter to get the iterator (the generator instance)
  PyObject* iterable = PyObject_CallObject(bytes_getter_, nullptr);
  if (!iterable) {
    return GetPythonErrorStatusAndRelease(gstate);
  }

  PyObject* iter = PyObject_GetIter(iterable);
  Py_DECREF(iterable);

  if (!iter) {
    PyGILState_Release(gstate);
    return absl::InternalError(
        "PyBlob: bytes_getter did not return an iterable.");
  }

  PyObject* chunk;
  absl::Status status = absl::OkStatus();

  // Loop through the chunks
  while ((chunk = PyIter_Next(iter))) {
    if (!PyBytes_Check(chunk)) {
      Py_DECREF(chunk);
      Py_DECREF(iter);
      PyGILState_Release(gstate);
      return absl::InvalidArgumentError(
          "PyBlob: Generator yielded a non-bytes object.");
    }

    // Pass this chunk to the callback
    size_t size = PyBytes_Size(chunk);
    const char* data = PyBytes_AsString(chunk);
    status = callback(absl::string_view(data, size));

    Py_DECREF(chunk);

    // If the callback returns an error, stop iterating
    if (!status.ok()) {
      break;
    }
  }

  Py_DECREF(iter);

  // Check if the loop ended because of a Python exception or just completion
  if (PyErr_Occurred()) {
    status = GetPythonErrorStatusAndRelease(gstate);
  } else {
    PyGILState_Release(gstate);
  }
  return status;
}

}  // namespace litert
