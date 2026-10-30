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

#include "litert/tflite_support/op_resolver/selected_ops_registry.h"

#include <cstddef>

namespace litert {
namespace tflite_support {
namespace {

constexpr size_t kMaxSelectedOpsCallbacks = 32;
SelectedOpsCallback g_selected_ops_callbacks[kMaxSelectedOpsCallbacks] = {};
size_t g_num_selected_ops_callbacks = 0;

}  // namespace

void RegisterSelectedOpsCallback(SelectedOpsCallback callback) {
  if (callback == nullptr) {
    return;
  }
  for (size_t i = 0; i < g_num_selected_ops_callbacks; ++i) {
    if (g_selected_ops_callbacks[i] == callback) {
      return;
    }
  }
  if (g_num_selected_ops_callbacks < kMaxSelectedOpsCallbacks) {
    g_selected_ops_callbacks[g_num_selected_ops_callbacks++] = callback;
  }
}

bool HasSelectedOpsCallback() { return g_num_selected_ops_callbacks > 0; }

void ApplySelectedOpsCallbacks(::tflite::MutableOpResolver* resolver) {
  if (resolver == nullptr) {
    return;
  }
  for (size_t i = 0; i < g_num_selected_ops_callbacks; ++i) {
    g_selected_ops_callbacks[i](resolver);
  }
}

}  // namespace tflite_support
}  // namespace litert
