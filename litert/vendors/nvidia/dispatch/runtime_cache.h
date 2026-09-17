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

#ifndef LITERT_VENDORS_NVIDIA_DISPATCH_RUNTIME_CACHE_H_
#define LITERT_VENDORS_NVIDIA_DISPATCH_RUNTIME_CACHE_H_

#include <cstddef>
#include <string>

#include "litert/cc/litert_expected.h"

namespace nvinfer1 {
class IRuntimeConfig;
}

namespace litert::nvidia {

// Validates candidate bytes in a fresh SDK cache, then atomically replaces path
// with those exact bytes. Does not attach or reserialize the validation cache.
// Validation or IO failure leaves the previous destination unchanged. The
// runtime configuration and candidate bytes must remain alive during this call.
Expected<void> PersistValidatedRuntimeCache(nvinfer1::IRuntimeConfig& config,
                                            const void* data, size_t size,
                                            const std::string& path);

}  // namespace litert::nvidia

#endif  // LITERT_VENDORS_NVIDIA_DISPATCH_RUNTIME_CACHE_H_
