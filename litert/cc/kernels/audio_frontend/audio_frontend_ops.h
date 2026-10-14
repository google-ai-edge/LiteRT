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

#ifndef ODML_LITERT_LITERT_CC_KERNELS_AUDIO_FRONTEND_AUDIO_FRONTEND_OPS_H_
#define ODML_LITERT_LITERT_CC_KERNELS_AUDIO_FRONTEND_AUDIO_FRONTEND_OPS_H_

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"

namespace litert {
namespace audio_frontend {

// Adds all audio frontend custom operations to the provided LiteRT Options.
Expected<void> AddAudioFrontendOps(Options& options);

}  // namespace audio_frontend
}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_KERNELS_AUDIO_FRONTEND_AUDIO_FRONTEND_OPS_H_
