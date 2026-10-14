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

#include "litert/cc/kernels/audio_frontend/audio_frontend_ops.h"

#include "litert/cc/kernels/audio_frontend/irfft_kernel.h"
#include "litert/cc/kernels/audio_frontend/overlap_add_kernel.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"

namespace litert {
namespace audio_frontend {

// Define instances of each kernel. These must live for the duration of the
// LiteRT runtime.
static IrfftKernel* irfft_kernel = new IrfftKernel();
static OverlapAddKernel* overlap_add_kernel = new OverlapAddKernel();

Expected<void> AddAudioFrontendOps(Options& options) {
  // TODO(b/466914743): Add more audio frontend ops.
  // Register IrfftKernel
  LITERT_RETURN_IF_ERROR(options.AddCustomOpKernel(*irfft_kernel));
  // Register OverlapAddKernel
  LITERT_RETURN_IF_ERROR(options.AddCustomOpKernel(*overlap_add_kernel));

  return {};
}

}  // namespace audio_frontend
}  // namespace litert
