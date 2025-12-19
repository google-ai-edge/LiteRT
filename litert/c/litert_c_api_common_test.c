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

// This file exists to verify that the below header files can build, link,
// and run as C code.
#ifdef __cplusplus
#error "This file should be compiled as C code, not as C++."
#endif

// Include all the header files in the litert/c directory.
#include "litert/c/litert_any.h"             // NOLINT
#include "litert/c/litert_common.h"          // NOLINT
#include "litert/c/litert_compiled_model.h"  // NOLINT
#include "litert/c/litert_custom_tensor_buffer.h"  // NOLINT
#include "litert/c/litert_event.h"  // NOLINT
#include "litert/c/litert_layout.h"          // NOLINT
#include "litert/c/internal/litert_logging.h"         // NOLINT
#include "litert/c/litert_metrics.h"         // NOLINT
#include "litert/c/litert_model.h"           // NOLINT
#include "litert/c/litert_builder.h"           // NOLINT
#include "litert/c/litert_model_types.h"      // NOLINT
#include "litert/c/litert_op_code.h"         // NOLINT
#include "litert/c/litert_op_options.h"      // NOLINT
#include "litert/c/litert_options.h"         // NOLINT
#include "litert/c/litert_profiler.h"        // NOLINT
#include "litert/c/litert_profiler_event.h"  // NOLINT
#include "litert/c/litert_tensor_buffer.h"   // NOLINT
#include "litert/c/litert_tensor_buffer_requirements.h"  // NOLINT
#include "litert/c/internal/litert_accelerator.h"  // NOLINT
#include "litert/c/internal/litert_accelerator_registration.h"  // NOLINT
#include "litert/c/internal/litert_dispatch_delegate.h"  // NOLINT
#include "litert/c/internal/litert_external_litert_buffer_context.h"  // NOLINT
#include "litert/c/internal/litert_tensor_buffer_registry.h"  // NOLINT
#include "litert/c/options/litert_cpu_options.h"  // NOLINT
#include "litert/c/options/litert_google_tensor_options.h"  // NOLINT
#include "litert/c/options/litert_google_tensor_options_type.h"  // NOLINT
#include "litert/c/options/litert_gpu_options.h"  // NOLINT
#include "litert/c/options/litert_mediatek_options.h"  // NOLINT
#include "litert/c/options/litert_qualcomm_options.h"  // NOLINT
#include "litert/c/options/litert_runtime_options.h"  // NOLINT

int main(void) { return 0; }
