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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_TENSORRT_FEATURES_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_TENSORRT_FEATURES_H_

#include "NvInferVersion.h"

// TensorRT-RTX 1.7 added build-time weight placeholders (a constant with a
// count and no values, stripped from the plan) and nvinfer1::IWeightsManager,
// which maps application-owned CUDA virtual memory into the weight memory of
// an engine. The weight store of the NVIDIA backend needs both.
#if defined(TRT_MAJOR_RTX) && \
    (TRT_MAJOR_RTX > 1 || (TRT_MAJOR_RTX == 1 && TRT_MINOR_RTX >= 7))
#define LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER 1
#else
#define LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER 0
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_TENSORRT_FEATURES_H_
