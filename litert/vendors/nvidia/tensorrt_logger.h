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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_TENSORRT_LOGGER_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_TENSORRT_LOGGER_H_

#include "litert/c/internal/litert_logging.h"
#include "NvInfer.h"

namespace litert::nvidia {

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LITERT_LOG(LITERT_ERROR, "TensorRT: %s", msg);
        break;
      case Severity::kWARNING:
        LITERT_LOG(LITERT_WARNING, "TensorRT: %s", msg);
        break;
      case Severity::kINFO:
        LITERT_LOG(LITERT_INFO, "TensorRT: %s", msg);
        break;
      case Severity::kVERBOSE:
        LITERT_LOG(LITERT_DEBUG, "TensorRT: %s", msg);
        break;
    }
  }
};

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_TENSORRT_LOGGER_H_
