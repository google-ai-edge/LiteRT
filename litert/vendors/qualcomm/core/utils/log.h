// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_LOG_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_LOG_H_

#include <cstdio>

#include "litert/vendors/qualcomm/core/common.h"

namespace qnn {

class QNNLogger {
 public:
  // Logging hook that takes variadic args.
  static void Log(LiteRtQnnLogLevel severity, const char* format, ...);

  // Set file descriptor
  static void SetLogFilePointer(FILE* fp);

  // Set log level
  static void SetLogLevel(LiteRtQnnLogLevel log_level);

 private:
  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  static FILE* log_file_pointer_;
  static LiteRtQnnLogLevel log_level_;
  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
};
}  // namespace qnn

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_VERBOSE(format, ...)                                     \
  ::qnn::QNNLogger::Log(kQnnLogLevelVerbose, ("VERBOSE: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_INFO(format, ...)                                  \
  ::qnn::QNNLogger::Log(kQnnLogLevelInfo, ("INFO: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_WARNING(format, ...)                                  \
  ::qnn::QNNLogger::Log(kQnnLogLevelWarn, ("WARNING: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_ERROR(format, ...)                                   \
  ::qnn::QNNLogger::Log(kQnnLogLevelError, ("ERROR: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_DEBUG(format, ...)                                   \
  ::qnn::QNNLogger::Log(kQnnLogLevelDebug, ("DEBUG: [Qnn] " format), \
                        ##__VA_ARGS__);

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_LOG_H_
