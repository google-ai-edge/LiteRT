#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_API_ERROR_REPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_API_ERROR_REPORTER_H_

#include <cstdarg>
#include <string>

namespace tflite {

// Interface for reporting errors during model execution
class ErrorReporter {
 public:
  virtual ~ErrorReporter() = default;
  
  // Report an error with a formatted message
  virtual int Report(const char* format, va_list args) = 0;
  
  // Report an error with a simple string
  virtual int Report(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = Report(format, args);
    va_end(args);
    return result;
  }
  
  // Report an error with a std::string
  virtual int ReportError(const std::string& message) {
    return Report("%s", message.c_str());
  }
};

// Default error reporter that prints to stderr
class StderrReporter : public ErrorReporter {
 public:
  int Report(const char* format, va_list args) override;
  
  static ErrorReporter* GetInstance();
};

// Get the default error reporter
ErrorReporter* DefaultErrorReporter();

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_API_ERROR_REPORTER_H_