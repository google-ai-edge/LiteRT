#include <iostream>
#include <ostream>
#include <cstdlib>

namespace absl {
namespace lts_20250814 {
enum class LogSeverity : int {
  kInfo = 0,
  kWarning = 1,
  kError = 2,
  kFatal = 3,
};
}
}

namespace tsl {
namespace internal {

class LogMessage {
 public:
  LogMessage(const char* file, int line, absl::lts_20250814::LogSeverity severity);
  ~LogMessage();
  std::ostream& stream();
};

LogMessage::LogMessage(const char* file, int line, absl::lts_20250814::LogSeverity severity) {}
LogMessage::~LogMessage() {}
std::ostream& LogMessage::stream() { return std::cerr; }

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  [[noreturn]] ~LogMessageFatal();
};

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, absl::lts_20250814::LogSeverity::kFatal) {}
LogMessageFatal::~LogMessageFatal() {
    std::cerr << std::endl;
    abort();
}

} // namespace internal
} // namespace tsl
