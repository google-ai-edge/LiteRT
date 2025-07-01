#ifndef TFLITE_PROFILING_TIME_H_
#define TFLITE_PROFILING_TIME_H_

#include <stdint.h>

namespace tflite {
namespace profiling {
namespace time {

// Get current time in microseconds
uint64_t NowMicros();

}  // namespace time
}  // namespace profiling
}  // namespace tflite

#endif  // TFLITE_PROFILING_TIME_H_