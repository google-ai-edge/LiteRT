#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_METADATA_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_METADATA_UTIL_H_

#include <string>
#include <vector>

namespace tflite {
namespace experimental {
namespace remat {

// Metadata utilities for rematerialization
class MetadataUtil {
 public:
  // Extract metadata from model
  static std::vector<std::string> ExtractMetadata(const void* model_data, size_t model_size);
  
  // Check if rematerialization is enabled
  static bool IsRematEnabled(const void* model_data, size_t model_size);
  
  // Get rematerialization configuration
  static std::string GetRematConfig(const void* model_data, size_t model_size);
};

}  // namespace remat
}  // namespace experimental
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_METADATA_UTIL_H_