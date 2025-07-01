#ifndef TENSORFLOW_COMPILER_MLIR_LITE_ALLOCATION_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_ALLOCATION_H_

#include <cstddef>
#include <memory>

namespace tensorflow {

// Basic allocation interface for TensorFlow Lite
class Allocation {
 public:
  virtual ~Allocation() = default;
  
  // Returns a pointer to the allocated memory
  virtual const void* base() const = 0;
  
  // Returns the size of the allocation in bytes
  virtual size_t bytes() const = 0;
  
  // Returns whether the allocation is valid
  virtual bool valid() const = 0;
};

// Memory-mapped file allocation implementation
class MMAPAllocation : public Allocation {
 public:
  MMAPAllocation(const char* filename, bool should_copy = false);
  ~MMAPAllocation() override;
  
  const void* base() const override { return data_; }
  size_t bytes() const override { return size_; }
  bool valid() const override { return data_ != nullptr; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
  bool owns_data_ = false;
};

// Simple memory allocation implementation
class MemoryAllocation : public Allocation {
 public:
  MemoryAllocation(const void* ptr, size_t num_bytes);
  ~MemoryAllocation() override = default;
  
  const void* base() const override { return data_; }
  size_t bytes() const override { return size_; }
  bool valid() const override { return data_ != nullptr; }

 private:
  const void* data_;
  size_t size_;
};

// Utility function to create allocations
std::unique_ptr<Allocation> GetAllocationFromFile(
    const char* filename, bool mmap_file = true, bool copy_to_local = false);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_ALLOCATION_H_