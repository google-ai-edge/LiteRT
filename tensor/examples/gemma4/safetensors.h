/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Source code origin:
// https://github.com/syoyo/safetensors-cpp/blob/main/safetensors.hh
//
// Original license:
// SPDX-License-Identifier: MIT Copyright 2023 - Present, Syoyo Fujita.
//
// Inspired from:
// https://gist.github.com/Narsil/5d6bf307995158ad2c4994f323967284

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_SAFETENSORS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_SAFETENSORS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensor/examples/gemma4/minijson.h"

#ifdef __ANDROID__
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
#include <android/asset_manager.h>
#else
struct AAssetManager;
#endif

extern AAssetManager* asset_manager;
#endif

namespace safetensors {

constexpr size_t kMaxDim =
    8;  // must be equal to SAFETENSORS_C_MAX_DIM in `safetensors-c.h`

enum dtype {
  kBOOL,
  kUINT8,
  kINT8,
  kINT16,
  kUINT16,
  kFLOAT16,
  kBFLOAT16,
  kINT32,
  kUINT32,
  kFLOAT32,
  kFLOAT64,
  kINT64,
  kUINT64,
};

template <typename T>
using ordered_dict = ::minijson::ordered_dict<T>;

struct tensor_t {
  safetensors::dtype dtype;
  std::vector<size_t> shape;
  std::array<size_t, 2> data_offsets;
};

struct safetensors_t {
  // we need ordered dict(preserves the order of key insertion)
  // as done in Python's OrderedDict, since JSON data may not be sorted by its
  // key string.
  ordered_dict<tensor_t> tensors;
  ordered_dict<std::string> metadata;
  std::vector<uint8_t> storage;  // empty when mmap'ed
  size_t header_size{0};         // JSON size

  bool mmaped{false};

  //
  // Following members are set when mmaped.
  //
  const uint8_t* mmap_addr{nullptr};
  size_t mmap_size{0};
  const uint8_t* databuffer_addr{nullptr};  // [mmap_addr + header_size + 8]
  size_t databuffer_size{0};                // mmap_size - header_size - 8
  // opaque pointer to safetensors_file and safetensors_mmap
  void* st_file{nullptr};
  void* st_mmap{nullptr};

  ~safetensors_t();
};

//
// Load safetensors from file.
// databuffer is copied to `safetensors_t::storage`.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool load_from_file(const std::string& filename, safetensors_t* st,
                    std::string* warn, std::string* err);

//
// Load safetensors data from memory.
// databuffer is copied to `safetensors_t::storage`.
//
// @param[in] addr Memory address of safetensors data.
// @param[in] nbytes The size in bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
//
bool load_from_memory(const uint8_t* addr, size_t nbytes,
                      const std::string& filename, safetensors_t* st,
                      std::string* warn, std::string* err);

//
// Load safetensors with memory mapping(i.e. zero-copy).
// databuffer is not copied to `safetensors_t` object, thus the app must hold
// file during `safetensor_t` object is live.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_file(const std::string& filename, safetensors_t* st,
                    std::string* warn, std::string* err);

//
// Load safetensors from mmaped region.
// databuffer is not copied to `safetensors_t` object, thus the app must not
// free/unmap `addr` during `safetensor_t` object is live.
//
// @param[in] addr mmaped memory address of safetensors data.
// @param[in] nbytes mmap bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_memory(const uint8_t* addr, size_t nbytes,
                      const std::string& filename, safetensors_t* st,
                      std::string* warn, std::string* err);

//
// Save safetensors to file.
//
// @param[in] st safetensors data.
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool save_to_file(const safetensors_t& st, const std::string& filename,
                  std::string* warn, std::string* err);

//
// Save safetensors to memory.
//
// @param[in] st safetensors data.
// @param[out] data_out Serialized safetensor data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool save_to_memory(const std::string& filename, std::vector<uint8_t>* data_out,
                    std::string* warn, std::string* err);

//
// Utility functions
//

// Returns shape[0] * shape[1] * ...
// Empty Tensor(any shape[i] is 0) returns 0.
// Zero-rank tensor([]) return 1.
size_t get_shape_size(const tensor_t& t);

// Returns dtype size in bytes.
size_t get_dtype_bytes(safetensors::dtype dtype);
std::string get_dtype_str(safetensors::dtype dtype);

// Validate data_offsets of all tensors in safetensors_t.
bool validate_data_offsets(const safetensors_t& st, std::string& err);

uint16_t float_to_bfloat16(float x);
float bfloat16_to_float(uint16_t x);

uint16_t float_to_fp16(float x);
float fp16_to_float(uint16_t x);

}  // namespace safetensors

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_SAFETENSORS_H_
