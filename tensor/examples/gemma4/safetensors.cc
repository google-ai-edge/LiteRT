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
#include "tensor/examples/gemma4/safetensors.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/minijson.h"
#include "tensor/internal/fp16.h"

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>  // for _fseeki64
#include <windows.h>
#endif

#ifdef __ANDROID__
AAssetManager* asset_manager = nullptr;
#endif

namespace safetensors {

// Max header(JSON) size. 100 MB as done in original safetensors implementation.
constexpr size_t kMaxJSONSize = 1024ull * 1024ull * 100ull;

namespace detail {

#ifdef _WIN32
std::wstring UTF8ToWchar(const std::string& str) {
  int wstr_size = MultiByteToWideChar(CP_UTF8, 0, str.data(),
                                      static_cast<int>(str.size()), nullptr, 0);
  std::wstring wstr(size_t(wstr_size), 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.size()),
                      &wstr[0], static_cast<int>(wstr.size()));
  return wstr;
}

std::string WcharToUTF8(const std::wstring& wstr) {
  int str_size = WideCharToMultiByte(CP_UTF8, 0, wstr.data(),
                                     static_cast<int>(wstr.size()), nullptr, 0,
                                     nullptr, nullptr);
  std::string str(size_t(str_size), 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.data(), static_cast<int>(wstr.size()),
                      &str[0], static_cast<int>(str.size()), nullptr, nullptr);
  return str;
}
#endif

bool ReadWholeFile(std::vector<unsigned char>* out, std::string* err,
                   const std::string& filepath, void*) {
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
  if (asset_manager) {
    AAsset* asset = AAssetManager_open(asset_manager, filepath.c_str(),
                                       AASSET_MODE_STREAMING);
    if (!asset) {
      if (err) {
        (*err) += "File open error : " + filepath + "\n";
      }
      return false;
    }
    size_t size = AAsset_getLength(asset);
    if (size == 0) {
      if (err) {
        (*err) += "Invalid file size : " + filepath +
                  " (does the path point to a directory?)";
      }
      return false;
    }
    out->resize(size);
    AAsset_read(asset, reinterpret_cast<char*>(&out->at(0)), size);
    AAsset_close(asset);
    return true;
  } else {
    if (err) {
      (*err) += "No asset manager specified : " + filepath + "\n";
    }
    return false;
  }
#else
#ifdef _WIN32
#if defined(__GLIBCXX__)  // mingw
  int file_descriptor =
      _wopen(UTF8ToWchar(filepath).c_str(), _O_RDONLY | _O_BINARY);
  // NOLINTNEXTLINE(build/deprecated): imported code, modify when needed.
  __gnu_cxx::stdio_filebuf<char> wfile_buf(file_descriptor, std::ios_base::in);
  std::istream f(&wfile_buf);
#elif defined(_MSC_VER) || defined(_LIBCPP_VERSION)
  // For libcxx, assume _LIBCPP_HAS_OPEN_WITH_WCHAR is defined to accept
  // `wchar_t *`
  std::ifstream f(UTF8ToWchar(filepath).c_str(), std::ifstream::binary);
#else
  // Unknown compiler/runtime
  std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
#else
  std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
  if (!f) {
    if (err) {
      (*err) += "File open error : " + filepath + "\n";
    }
    return false;
  }

  // For directory(and pipe?), peek() will fail(Posix gnustl/libc++ only)
  f.peek();
  if (!f) {
    if (err) {
      (*err) +=
          "File read error. Maybe empty file or invalid file : " + filepath +
          "\n";
    }
    return false;
  }

  f.seekg(0, f.end);
  size_t sz = static_cast<size_t>(f.tellg());

  // std::cout << "sz = " << sz << "\n";
  f.seekg(0, f.beg);

  if (static_cast<int64_t>(sz) < 0) {
    if (err) {
      (*err) += "Invalid file size : " + filepath +
                " (does the path point to a directory?)";
    }
    return false;
  } else if (sz == 0) {
    if (err) {
      (*err) += "File is empty : " + filepath + "\n";
    }
    return false;
  } else if (sz >= (std::numeric_limits<std::streamoff>::max)()) {
    if (err) {
      (*err) += "Invalid file size : " + filepath + "\n";
    }
    return false;
  }

  out->resize(sz);
  f.read(reinterpret_cast<char*>(&out->at(0)),
         static_cast<std::streamsize>(sz));

  return true;
#endif
}

bool parse_metadata(const ::minijson::value& v, ordered_dict<std::string>& dst,
                    std::string* err) {
  if (auto po = v.as<::minijson::object>()) {
    for (size_t i = 0; i < po->size(); i++) {
      ::minijson::value ov;
      if (!po->at(i, &ov)) {
        if (err) {
          (*err) +=
              "[Internal error] Invalid object found in __metadata__, at "
              "index " +
              std::to_string(i) + ".\n";
        }
        return false;
      }

      if (auto so = ov.as<std::string>()) {
        if (dst.count(po->keys()[i])) {
          // This should not be happen though
          if (err) {
            (*err) += "Duplicate key `" + po->keys()[i] +
                      "` found in __metadata__.\n";
          }
          return false;
        }

        dst.insert(po->keys()[i], *so);
      } else {
        if (err) {
          (*err) += "`" + po->keys()[i] + "` must be string value.\n";
        }
        return false;
      }
    }
  } else {
    if (err) {
      (*err) += "`__metadata__` value must be JSON object.\n";
    }
    return false;
  }

  return true;
}

bool parse_dtype(const ::minijson::value& v, safetensors::dtype& dtype,
                 std::string* err) {
  if (auto so = v.as<std::string>()) {
    if ((*so) == "BOOL") {
      dtype = safetensors::dtype::kBOOL;
    } else if ((*so) == "U8") {
      dtype = safetensors::dtype::kUINT8;
    } else if ((*so) == "I8") {
      dtype = safetensors::dtype::kINT8;
    } else if ((*so) == "U16") {
      dtype = safetensors::dtype::kUINT16;
    } else if ((*so) == "I16") {
      dtype = safetensors::dtype::kINT16;
    } else if ((*so) == "U32") {
      dtype = safetensors::dtype::kUINT32;
    } else if ((*so) == "I32") {
      dtype = safetensors::dtype::kINT32;
    } else if ((*so) == "U64") {
      dtype = safetensors::dtype::kUINT64;
    } else if ((*so) == "I64") {
      dtype = safetensors::dtype::kINT64;
    } else if ((*so) == "F16") {
      dtype = safetensors::dtype::kFLOAT16;
    } else if ((*so) == "BF16") {
      dtype = safetensors::dtype::kBFLOAT16;
    } else if ((*so) == "F32") {
      dtype = safetensors::dtype::kFLOAT32;
    } else if ((*so) == "F64") {
      dtype = safetensors::dtype::kFLOAT64;
    } else {
      if (err) {
        (*err) += "Unknown `dtype` string: " + *so + ".\n";
      }
      return false;
    }
  } else {
    if (err) {
      (*err) +=
          "`dtype` item should be string type but got " + v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_shape(const ::minijson::value& v, std::vector<size_t>& dst,
                 std::string* err) {
  // NOTE:
  // - Empty tensors (tensors with 1 dimension being 0) are allowed
  // - [] is allowed(0-Rank tensor = merely a scalar)
  if (auto pa = v.as<::minijson::array>()) {
    ::minijson::array::const_iterator i;

    for (i = pa->begin(); i != pa->end(); i++) {
      if (auto pn = i->as<::minijson::number>()) {
        if (dst.size() >= kMaxDim) {
          if (err) {
            (*err) += "`shape` length must be less than " +
                      std::to_string(kMaxDim) + " but got " +
                      std::to_string(dst.size()) + ".\n";
          }
          return false;
        }

        dst.push_back(size_t(*pn));

      } else {
        if (err) {
          (*err) += "Array item in `shape` must be number type, but got " +
                    i->type_name() + ".\n";
        }
        return false;
      }
    }
  } else {
    if (err) {
      (*err) +=
          "`shape` value must be JSON array, but got " + v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_data_offsets(const ::minijson::value& v, std::array<size_t, 2>& dst,
                        std::string* err) {
  if (auto pa = v.as<::minijson::array>()) {
    ::minijson::array::const_iterator i;
    size_t cnt = 0;

    for (i = pa->begin(); i != pa->end(); i++) {
      if (auto pn = i->as<::minijson::number>()) {
        if (cnt >= 2) {
          if (err) {
            (*err) += "`data_offsets` length must be 2.\n";
          }
          return false;
        }

        dst[cnt] = size_t(*pn);

        cnt++;

      } else {
        if (err) {
          (*err) +=
              "Array item in `data_offsets` must be number type, but got " +
              i->type_name() + ".\n";
        }
        return false;
      }
    }

    if (cnt != 2) {
      if (err) {
        (*err) += "`data_offsets` length must be 2.\n";
      }
      return false;
    }
  } else {
    if (err) {
      (*err) += "`data_offsets` value must be JSON array, but got " +
                v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_tensor(const std::string& name, const ::minijson::value& v,
                  tensor_t& tensor, std::string* err) {
  if (auto po = v.as<::minijson::object>()) {
    bool dtype_found{false};
    bool shape_found{false};
    bool data_offsets_found{false};

    dtype dtype;
    std::vector<size_t> shape;
    std::array<size_t, 2> data_offsets{};

    for (size_t i = 0; i < po->size(); i++) {
      std::string key = po->keys()[i];

      if (key == "dtype") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `dtype` has invalid object.\n";
          }
          return false;
        }

        if (!parse_dtype(value, dtype, err)) {
          return false;
        }

        dtype_found = true;
      } else if (key == "shape") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `shape` has invalid object.\n";
          }
          return false;
        }

        if (!parse_shape(value, shape, err)) {
          return false;
        }

        shape_found = true;
      } else if (key == "data_offsets") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `data_offsets` has invalid object.\n";
          }
          return false;
        }
        if (!parse_data_offsets(value, data_offsets, err)) {
          return false;
        }

        data_offsets_found = true;
      } else {
        // Unknown key. Report error?
      }
    }

    if (!dtype_found) {
      if (err) {
        (*err) += "`" + name + "` does not have `dtype` item.\n";
      }
      return false;
    }

    if (!shape_found) {
      if (err) {
        (*err) += "`" + name + "` does not have `shape` item.\n";
      }
      return false;
    }

    bool is_empty_tensor{false};
    if (!shape.empty()) {
      for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0) {
          is_empty_tensor = true;
          break;
        }
      }
    }

    if (is_empty_tensor) {
      // They are not storing any data in the databuffer, yet retaining size in
      // the header. So ignore data_offsets
      if (data_offsets_found) {
        // TODO: make this warn instead of err?
        if (err) {
          (*err) +=
              "`" + name +
              "` is empty tensors(tensors with 1 dimension being 0), and no "
              "data in databuffer, but `data_offsets` item is provided.\n";
        }
        return false;
      }
    } else {
      if (!data_offsets_found) {
        if (err) {
          (*err) += "`" + name + "` does not have `data_offsets` item.\n";
        }
        return false;
      }
    }

    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.data_offsets = data_offsets;

  } else {
    if (err) {
      (*err) += "`" + name + "` value must be JSON object.\n";
    }
    return false;
  }

  return true;
}

// From llama.cpp
#if defined(_WIN32)
static std::string safetensors_format_win_err(DWORD err) {
  LPSTR buf;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0,
      NULL);
  if (!size) {
    return "FormatMessageA failed";
  }
  std::string ret(buf, size);
  LocalFree(buf);
  return ret;
}
#endif

struct safetensors_file {
  // use FILE * so we don't have to re-open the file to mmap
  FILE* fp{nullptr};
  size_t size{0};
  mutable bool valid{false};
  std::string err;

  safetensors_file(const char* fname, const char* mode) {
    fp = std::fopen(fname, mode);
    if (fp == nullptr) {
      err = "failed to open " + std::string(fname) + ":" +
            std::string(strerror(errno)) + "\n";
      valid = false;
    } else {
      seek(0, SEEK_END);
      size = tell();
      seek(0, SEEK_SET);
      valid = true;
    }
  }

  ~safetensors_file() {
    if (fp) {
      std::fclose(fp);
      fp = nullptr;
    }
  }

  size_t tell() const {
#ifdef _WIN32
    auto ret = _ftelli64(fp);
#else
    auto ret = std::ftell(fp);
#endif
    if (ret == -1) {
      // this really shouldn't fail
      valid = false;
      return 0;
    }

    return (size_t)ret;
  }

  void seek(size_t offset, int whence) const {
#ifdef _WIN32
    auto ret = _fseeki64(fp, (__int64)offset, whence);
#else
    // NOLINTNEXTLINE(*-runtime-int)
    auto ret = std::fseek(fp, static_cast<long>(offset), whence);
#endif
    if (ret == 0) {
      valid = false;
    }
  }

  bool& is_valid() const { return valid; }

  const std::string& get_error() const { return err; }
};

struct safetensors_mmap {
  uint8_t* addr{nullptr};
  size_t size{0};

  bool valid{false};
  std::string warn;
  std::string err;

  bool is_valid() const { return valid; }

  const std::string& get_error() const { return err; }

  const std::string& get_warning() const { return warn; }

  safetensors_mmap(const safetensors_mmap&) = delete;

#ifdef _POSIX_MAPPED_FILES
  static constexpr bool kSupported = true;

  explicit safetensors_mmap(struct safetensors_file* file,
                            size_t prefetch = (size_t)-1 /* -1 = max value */,
                            bool numa = false) {
    size = file->size;
    int fd = fileno(file->fp);
    int flags = MAP_SHARED;
    // prefetch/readahead impairs performance on NUMA systems
    if (numa) {
      prefetch = 0;
    }
#ifdef __linux__
    if (prefetch) {
      flags |= MAP_POPULATE;
    }
#endif
    addr = reinterpret_cast<uint8_t*>(
        mmap(nullptr, file->size, PROT_READ, flags, fd, 0));
    if (addr == MAP_FAILED) {
      valid = false;
      err = "mmap failed: " + std::string(strerror(errno)) + "\n";

      size = 0;
      addr = nullptr;

      return;
    }

    if (prefetch > 0) {
      // Advise the kernel to preload the mapped memory.
#if defined(POSIX_MADV_WILLNEED)
      if (posix_madvise(addr, std::min(file->size, prefetch),
                        POSIX_MADV_WILLNEED)) {
        warn += "posix_madvise(.., POSIX_MADV_WILLNEED) failed: " +
                std::string(strerror(errno)) + "\n";
      }
#elif defined(MADV_WILLNEED)
      if (madvise(addr, std::min(file->size, prefetch), MADV_WILLNEED)) {
        warn += "madvise(.., MADV_WILLNEED) failed: " +
                std::string(strerror(errno)) + "\n";
      }
#endif
    }
    if (numa) {
      // Advise the kernel not to use readahead
      // (because the next page might not belong on the same node).
#if defined(POSIX_MADV_RANDOM)
      if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
        warn += "posix_madvise(.., POSIX_MADV_RANDOM) failed: " +
                std::string(strerror(errno)) + "\n";
      }
#elif defined(MADV_RANDOM)
      if (madvise(addr, file->size, MADV_RANDOM)) {
        warn +=
            "madvise(.., MADV_RANDOM) failed: " + std::string(strerror(errno)) +
            "\n";
      }
#endif
    }

    valid = true;
  }

  ~safetensors_mmap() {
    if (valid) {
      munmap(addr, size);
    }
    size = 0;
    addr = nullptr;
    valid = false;
  }

#elif defined(_WIN32)
  static constexpr bool kSupported = true;

  safetensors_mmap(struct safetensors_file* file, bool prefetch = true,
                   bool numa = false) {
    (void)numa;

    size = file->size;

    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    DWORD error = GetLastError();

    if (hMapping == NULL) {
      // TODO: get error message
      err = "CreateFileMappingA failed: " + safetensors_format_win_err(error) +
            "\n";
      valid = false;
      size = 0;
      addr = nullptr;
      return;
    }

    addr = reinterpret_cast<uint8_t*>(
        MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
    error = GetLastError();
    CloseHandle(hMapping);

    if (addr == NULL) {
      err = "MapViewOfFile failed: " + safetensors_format_win_err(error) + "\n";
    }

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
    if (prefetch) {
      // PrefetchVirtualMemory is only present on Windows 8 and above, so we
      // dynamically load it
      BOOL(WINAPI * pPrefetchVirtualMemory)(HANDLE, ULONG_PTR,
                                            PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      // may fail on pre-Windows 8 systems
      pPrefetchVirtualMemory =
          reinterpret_cast<decltype(pPrefetchVirtualMemory)>(
              GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

      if (pPrefetchVirtualMemory) {
        // advise the kernel to preload the mapped memory
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = (SIZE_T)size;
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
          warn += "PrefetchVirtualMemory failed: " +
                  safetensors_format_win_err(GetLastError()) + "\n";
        }
      }
    }
#endif
  }
  ~safetensors_mmap() {
    if (!UnmapViewOfFile(addr)) {
      warn += "UnmapViewOfFile failed: " +
              safetensors_format_win_err(GetLastError()) + "\n";
    }
  }
#else
  static constexpr bool kSupported = false;

  safetensors_mmap(struct safetensors_file* file, bool prefetch = true,
                   bool numa = false) {
    (void)file;
    (void)prefetch;
    (void)numa;

    valid = false;
    err = "mmap not supported\n";
    addr = nullptr;
    size = 0;
  }
#endif
};

bool parse_safetensors_header(const uint8_t* addr, const size_t nbytes,
                              const std::string& filename, safetensors_t* st,
                              std::string* warn, std::string* err) {
  if (nbytes < 16) {
    if (err) {
      (*err) += "Size is too short.\n";
    }
    return false;
  }

  uint64_t header_size{0};
  memcpy(reinterpret_cast<unsigned char*>(&header_size), addr,
         sizeof(uint64_t));

  if (header_size < 4) {
    if (err) {
      (*err) += "Header size is too short.\n";
    }
    return false;
  }

  if ((8 + header_size) > nbytes) {
    if (err) {
      (*err) += "Header size " + std::to_string(header_size) +
                " + 8 exceeds input size " + std::to_string(nbytes) + " .\n";
    }
    return false;
  }

  if (header_size > kMaxJSONSize) {
    if (err) {
      (*err) += "Header JSON size exceeds the limit(" +
                std::to_string(kMaxJSONSize) + ").\n";
    }
    return false;
  }

  // assume JSON data is small enough.
  std::string json_str(reinterpret_cast<const char*>(&addr[8]), header_size);
  const char* p = json_str.c_str();

  ::minijson::value v;
  ::minijson::error e = ::minijson::parse(p, v);

  if (e != ::minijson::no_error) {
    if (err) {
      std::string json_err(::minijson::errstr(e));
      (*err) += "JSON parse error: " + json_err + "\n";
    }

    return false;
  }

  ordered_dict<tensor_t> tensors;
  ordered_dict<std::string> metadata;

  // root element must be dict.
  if (auto po = v.as<::minijson::object>()) {
    for (size_t i = 0; i < po->size(); i++) {
      std::string key = po->keys()[i];

      if (key == "__metadata__") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. Invalid object in __metadata__.\n";
          }
          return false;
        }

        if (!detail::parse_metadata(value, metadata, err)) {
          return false;
        }
      } else {
        // tensor

        if (tensors.count(key)) {
          if (err) {
            (*err) += "Duplicate key `" + key + "` found.\n";
          }
          return false;
        }

        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. Invalid object in `" + key + "`.\n";
          }
          return false;
        }

        tensor_t tensor;
        if (!detail::parse_tensor(key, value, tensor, err)) {
          return false;
        }

        tensors.insert(key, std::move(tensor));
      }
    }
  } else {
    if (err) {
      (*err) += "JSON root elements must be object(dict)\n";
    }
  }

  st->tensors = std::move(tensors);
  st->metadata = std::move(metadata);
  st->header_size = header_size;

#if 0
  size_t databuffer_size = nbytes - header_size - 8;

  st->storage.resize(nbytes);
  memcpy(st->storage.data(), addr + 8 + header_size, nbytes);

  st->mmaped = false;
  st->mmap_addr = addr + 8 + header_size;
  st->mmap_size = 0;
#endif

  return true;
}

}  // namespace detail

safetensors_t::~safetensors_t() {
  if (st_mmap) {
    detail::safetensors_mmap* p =
        reinterpret_cast<detail::safetensors_mmap*>(st_mmap);
    delete p;
    st_mmap = nullptr;
  }

  if (st_file) {
    detail::safetensors_file* p =
        reinterpret_cast<detail::safetensors_file*>(st_file);
    delete p;
    st_file = nullptr;
  }
}

//
// - 8byte: header_size
// - json data(header_size bytes)
// - tensor data(filesize - header_size)
//

bool load_from_file(const std::string& filename, safetensors_t* st,
                    std::string* warn, std::string* err) {
  std::vector<unsigned char> data;
  if (!detail::ReadWholeFile(&data, err, filename, nullptr)) {
    return false;
  }

  return load_from_memory(reinterpret_cast<const uint8_t*>(data.data()),
                          data.size(), filename, st, warn, err);
}

bool load_from_memory(const uint8_t* addr, const size_t nbytes,
                      const std::string& filename, safetensors_t* st,
                      std::string* warn, std::string* err) {
  if (nbytes < 16) {
    if (err) {
      (*err) += "Size is too short.\n";
    }
    return false;
  }

  if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn,
                                        err)) {
    return false;
  }

  size_t databuffer_size = nbytes - st->header_size - 8;

  st->storage.resize(databuffer_size);
  memcpy(st->storage.data(), addr + 8 + st->header_size, databuffer_size);

  st->mmaped = false;
  st->mmap_addr = nullptr;
  st->mmap_size = 0;
  st->databuffer_addr = nullptr;
  st->databuffer_size = 0;

  return true;
}

bool mmap_from_file(const std::string& filename, safetensors_t* st,
                    std::string* warn, std::string* err) {
  if (!st) {
    return false;
  }

  auto pf = std::make_unique<detail::safetensors_file>(filename.c_str(), "rb");
  if (!pf->is_valid()) {
    if (err) {
      (*err) += pf->get_error();
    }
    return false;
  }

  // TODO: prefetch, numa
  auto pm =
      std::make_unique<detail::safetensors_mmap>(pf.get(), /*prefetch=*/0);

  bool ret = mmap_from_memory(pm->addr, pm->size, filename, st, warn, err);

  if (!ret) {
    return false;
  }

  st->mmap_addr = pm->addr;
  st->mmap_size = pm->size;

  st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
  st->databuffer_size = st->mmap_size - (8 + st->header_size);

  // retain pointer
  st->st_file = pf.release();
  st->st_mmap = pm.release();

  st->mmaped = true;

  return true;
}

bool mmap_from_memory(const uint8_t* addr, const size_t nbytes,
                      const std::string& filename, safetensors_t* st,
                      std::string* warn, std::string* err) {
  if (!addr) {
    return false;
  }

  if (nbytes < 16) {
    return false;
  }

  if (!st) {
    return false;
  }

  if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn,
                                        err)) {
    return false;
  }

  st->mmaped = true;

  st->mmap_addr = addr;
  st->mmap_size = nbytes;

  st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
  st->databuffer_size = st->mmap_size - (8 + st->header_size);

  return true;
}

float bfloat16_to_float(uint16_t x) {
  litert::tensor::bf16_t y;
  y.val = x;
  return y;
}

uint16_t float_to_bfloat16(float x) {
  return litert::tensor::bf16_t::fp32_to_bf16(x);
}

float fp16_to_float(uint16_t x) {
  return litert::tensor::fp16_ieee_to_fp32_value(x);
}

uint16_t float_to_fp16(float x) {
  return litert::tensor::fp16_ieee_from_fp32_value(x);
}

size_t get_dtype_bytes(const safetensors::dtype dtype) {
  size_t sz = 0;

  switch (dtype) {
    case safetensors::dtype::kBOOL:
      // Original Rust implementaion uses 1.
      sz = 1;
      break;
    case safetensors::dtype::kUINT8:
      sz = 1;
      break;
    case safetensors::dtype::kINT8:
      sz = 1;
      break;
    case safetensors::dtype::kUINT16:
      sz = 2;
      break;
    case safetensors::dtype::kINT16:
      sz = 2;
      break;
    case safetensors::dtype::kINT32:
      sz = 4;
      break;
    case safetensors::dtype::kUINT32:
      sz = 4;
      break;
    case safetensors::dtype::kFLOAT16:
      sz = 2;
      break;
    case safetensors::dtype::kBFLOAT16:
      sz = 2;
      break;
    case safetensors::dtype::kFLOAT32:
      sz = 4;
      break;
    case safetensors::dtype::kFLOAT64:
      sz = 8;
      break;
    case safetensors::dtype::kINT64:
      sz = 8;
      break;
    case safetensors::dtype::kUINT64:
      sz = 8;
      break;
  }

  return sz;
}

std::string get_dtype_str(const safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kBOOL:
      return "BOOL";
    case safetensors::dtype::kUINT8:
      return "U8";
    case safetensors::dtype::kINT8:
      return "I8";
    case safetensors::dtype::kUINT16:
      return "U16";
    case safetensors::dtype::kINT16:
      return "I16";
    case safetensors::dtype::kINT32:
      return "I32";
    case safetensors::dtype::kUINT32:
      return "U32";
    case safetensors::dtype::kFLOAT16:
      return "F16";
    case safetensors::dtype::kBFLOAT16:
      return "BF16";
    case safetensors::dtype::kFLOAT32:
      return "F32";
    case safetensors::dtype::kFLOAT64:
      return "F64";
    case safetensors::dtype::kINT64:
      return "I64";
    case safetensors::dtype::kUINT64:
      return "U64";
  }
  return "???";
}

// Empty Tensor returns 0.
// Zero-rank Tensor reuturns 1(scalar)
size_t get_shape_size(const tensor_t& t) {
  if (t.shape.empty()) {
    return 1;
  }

  if (t.shape.size() >= kMaxDim) {  // invalid ndim
    return 0;
  }

  size_t sz = 1;

  for (size_t i = 0; i < t.shape.size(); i++) {
    sz *= t.shape[i];
  }

  return sz;
}

bool validate_data_offsets(const safetensors_t& st, std::string& err) {
  bool valid{true};

  std::stringstream ss;

  size_t databuffersize;
  if (st.mmaped) {
    databuffersize = st.databuffer_size;
  } else {
    databuffersize = st.storage.size();
  }

  size_t ntensors{0};
  // Iterate with key insertion order.
  for (size_t i = 0; i < st.tensors.size(); i++) {
    std::string key = st.tensors.keys()[i];

    tensor_t tensor;
    if (!st.tensors.at(i, &tensor)) {
      ss << "Internal error: Failed to get tensor at [" << i << "]\n";
      valid = false;
      continue;
    }

    if (tensor.data_offsets[0] > tensor.data_offsets[1]) {
      ss << key << ".data_offsets.BEGIN " << tensor.data_offsets[0]
         << " must be less than or equal to data_offsets.END "
         << tensor.data_offsets[1] << "\n";
      valid = false;
    }

    size_t tensor_size = get_dtype_bytes(tensor.dtype) * get_shape_size(tensor);

    if (tensor_size == 0) {
      // OK
      continue;
    }

    // data_offsets are absolute offset from the databuffer(file)
    if (tensor.data_offsets[0] > databuffersize) {
      ss << "Tensor `" << key << "`.data_offset.BEGIN "
         << tensor.data_offsets[0] << " exceeds databuffer size "
         << databuffersize << ".\n";
      valid = false;
    }

    if (tensor.data_offsets[1] > databuffersize) {
      ss << "Tensor `" << key << "`.data_offset.END " << tensor.data_offsets[1]
         << " exceeds databuffer size " << databuffersize << ".\n";
      valid = false;
    }

    size_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];

    if (tensor_size != data_size) {
      ss << "Data size mismatch. The size in Tensor `" << key << "` is "
         << tensor_size << ", but the size from data_offsets is " << data_size
         << "\n";
      valid = false;
    }

    ntensors++;
    if (ntensors == st.tensors.size()) {
      // Last element's data_offsets[1] must be equal to databuffer size.
      if (tensor.data_offsets[1] != databuffersize) {
        ss << "The last tensor's data_offset.END(" << tensor.data_offsets[1]
           << ") must be equal to databufer size " << databuffersize << ".\n";
        valid = false;
      }
    }
  }

  if (!valid) {
    err = ss.str();
  }

  return valid;
}

bool save_to_memory(const safetensors_t& st, std::vector<uint8_t>* dst,
                    std::string* warn, std::string* err) {
  // directly serialize JSON string.
  std::stringstream ss;

  // NOTE: The last offset **must** be the end of the file,
  // so write __metadata__ first(if metadata part exists)

  std::string current_err;
  if (!validate_data_offsets(st, current_err)) {
    if (err) {
      (*err) += "Invalid safensors is provided.\n";
      (*err) += current_err;
    }
    return false;
  }

  ss << "{";
  if (st.metadata.size()) {
    ss << "\"__metadata__\": {";
    size_t nmeta = 0;
    for (size_t i = 0; i < st.metadata.size(); i++) {
      std::string key = st.metadata.keys()[i];
      std::string value;
      st.metadata.at(i, &value);

      if (nmeta > 0) {
        ss << ", ";
      }
      ss << "\"" + key + "\": \"" << value << "\"";
      nmeta++;
    }
    ss << "}";

    if (st.tensors.size()) {
      ss << ", ";
    }
  }

  size_t ntensors = 0;
  {
    for (size_t i = 0; i < st.tensors.size(); i++) {
      std::string key = st.tensors.keys()[i];
      safetensors::tensor_t tensor;
      st.tensors.at(i, &tensor);

      if (tensor.shape.size() > safetensors::kMaxDim) {
        if (err) {
          (*err) += key + ".shape is too large.\n";
          (*err) += current_err;
        }
        return false;
      }

      if (ntensors > 0) {
        ss << ", ";
      }
      ss << "\"" << key << "\": {";
      ss << "\"dtype\": \"" << safetensors::get_dtype_str(tensor.dtype)
         << "\", ";
      ss << "\"shape\": [";
      for (size_t i = 0; i < tensor.shape.size(); i++) {
        if (i > 0) {
          ss << ", ";
        }
        ss << tensor.shape[i];
      }
      ss << "]";
      ss << ", \"data_offsets\": [" << tensor.data_offsets[0] << ", "
         << tensor.data_offsets[1] << "]";
      ss << "}";
      ntensors++;
    }
  }
  ss << "}";

  std::string header_str = ss.str();

  uint64_t header_size = header_str.size();  // do not include '\n'

  const void* databuffer_addr{nullptr};
  size_t databuffer_size{0};
  if (st.mmaped) {
    databuffer_size = st.databuffer_size;
    databuffer_addr = st.databuffer_addr;
  } else {
    databuffer_size = st.storage.size();
    databuffer_addr = reinterpret_cast<const void*>(st.storage.data());
  }

  // make databuffer addr start from the multiple of 8.
  size_t pad_bytes = 0;
  if ((header_size % 8) != 0) {
    pad_bytes = 8 - (header_size % 8);
  }
  // printf("header_size = %d\n", static_cast<int>(header_size));
  // printf("pad_bytes = %d\n", static_cast<int>(pad_bytes));
  size_t padded_header_size = header_size + pad_bytes;
  dst->resize(8 + padded_header_size + databuffer_size);

  // write padded header_size
  memcpy(dst->data(), &padded_header_size, 8);

  // write header
  memcpy(dst->data() + 8, header_str.data(), header_size);

  // Use whitespace for trailing padding.
  memset(dst->data() + 8 + header_size, 0x20, pad_bytes);

  memcpy(dst->data() + 8 + padded_header_size, databuffer_addr,
         databuffer_size);

  return true;
}

bool save_to_file(const safetensors_t& st, const std::string& filename,
                  std::string* warn, std::string* err) {
  // TODO: Use more reliable io.
  std::ofstream ofs(filename, std::ios::binary);

  if (!ofs) {
    if (err) {
      (*err) += "Failed to open `" + filename +
                "` to write. File is either existing directory or "
                "write-protected, or disk is full?\n";
    }
    return false;
  }

  std::vector<uint8_t> buf;
  if (!save_to_memory(st, &buf, warn, err)) {
    return false;
  }

  ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
  if (!ofs) {
    if (err) {
      (*err) += "Failed to write safetensor data to `" + filename +
                "`. Maybe no disk space available?(Required bytes : " +
                std::to_string(buf.size()) + "\n";
    }
    return false;
  }

  return true;
}

}  // namespace safetensors
