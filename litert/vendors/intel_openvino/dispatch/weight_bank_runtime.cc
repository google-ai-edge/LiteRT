// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/intel_openvino/dispatch/weight_bank_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"

// Temp-file staging is platform-split. POSIX uses mkstemp, which creates the
// file atomically (O_CREAT|O_EXCL, mode 0600); std::filesystem has no
// equivalent, so the portable spelling would be an exists()-then-open probe
// loop that races. Keeping <filesystem> out of the ELF build also keeps
// libstdc++'s vague-linkage std::filesystem::path symbols out of this .so,
// where a differently-built module on the loader path (OpenVINO's libs) could
// interpose them. MSVC / clang-cl emit COFF and have no such interposition, so
// Windows keeps std::filesystem.
#ifdef LITERT_WINDOWS_OS
#include <cstdio>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <random>
#include <system_error>
#else
#include <errno.h>
#include <stdlib.h>  // mkstemp: POSIX, not declared by <cstdlib>.
#include <unistd.h>
#endif

namespace litert::openvino {
namespace {

#ifndef LITERT_WINDOWS_OS
// mkstemp needs the directory spelled out in its template, so mirror what
// std::filesystem::temp_directory_path() probes. Android has no /tmp, so
// staging there requires TMPDIR to be set -- unchanged from the previous
// std::filesystem spelling, which failed the same way.
std::string PosixTempDir() {
  for (const char* var : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* value = std::getenv(var);
        value != nullptr && value[0] != '\0') {
      return value;
    }
  }
  return "/tmp";
}

// A single write(2) is capped at 0x7ffff000 bytes on Linux and can be cut
// short by a signal, so the multi-GB pool needs this loop: a short write is
// not an error. Returns 0 on success, else the failing errno.
int WriteAll(int fd, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  while (size > 0) {
    const ssize_t written = ::write(fd, p, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      return errno;
    }
    p += written;
    size -= static_cast<size_t>(written);
  }
  return 0;
}
#endif  // !LITERT_WINDOWS_OS

}  // namespace

litert::Expected<std::vector<BoundWeight>> GpuSharedBank::Bind(
    ov::Core& core, const OpenVinoGlobalGraph& global_graph,
    const ov::CompiledModel& compiled_model,
    const std::map<std::string, uint32_t>& const_map) {
  ov::RemoteContext ctx = core.get_default_context("GPU");

  // Hold the lock across the whole body: the first partition allocates+fills
  // the pool, later partitions read base_/offset_. Serializes concurrent
  // partitions driving the same model (this bank is owned per device context /
  // per model).
  absl::MutexLock lock(gpu_bank_mutex_);
  if (!bank_ready_) {
    size_t total = 0;
    for (const auto& entry : global_graph.buffers) total += entry.bytes.size();
    // Allocate the pool as one usm-host buffer of bytes; per-weight views
    // reinterpret slices as the weight's element type below.
    gpu_usm_ =
        ctx.create_tensor(ov::element::u8, ov::Shape{total},
                          {{ov::intel_gpu::shared_mem_type.name(),
                            ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
    base_ =
        gpu_usm_.get_params().at(ov::intel_gpu::mem_handle.name()).as<void*>();
    auto* base = static_cast<uint8_t*>(base_);
    size_t off = 0;
    // TODO(PR #8745 #7): recheck USM view alignment.
    for (const auto& entry : global_graph.buffers) {  // ascending buffer_id
      std::memcpy(base + off, entry.bytes.data(), entry.bytes.size());
      weight_offset_[entry.id] = off;
      off += entry.bytes.size();
    }
    bank_ready_ = true;
    LITERT_LOG(LITERT_INFO,
               "GlobalGraph: allocated shared usm-host bank (%zu bytes)",
               total);
  }

  auto* base = static_cast<uint8_t*>(base_);
  std::vector<BoundWeight> bound;
  const auto& inputs = compiled_model.inputs();
  for (size_t p = 0; p < inputs.size(); ++p) {
    // Match by the input's friendly_name (set at compile when the weight was
    // promoted to a Parameter), not by port index -- import_model may reorder
    // inputs, so positional keying is unsafe.
    const auto it = const_map.find(inputs[p].get_node()->get_friendly_name());
    if (it == const_map.end()) {
      continue;  // real activation input, not a shared weight
    }
    const auto off_it = weight_offset_.find(it->second);
    if (off_it == weight_offset_.end()) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "GlobalGraph: const_map buffer_id not in pool");
    }
    // Wrap the slice of the shared usm-host buffer as a remote tensor view of
    // the weight's own element type (zero-copy: same USM pointer, no reorder).
    ov::RemoteTensor view =
        ctx.create_tensor(inputs[p].get_element_type(), inputs[p].get_shape(),
                          {{ov::intel_gpu::shared_mem_type.name(),
                            ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
                           {ov::intel_gpu::mem_handle.name(),
                            static_cast<ov::intel_gpu::gpu_handle_param>(
                                base + off_it->second)}});
    bound.push_back(BoundWeight{p, std::move(view)});
  }
  return bound;
}

NpuSharedBank::~NpuSharedBank() {
  // The device context outlives every compiled model imported from this bank,
  // so NPUW's mmap of the file is already gone by the time this runs.
  std::string path;
  {
    absl::MutexLock lock(npu_bank_mutex_);
    if (!owns_bank_file_) return;
    path = bank_path_;
  }
  if (path.empty()) return;
#ifdef LITERT_WINDOWS_OS
  std::error_code ec;
  std::filesystem::remove(std::filesystem::path(path), ec);
#else
  ::unlink(path.c_str());
#endif
}

std::string NpuSharedBank::Path() const {
  absl::MutexLock lock(npu_bank_mutex_);
  return bank_path_;
}

std::string NpuSharedBank::EnsureOnDisk(const void* data, size_t size) {
  absl::MutexLock lock(npu_bank_mutex_);
  if (!bank_path_.empty()) return bank_path_;  // write once, reuse

  // Guard against a null/empty pool: writing 0 bytes yields an empty file that
  // NPUW would mmap to a zero-length region, so data()+bin_offset is undefined.
  if (data == nullptr || size == 0) {
    LITERT_LOG(LITERT_ERROR,
               "EnsureOnDisk: refusing to stage null/empty pool "
               "(data=%p size=%zu)",
               data, size);
    return {};
  }

  // Deployment override: point at a pre-staged bank and skip the write.
  if (const char* override_path = std::getenv("LITERT_OV_WEIGHTS_PATH");
      override_path != nullptr && override_path[0] != '\0') {
    bank_path_ = override_path;
    LITERT_LOG(LITERT_INFO,
               "NPU weight sharing: using LITERT_OV_WEIGHTS_PATH override '%s'",
               bank_path_.c_str());
    return bank_path_;
  }

#ifdef LITERT_WINDOWS_OS
  std::error_code ec;
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path(ec);
  if (ec) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: no temp directory available: %s",
               ec.message().c_str());
    return {};
  }

  // No mkstemp on Windows; probe random names in the system temp dir.
  std::random_device rd;
  std::filesystem::path file_path;
  bool found = false;
  for (int attempt = 0; attempt < 64; ++attempt) {
    const uint64_t r = (static_cast<uint64_t>(rd()) << 32) ^
                       static_cast<uint64_t>(rd());
    char name[40];
    std::snprintf(name, sizeof(name), "litert_ov_bank_%016llx.bin",
                  static_cast<unsigned long long>(r));
    file_path = tmp_dir / name;
    if (!std::filesystem::exists(file_path, ec)) {
      found = true;
      break;
    }
  }
  if (!found) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: could not find a free temp file name");
    return {};
  }

  std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to open temp file '%s'",
               file_path.string().c_str());
    return {};
  }
  out.write(static_cast<const char*>(data),
            static_cast<std::streamsize>(size));
  out.close();
  if (!out) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to write temp file '%s'",
               file_path.string().c_str());
    std::filesystem::remove(file_path, ec);
    return {};
  }

  bank_path_ = file_path.string();
#else
  std::string name_template = PosixTempDir();
  if (name_template.empty() || name_template.back() != '/') {
    name_template.push_back('/');
  }
  name_template += "litert_ov_bank_XXXXXX";
  // mkstemp rewrites the XXXXXX in place and creates the file atomically, so
  // two processes can never end up staging into the same bank, and mode 0600
  // keeps the pool unreadable by other users of a shared temp dir.
  std::vector<char> path_buf(name_template.begin(), name_template.end());
  path_buf.push_back('\0');
  const int fd = ::mkstemp(path_buf.data());
  if (fd < 0) {
    LITERT_LOG(LITERT_ERROR, "NPU weight sharing: mkstemp('%s') failed: %s",
               name_template.c_str(), std::strerror(errno));
    return {};
  }
  const std::string file_path(path_buf.data());

  const int write_err = WriteAll(fd, data, size);
  // Close even when the write failed, and treat a close error as a write
  // error: ENOSPC / EIO can surface only here.
  const int close_err = (::close(fd) == 0) ? 0 : errno;
  if (write_err != 0 || close_err != 0) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to write temp file '%s': %s",
               file_path.c_str(),
               std::strerror(write_err != 0 ? write_err : close_err));
    ::unlink(file_path.c_str());
    return {};
  }

  bank_path_ = file_path;
#endif  // LITERT_WINDOWS_OS
  owns_bank_file_ = true;
  LITERT_LOG(LITERT_INFO,
             "NPU weight sharing: staged %zu-byte weights bank to '%s'", size,
             bank_path_.c_str());
  return bank_path_;
}

}  // namespace litert::openvino
