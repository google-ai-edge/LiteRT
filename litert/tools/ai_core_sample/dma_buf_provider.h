// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_TOOLS_AI_CORE_SAMPLE_DMA_BUF_PROVIDER_H_
#define ODML_LITERT_LITERT_TOOLS_AI_CORE_SAMPLE_DMA_BUF_PROVIDER_H_

#include <cstdint>
#include <mutex>  // NOLINT
#include <string>
#include <utility>

#include "litert/c/litert_model_types.h"

namespace litert::tools {

// Simulates AICore when the whole model already lives in ONE caller-owned
// DMA-BUF. The model file is loaded into that DMA-BUF once (on the first
// request, so the staging offset can be chosen to make weights 4KB-aligned),
// and nothing is read or copied after that. Every request is answered from the
// same buffer with a fresh dup() of its fd: the HTP backend requires each
// dataProvider call to return a distinct fd and maps by fd, so a dup'd fd lets
// many sub-ranges of one buffer be served as-is.
//
// Request offsets are positions inside the model file. `model_path` must be
// the same file the model was created from.
class PreloadedDmaBufProvider {
 public:
  explicit PreloadedDmaBufProvider(std::string model_path)
      : model_path_(std::move(model_path)) {}
  ~PreloadedDmaBufProvider();

  PreloadedDmaBufProvider(const PreloadedDmaBufProvider&) = delete;
  PreloadedDmaBufProvider& operator=(const PreloadedDmaBufProvider&) = delete;

  LiteRtDmaContextBinaryCallback AsCallback();

 private:
  static LiteRtStatus Provide(LiteRtDmaContextDataRequest request,
                              LiteRtDmaContextDataResponse* response,
                              void* user_data);
  static LiteRtStatus Release(LiteRtDmaContextDataResponse response,
                              void* user_data);

  // Loads the model file into one DMA-BUF, placed `pad` bytes from the start.
  LiteRtStatus Stage(uint64_t pad);

  const std::string model_path_;
  std::mutex mutex_;
  int dmabuf_fd_ = -1;
  void* base_ = nullptr;
  uint64_t size_ = 0;      // DMA-BUF size
  uint64_t pad_ = 0;       // model starts this many bytes into the DMA-BUF
  uint64_t file_size_ = 0;
};

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_AI_CORE_SAMPLE_DMA_BUF_PROVIDER_H_
