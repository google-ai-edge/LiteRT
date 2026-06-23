// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_UTILS_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {

struct PowerConfig {
  static constexpr uint32_t kSleepMinLatency = 40;
  static constexpr uint32_t kSleepLowLatency = 100;
  static constexpr uint32_t kSleepMediumLatency = 1000;
  static constexpr uint32_t kSleepHighLatency = 2000;
  static constexpr uint32_t kSleepMaxLatency = 65535;
  static constexpr uint32_t kDcvsDisable = 0;
  static constexpr uint32_t kDcvsEnable = 1;
  // default rpc control latency - 0 us
  static constexpr uint32_t kRpcControlLatency = 0;
  // default rpc polling time for high power modes - 9999 us
  static constexpr uint32_t kRpcPollingTimeHighPower = 9999;
};

template <typename T, std::size_t N>
void SetNullTermPtrArray(absl::Span<const T> src,
                         std::array<const T*, N>& dst) {
  size_t min_size = std::min(src.size(), dst.size() - 1);
  for (std::size_t i = 0; i < min_size; ++i) {
    dst[i] = &src[i];
  }
  dst[min_size] = nullptr;
}

inline std::optional<SocInfo> FindSocInfo(const SnapdragonModel& soc_model) {
  for (auto i = 0; i < kNumSocInfos; ++i) {
    if (soc_model == kSocInfos[i].soc_model) {
      return kSocInfos[i];
    }
  }
  return std::nullopt;
}

// Serialises async upvote/downvote on a background thread, so Execute() never
// blocks on the slow FastRPC setPowerConfig() call.
inline constexpr std::chrono::milliseconds kDownVoteDelayMs{300};

class VotingThread {
 public:
  enum class VoteType { kUpVote, kDownVote };

  explicit VotingThread(std::function<void(VoteType)> apply_vote)
      : apply_vote_(std::move(apply_vote)),
        thread_(&VotingThread::WorkerLoop, this) {}

  ~VotingThread() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

  VotingThread(const VotingThread&) = delete;
  VotingThread& operator=(const VotingThread&) = delete;
  VotingThread(VotingThread&&) = delete;
  VotingThread& operator=(VotingThread&&) = delete;

  void Enqueue(VoteType vote, bool debounce = false) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push({vote, debounce});
    }
    cv_.notify_one();
  }

 private:
  struct VoteInfo {
    VoteType vote;
    bool debounce;
  };

  void WorkerLoop() {
    while (true) {
      VoteType vote;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
        if (stop_ && queue_.empty()) break;

        auto [v, debounce] = queue_.front();
        queue_.pop();

        // Debounce a downvote: if a new message arrives within the window,
        // drop this downvote and restart the loop.
        if (v == VoteType::kDownVote && debounce) {
          if (cv_.wait_for(lock, kDownVoteDelayMs,
                           [this] { return !queue_.empty() || stop_; })) {
            continue;
          }
        }
        vote = v;
      }  // release lock before the slow apply_vote_ (FastRPC) call
      apply_vote_(vote);
    }
  }

  std::function<void(VoteType)> apply_vote_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<VoteInfo> queue_;
  bool stop_{false};
  std::thread thread_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_UTILS_H_
