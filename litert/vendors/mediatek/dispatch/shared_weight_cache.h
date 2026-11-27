// Copyright 2026 MediaTek Inc.
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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_SHARED_WEIGHT_CACHE_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_SHARED_WEIGHT_CACHE_H_

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <utility>

#include "litert/cc/litert_expected.h"

namespace litert::mediatek {

// Weak index of live objects built from a byte range of a mapped model,
// keyed by the range's address and size. Callers hold the strong references,
// so an object lives exactly as long as someone uses it, and destroying the
// cache never frees an object that is still in use.
template <typename T>
class SharedWeightCache {
 public:
  using Creator = std::function<litert::Expected<std::shared_ptr<T>>()>;

  // Returns the live object for [data, data + size), or the one `create`
  // makes. A failed `create` is not cached.
  litert::Expected<std::shared_ptr<T>> GetOrCreate(const void* data,
                                                   size_t size,
                                                   const Creator& create) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = index_[{data, size}];
    if (auto live = slot.lock()) {
      return live;
    }
    auto created = create();
    if (created) {
      slot = *created;
    }
    return created;
  }

 private:
  std::mutex mutex_;
  std::map<std::pair<const void*, size_t>, std::weak_ptr<T>> index_;
};

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_SHARED_WEIGHT_CACHE_H_
