// Copyright (c) 2025 MediaTek Inc.
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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_EXTRA_DATA_MGR_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_EXTRA_DATA_MGR_H_

#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"

namespace litert::mediatek {

class ExtraDataMgr {
 public:
  Expected<size_t> Register(size_t bytes) {
    if (bytes <= 0) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Bytes must be greater than 0");
    }
    extra_data_vec_.emplace_back(std::make_unique<uint8_t[]>(bytes));
    return extra_data_vec_.size() - 1;
  }

  uint8_t* Get(size_t index) {
    if (index < 0 || index >= extra_data_vec_.size()) {
      LITERT_LOG(LITERT_ERROR, "Index out of bound.");
      return nullptr;
    }
    return extra_data_vec_[index].get();
  }

 private:
  std::vector<std::unique_ptr<uint8_t[]>> extra_data_vec_;
};

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_EXTRA_DATA_MGR_H_
