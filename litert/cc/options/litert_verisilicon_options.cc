// Copyright 2025 Google LLC.
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
#include "litert/cc/options/litert_verisilicon_options.h"

#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_verisilicon_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////
namespace litert::verisilicon {

Expected<VerisiliconOptions> VerisiliconOptions::Create() {
  LrtVerisiliconOptions options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtCreateVerisiliconOptions(&options));
  return VerisiliconOptions(options);
}

Expected<void> VerisiliconOptions::SetDeviceIndex(unsigned int device_index) {
  internal::AssertOk(LrtVerisiliconOptionsSetDeviceIndex, Get(), device_index);
  return {};
}

Expected<unsigned int> VerisiliconOptions::GetDeviceIndex() const {
  unsigned int device_index = 0;
  internal::AssertOk(LrtVerisiliconOptionsGetDeviceIndex, Get(), &device_index);
  return device_index;
}

Expected<void> VerisiliconOptions::SetCoreIndex(unsigned int core_index) {
  internal::AssertOk(LrtVerisiliconOptionsSetCoreIndex, Get(), core_index);
  return {};
}

Expected<unsigned int> VerisiliconOptions::GetCoreIndex() const {
  unsigned int core_index = 0;
  internal::AssertOk(LrtVerisiliconOptionsGetCoreIndex, Get(), &core_index);
  return core_index;
}

Expected<void> VerisiliconOptions::SetTimeOut(unsigned int time_out) {
  internal::AssertOk(LrtVerisiliconOptionsSetTimeOut, Get(), time_out);
  return {};
}

Expected<unsigned int> VerisiliconOptions::GetTimeOut() const {
  unsigned int time_out = 0;
  ;
  internal::AssertOk(LrtVerisiliconOptionsGetTimeOut, Get(), &time_out);
  return time_out;
}

Expected<void> VerisiliconOptions::SetProfileLevel(unsigned int level) {
  internal::AssertOk(LrtVerisiliconOptionsSetProfileLevel, Get(), level);
  return {};
}

Expected<unsigned int> VerisiliconOptions::GetProfileLevel() const {
  unsigned int level = 0;
  ;
  internal::AssertOk(LrtVerisiliconOptionsGetProfileLevel, Get(), &level);
  return level;
}

Expected<void> VerisiliconOptions::SetDumpNBG(bool enable) {
  internal::AssertOk(LrtVerisiliconOptionsSetDumpNBG, Get(), enable);
  return {};
}

Expected<bool> VerisiliconOptions::GetDumpNBG() const {
  bool enable = 0;
  ;
  internal::AssertOk(LrtVerisiliconOptionsGetDumpNBG, Get(), &enable);
  return enable;
}
}  // namespace litert::verisilicon
