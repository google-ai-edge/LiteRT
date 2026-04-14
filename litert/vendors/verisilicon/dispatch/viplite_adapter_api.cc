// Copyright 2025 Vivante Corporation.
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

#include "litert/vendors/verisilicon/dispatch/viplite_adapter_api.h"

#include <dlfcn.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_verisilicon_options.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

#define LOAD_SYMB(S, H)                                                   \
  if (auto maybe_H = dlib_.LookupSymbol<void*>(#S); maybe_H.HasValue()) { \
    H = reinterpret_cast<decltype(&S)>(std::move(maybe_H.Value()));       \
  } else {                                                                \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #S,        \
               dlib_.DlError());                                          \
    H = nullptr;                                                          \
  }

namespace litert {
namespace verisilicon {

VipliteAdapterApi::VipliteAdapterApi() : api_(new Api) {}

litert::Expected<VipliteAdapterApi::Ptr> VipliteAdapterApi::Create(
    std::optional<std::string> shared_library_dir) {
  std::unique_ptr<VipliteAdapterApi> viplite_adapter_api(new VipliteAdapterApi);

  if (auto status = viplite_adapter_api->LoadSymbols(shared_library_dir);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to load viplite shared library: %s",
               status.Error().Message().c_str());
    return status.Error();
  }

  LITERT_RETURN_IF_ERROR(viplite_adapter_api->GetVipliteVersion());

  return viplite_adapter_api;
}

litert::Expected<void> VipliteAdapterApi::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  constexpr auto kLibNBGlinkerLib = "libNBGlinker.so";
  std::vector<std::string> so_paths;

  // Add hal lib
  so_paths.push_back("libVIPhal.so");

  // Add the library from the provided shared lib directory if available.
  shared_library_dir.has_value()
      ? so_paths.push_back(
            absl::StrCat(*shared_library_dir, "/", kLibNBGlinkerLib))
      : so_paths.push_back(kLibNBGlinkerLib);

  for (auto& so_path : so_paths) {
    auto maybe_dlib = SharedLibrary::Load(so_path, RtldFlags::Default());
    if (maybe_dlib.HasValue()) {
      LITERT_LOG(LITERT_INFO,
                 "Loading Verisilicon VIPLite adapter .so from: %s",
                 so_path.c_str());
      dlib_ = std::move(maybe_dlib.Value());
    }
  }

  if (!dlib_.Loaded()) {
    return litert::Error(kLiteRtStatusErrorDynamicLoading,
                         "Failed to load VIPLite shared library");
  }

  LITERT_LOG(LITERT_INFO, "Loaded VIPLite shared library.");

  // Binds all supported symbols from the shared library to the function
  // pointers.
  LOAD_SYMB(vip_get_version, api_->get_version);
  LOAD_SYMB(vip_init, api_->init);
  LOAD_SYMB(vip_destroy, api_->deinit);
  LOAD_SYMB(vip_query_hardware, api_->query_hardware);
  LOAD_SYMB(vip_create_buffer, api_->create_buffer);
  LOAD_SYMB(vip_destroy_buffer, api_->destroy_buffer);
  LOAD_SYMB(vip_map_buffer, api_->map_buffer);
  LOAD_SYMB(vip_unmap_buffer, api_->unmap_buffer);
  LOAD_SYMB(vip_get_buffer_size, api_->get_buffer_size);
  LOAD_SYMB(vip_flush_buffer, api_->flush_buffer);
  LOAD_SYMB(vip_create_network, api_->create_network);
  LOAD_SYMB(vip_destroy_network, api_->destroy_network);
  LOAD_SYMB(vip_prepare_network, api_->prepare_network);
  LOAD_SYMB(vip_finish_network, api_->finish_network);
  LOAD_SYMB(vip_set_network, api_->set_network);
  LOAD_SYMB(vip_query_network, api_->query_network);
  LOAD_SYMB(vip_run_network, api_->run_network);
  LOAD_SYMB(vip_trigger_network, api_->trigger_network);
  LOAD_SYMB(vip_wait_network, api_->wait_network);
  LOAD_SYMB(vip_cancel_network, api_->cancel_network);
  LOAD_SYMB(vip_query_input, api_->query_input);
  LOAD_SYMB(vip_query_output, api_->query_output);
  LOAD_SYMB(vip_set_input, api_->set_input);
  LOAD_SYMB(vip_set_output, api_->set_output);
  LOAD_SYMB(vip_create_group, api_->create_group);
  LOAD_SYMB(vip_destroy_group, api_->destroy_group);
  LOAD_SYMB(vip_set_group, api_->set_group);
  LOAD_SYMB(vip_query_group, api_->query_group);
  LOAD_SYMB(vip_add_network, api_->add_network);
  LOAD_SYMB(vip_run_group, api_->run_group);
  LOAD_SYMB(vip_trigger_group, api_->trigger_group);
  LOAD_SYMB(vip_wait_group, api_->wait_group);
  LOAD_SYMB(vip_power_management, api_->power_management);

  LITERT_LOG(LITERT_INFO, "Viplite symbols loaded");
  return {};
}

litert::Expected<void> VipliteAdapterApi::GetVipliteVersion() {
  runtime_version_ = api().get_version();
  if (runtime_version_ < 0x20000) {
    LITERT_LOG(LITERT_ERROR, "Viplite version is too low");
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Viplite version is too low");
  }
  LITERT_LOG(LITERT_INFO, "Viplite version: %d.%d.%d",
             runtime_version_ >> 16 & 0xFF, runtime_version_ >> 8 & 0xFF,
             runtime_version_ & 0xFF);
  return {};
}

}  // namespace verisilicon
}  // namespace litert
