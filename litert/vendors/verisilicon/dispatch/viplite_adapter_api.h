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

#ifndef ODML_LITERT_LITERT_VENDORS_VERISILICON_VIPLITE_ADAPTER_API_H_
#define ODML_LITERT_LITERT_VENDORS_VERISILICON_VIPLITE_ADAPTER_API_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/verisilicon/dispatch/vip_lite.h"

namespace litert::verisilicon {
constexpr const size_t kVipliteCacheLineAlignment = 64;
constexpr const size_t kVipliteAddressAlignment = 256;

using VipNetworkPtr = std::unique_ptr<vip_network, void (*)(vip_network*)>;

class VipliteAdapterApi {
 public:
  using Ptr = std::unique_ptr<VipliteAdapterApi>;
  struct Api;

  VipliteAdapterApi(VipliteAdapterApi&) = delete;
  VipliteAdapterApi(VipliteAdapterApi&&) = delete;
  VipliteAdapterApi& operator=(const VipliteAdapterApi&) = delete;
  VipliteAdapterApi& operator=(VipliteAdapterApi&&) = delete;

  static Expected<Ptr> Create(
      std::optional<std::string> shared_library_dir);

  const Api& api() const { return *api_; }

  litert::Expected<void> GetVipliteVersion();

 private:
  VipliteAdapterApi();
  litert::Expected<void> LoadSymbols(
      std::optional<std::string> shared_library_dir);

  // Handle to the shared library that implements the Neuron API.
  //
  // This will keep the shared library open until the NeuronAdapterApi object is
  // destroyed.
  SharedLibrary dlib_;
  std::unique_ptr<Api> api_;
  unsigned int runtime_version_;
};


// A convenient struct for holding function pointers to VIPLite API
// symbols. These function pointers will be loaded to the shared library on
// device during runtime.
struct VipliteAdapterApi::Api {
  decltype(&vip_get_version) get_version = nullptr;
  decltype(&vip_init) init = nullptr;
  decltype(&vip_destroy) deinit = nullptr;
  decltype(&vip_query_hardware) query_hardware = nullptr;
  decltype(&vip_create_buffer) create_buffer = nullptr;
  decltype(&vip_destroy_buffer)  destroy_buffer= nullptr;
  decltype(&vip_map_buffer) map_buffer = nullptr;
  decltype(&vip_unmap_buffer) unmap_buffer = nullptr;
  decltype(&vip_get_buffer_size) get_buffer_size = nullptr;
  decltype(&vip_flush_buffer) flush_buffer = nullptr;
  decltype(&vip_create_network) create_network = nullptr;
  decltype(&vip_destroy_network) destroy_network = nullptr;
  decltype(&vip_prepare_network) prepare_network = nullptr;
  decltype(&vip_finish_network) finish_network = nullptr;
  decltype(&vip_set_network) set_network = nullptr;
  decltype(&vip_query_network) query_network = nullptr;
  decltype(&vip_run_network) run_network = nullptr;
  decltype(&vip_trigger_network) trigger_network = nullptr;
  decltype(&vip_wait_network) wait_network = nullptr;
  decltype(&vip_cancel_network) cancel_network = nullptr;
  decltype(&vip_query_input) query_input = nullptr;
  decltype(&vip_query_output) query_output = nullptr;
  decltype(&vip_set_input) set_input = nullptr;
  decltype(&vip_set_output) set_output = nullptr;
  decltype(&vip_create_group) create_group = nullptr;
  decltype(&vip_destroy_group) destroy_group = nullptr;
  decltype(&vip_set_group) set_group = nullptr;
  decltype(&vip_query_group) query_group = nullptr;
  decltype(&vip_add_network) add_network = nullptr;
  decltype(&vip_run_group) run_group = nullptr;
  decltype(&vip_trigger_group) trigger_group = nullptr;
  decltype(&vip_wait_group) wait_group = nullptr;
  decltype(&vip_power_management) power_management = nullptr;

};

}  // namespace litert::verisilicon

#endif  // ODML_LITERT_LITERT_VENDORS_VERISILICON_VIPLITE_ADAPTER_API_H_
