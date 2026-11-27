// Copyright 2026 Amlogic, Inc.
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

#include "nnsdk_func.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "litert/c/internal/litert_logging.h"

namespace aml_nnsdk {
namespace {

static void* handle = nullptr;
static int nnsdk_ref_count = 0;
static NNSDK2_FUNC_PTR nnsdk2_func = {};
static bool nnsdk2_available = false;
static std::string nnsdk_library_dir;
static std::mutex nnsdk_load_mu;

void* LoadOptionalSymbol(const char* name) {
  dlerror();
  return dlsym(handle, name);
}

bool LoadNnsdk2Symbols() {
  memset(&nnsdk2_func, 0, sizeof(NNSDK2_FUNC_PTR));

  void* init_sym = LoadOptionalSymbol("amlnn_init");
  if (init_sym == nullptr) {
    LITERT_LOG(LITERT_ERROR, "NNSDK2 amlnn_init not found in libnnsdk.so");
    return false;
  }

  nnsdk2_func.init = reinterpret_cast<amlnn_init_func>(init_sym);
  nnsdk2_func.destroy =
      reinterpret_cast<amlnn_destroy_func>(LoadOptionalSymbol("amlnn_destroy"));
  nnsdk2_func.query =
      reinterpret_cast<amlnn_query_func>(LoadOptionalSymbol("amlnn_query"));
  nnsdk2_func.inputs_set = reinterpret_cast<amlnn_inputs_set_func>(
      LoadOptionalSymbol("amlnn_inputs_set"));
  nnsdk2_func.run =
      reinterpret_cast<amlnn_run_func>(LoadOptionalSymbol("amlnn_run"));
  nnsdk2_func.outputs_get = reinterpret_cast<amlnn_outputs_get_func>(
      LoadOptionalSymbol("amlnn_outputs_get"));

  if (nnsdk2_func.destroy == nullptr || nnsdk2_func.query == nullptr ||
      nnsdk2_func.inputs_set == nullptr || nnsdk2_func.run == nullptr ||
      nnsdk2_func.outputs_get == nullptr) {
    LITERT_LOG(LITERT_ERROR, "NNSDK2 symbols incomplete");
    memset(&nnsdk2_func, 0, sizeof(NNSDK2_FUNC_PTR));
    return false;
  }

  // Multi-subgraph API (optional on older SDKs).
  // Count: amlnn_query(AMLNN_QUERY_SUBGRAPH_NUM).
  // Select: amlnn_select_subgraph(context, index).
  nnsdk2_func.select_subgraph = reinterpret_cast<amlnn_select_subgraph_func>(
      LoadOptionalSymbol("amlnn_select_subgraph"));

  LITERT_LOG(LITERT_INFO, "NNSDK2 symbols loaded (select_subgraph=%p)",
            reinterpret_cast<void*>(nnsdk2_func.select_subgraph));
  return true;
}

AML_NN load_nnsdk_func() {
  std::lock_guard<std::mutex> lock(nnsdk_load_mu);
  AML_NN aml_nn;
  memset(&aml_nn, 0, sizeof(AML_NN));
  if (handle == nullptr) {
    std::string nnsdk_path = "libnnsdk.so";
    if (!nnsdk_library_dir.empty()) {
      nnsdk_path = nnsdk_library_dir;
      if (nnsdk_path.back() != '/') {
        nnsdk_path.push_back('/');
      }
      nnsdk_path += "libnnsdk.so";
    }
    dlerror();
    handle = dlopen(nnsdk_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
      const char* err = dlerror();
      LITERT_LOG(LITERT_ERROR, "open libnnsdk failed path=%s err=%s", nnsdk_path.c_str(),
                err != nullptr ? err : "unknown");
      return {};
    }
    nnsdk2_available = LoadNnsdk2Symbols();
    if (!nnsdk2_available) {
      LITERT_LOG(LITERT_ERROR, "NNSDK2 required but symbols not available");
      dlclose(handle);
      handle = nullptr;
      return {};
    }
  }

  ++nnsdk_ref_count;
  aml_nn.nnsdk2_func_ptr = nnsdk2_func;
  aml_nn.available = nnsdk2_available;
  return aml_nn;
}

void release_nnsdk_locked() {
  if (nnsdk_ref_count <= 0) {
    return;
  }
  --nnsdk_ref_count;
  if (nnsdk_ref_count > 0 || handle == nullptr) {
    return;
  }
  dlclose(handle);
  handle = nullptr;
  memset(&nnsdk2_func, 0, sizeof(nnsdk2_func));
  nnsdk2_available = false;
  LITERT_LOG(LITERT_INFO, "dlclose libnnsdk.so");
}

}  // namespace

AML_NN* AML_NNImplementation() {
  // Does not dlopen. Callers that need symbols use CreateAmlNn/ReleaseNnsdk.
  static AML_NN aml_nn;
  return &aml_nn;
}

AML_NN CreateAmlNn() {
  AML_NN aml_nn = load_nnsdk_func();
  aml_nn.qcontext = nullptr;
  return aml_nn;
}

void ReleaseNnsdk() {
  std::lock_guard<std::mutex> lock(nnsdk_load_mu);
  release_nnsdk_locked();
}

void SetNnsdkLibraryDir(const char* dir) {
  std::lock_guard<std::mutex> lock(nnsdk_load_mu);
  if (handle != nullptr) {
    LITERT_LOG(LITERT_WARNING,
               "libnnsdk.so is already loaded; DispatchLibraryDir is ignored");
    return;
  }
  nnsdk_library_dir = (dir != nullptr) ? dir : "";
}

bool IsNnsdk2Available(const AML_NN* aml_nn) {
  return aml_nn != nullptr && aml_nn->available;
}

int Nnsdk2QuerySubgraphNum(const AML_NN* aml_nn, void* context,
                           int* subgraph_num) {
  if (subgraph_num == nullptr) {
    return -1;
  }
  *subgraph_num = 1;
  if (aml_nn == nullptr || context == nullptr ||
      aml_nn->nnsdk2_func_ptr.query == nullptr) {
    return 0;
  }

  amlnn_subgraph_num info;
  memset(&info, 0, sizeof(info));
  const int ret = aml_nn->nnsdk2_func_ptr.query(
      context, AMLNN_QUERY_SUBGRAPH_NUM, &info, sizeof(info));
  if (ret != AMLNN_SUCCESS) {
    // Older SDK / single-subgraph model: treat as one subgraph.
    LITERT_LOG(LITERT_INFO, "[AML Dispatch] AMLNN_QUERY_SUBGRAPH_NUM unsupported ret=%d; "
              "assume n_subgraph=1",
              ret);
    *subgraph_num = 1;
    return 0;
  }
  if (info.n_subgraph == 0) {
    LITERT_LOG(LITERT_ERROR, "[AML Dispatch] AMLNN_QUERY_SUBGRAPH_NUM returned 0");
    return -1;
  }
  *subgraph_num = static_cast<int>(info.n_subgraph);
  return 0;
}

int Nnsdk2SelectSubgraphIfNeeded(const AML_NN* aml_nn, void* context,
                                 int subgraph_idx, int subgraph_num) {
  if (aml_nn == nullptr || context == nullptr) {
    return -1;
  }
  if (subgraph_idx < 0) {
    return -1;
  }
  // Single-subgraph model: nothing to select.
  if (subgraph_num <= 1) {
    return 0;
  }
  if (subgraph_idx >= subgraph_num) {
    LITERT_LOG(LITERT_ERROR, "[AML Dispatch] subgraph_idx=%d out of range (n_subgraph=%d)",
              subgraph_idx, subgraph_num);
    return -1;
  }
  if (aml_nn->nnsdk2_func_ptr.select_subgraph == nullptr) {
    LITERT_LOG(LITERT_ERROR, "[AML Dispatch] multi-subgraph model needs amlnn_select_subgraph "
              "(idx=%d, n=%d)",
              subgraph_idx, subgraph_num);
    return -1;
  }
  const int ret = aml_nn->nnsdk2_func_ptr.select_subgraph(
      context, static_cast<uint32_t>(subgraph_idx));
  if (ret != AMLNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "[AML Dispatch] amlnn_select_subgraph fail idx=%d ret=%d",
              subgraph_idx, ret);
    return ret;
  }
  return 0;
}

}  // namespace aml_nnsdk
