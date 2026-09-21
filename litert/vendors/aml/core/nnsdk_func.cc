/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "nnsdk_func.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "adla_log.h"

namespace tflite {
namespace {

static void* handle = nullptr;
static int adla_first_loading = 0;
static NNSDK2_FUNC_PTR nnsdk2_func = {};
static bool nnsdk2_available = false;

void* LoadOptionalSymbol(const char* name) {
  dlerror();
  return dlsym(handle, name);
}

bool LoadNnsdk2Symbols() {
  memset(&nnsdk2_func, 0, sizeof(NNSDK2_FUNC_PTR));

  void* init_sym = LoadOptionalSymbol("amlnn_init");
  if (init_sym == nullptr) {
    ADLA_LOGE("NNSDK2 amlnn_init not found in libnnsdk.so");
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
    ADLA_LOGE("NNSDK2 symbols incomplete");
    memset(&nnsdk2_func, 0, sizeof(NNSDK2_FUNC_PTR));
    return false;
  }

  // Multi-subgraph API (optional on older SDKs).
  // Count: amlnn_query(AMLNN_QUERY_SUBGRAPH_NUM).
  // Select: amlnn_select_subgraph(context, index).
  nnsdk2_func.select_subgraph = reinterpret_cast<amlnn_select_subgraph_func>(
      LoadOptionalSymbol("amlnn_select_subgraph"));

  ADLA_LOGI("NNSDK2 symbols loaded (select_subgraph=%p)",
            reinterpret_cast<void*>(nnsdk2_func.select_subgraph));
  return true;
}

AML_NN load_nnsdk_func() {
  AML_NN aml_nn;
  memset(&aml_nn, 0, sizeof(AML_NN));
  if (adla_first_loading == 0) {
    dlerror();
    handle = dlopen("libnnsdk.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
      const char* err = dlerror();
      std::cout << "open libnnsdk failed, Error log = "
                << (err != nullptr ? err : "unknown") << std::endl;
      ADLA_LOGE("open libnnsdk failed");
      return {};
    }
    nnsdk2_available = LoadNnsdk2Symbols();
    if (!nnsdk2_available) {
      ADLA_LOGE("NNSDK2 required but symbols not available");
    }
    adla_first_loading = 1;
  }

  aml_nn.nnsdk2_func_ptr = nnsdk2_func;
  aml_nn.available = nnsdk2_available;
  return aml_nn;
}

}  // namespace

AML_NN* AML_NNImplementation() {
  static AML_NN aml_nn = load_nnsdk_func();
  return &aml_nn;
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
    ADLA_LOGI("[AML Dispatch] AMLNN_QUERY_SUBGRAPH_NUM unsupported ret=%d; "
              "assume n_subgraph=1",
              ret);
    *subgraph_num = 1;
    return 0;
  }
  if (info.n_subgraph == 0) {
    ADLA_LOGE("[AML Dispatch] AMLNN_QUERY_SUBGRAPH_NUM returned 0");
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
    ADLA_LOGE("[AML Dispatch] subgraph_idx=%d out of range (n_subgraph=%d)",
              subgraph_idx, subgraph_num);
    return -1;
  }
  if (aml_nn->nnsdk2_func_ptr.select_subgraph == nullptr) {
    ADLA_LOGE("[AML Dispatch] multi-subgraph model needs amlnn_select_subgraph "
              "(idx=%d, n=%d)",
              subgraph_idx, subgraph_num);
    return -1;
  }
  const int ret = aml_nn->nnsdk2_func_ptr.select_subgraph(
      context, static_cast<uint32_t>(subgraph_idx));
  if (ret != AMLNN_SUCCESS) {
    ADLA_LOGE("[AML Dispatch] amlnn_select_subgraph fail idx=%d ret=%d",
              subgraph_idx, ret);
    return ret;
  }
  return 0;
}

}  // namespace tflite
