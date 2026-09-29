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
 *
 * @file    nnsdk_func.h
 * @module  aml_runtime
 * @brief   dlopen wrappers for libnnsdk.so (NNSDK2 / amlnn_* APIs) used by dispatch.
 * @note    Runtime only; not part of aml_compiler_core. NNSDK1 (module_*) removed.
 ******************************************************************************/

#ifndef NNSDK_FUNC_H__
#define NNSDK_FUNC_H__

#include <dlfcn.h>

#include <cstdint>

#include "nnsdk2.h"

namespace tflite {

typedef int (*amlnn_init_func)(void** context, void* model, uint32_t size,
                               amlnn_init_config* init_config);
typedef int (*amlnn_destroy_func)(void* context);
typedef int (*amlnn_query_func)(void* context, amlnn_query_cmd query_cmd,
                                void* info, uint32_t size);
typedef int (*amlnn_inputs_set_func)(void* context, uint32_t n_inputs,
                                     amlnn_input inputs[]);
typedef int (*amlnn_run_func)(void* context, amlnn_run_config* run_config);
typedef int (*amlnn_outputs_get_func)(void* context, uint32_t n_outputs,
                                      amlnn_output outputs[]);

/**
 * @brief Optional multi-subgraph selector (amlnn_select_subgraph).
 * @note Subgraph count is queried via amlnn_query(AMLNN_QUERY_SUBGRAPH_NUM).
 */
typedef int (*amlnn_select_subgraph_func)(void* context,
                                          uint32_t subgraph_index);

/**
 * @brief Function-pointer table for NNSDK2 (amlnn_* APIs).
 * @note select_subgraph is optional (dlsym); older SDKs may omit it.
 */
typedef struct _nnsdk2_func_api {
  amlnn_init_func init;
  amlnn_destroy_func destroy;
  amlnn_query_func query;
  amlnn_inputs_set_func inputs_set;
  amlnn_run_func run;
  amlnn_outputs_get_func outputs_get;
  amlnn_select_subgraph_func select_subgraph;
} NNSDK2_FUNC_PTR;

/**
 * @brief Simple NHWC tensor descriptor used by some host helpers.
 */
typedef struct {
  uint32_t batches;
  uint32_t height;
  uint32_t width;
  uint32_t depth;
  uint8_t* data;
  uint32_t dataLen;
  uint32_t data_valid_len;
} aml_nn_tensordef;

/**
 * @brief Shared NNSDK2 runtime state for LiteRT AML dispatch.
 * @var qcontext         Active NNSDK2 context handle.
 * @var nnsdk2_func_ptr  Resolved NNSDK2 API symbols.
 * @var available        True if NNSDK2 base symbols loaded.
 */
struct AML_NN {
  void* qcontext = nullptr;
  NNSDK2_FUNC_PTR nnsdk2_func_ptr = {};
  bool available = false;
};

/**
 * @brief Return the process-wide AML_NN singleton (dlopen once).
 */
AML_NN* AML_NNImplementation();

/**
 * @brief True if NNSDK2 symbols were successfully resolved.
 */
bool IsNnsdk2Available(const AML_NN* aml_nn);

/**
 * @brief Query subgraph count via AMLNN_QUERY_SUBGRAPH_NUM.
 * @param[out] subgraph_num Receives n_subgraph; set to 1 if query unsupported.
 * @return 0 on success (including fallback to 1); <0 on hard failure.
 */
int Nnsdk2QuerySubgraphNum(const AML_NN* aml_nn, void* context,
                           int* subgraph_num);

/**
 * @brief Select active subgraph when multi-subgraph APIs are present.
 * @param subgraph_idx Partition index from AML_Dispatch_Info.
 * @param subgraph_num Queried count from Nnsdk2QuerySubgraphNum().
 * @return 0 on success / not needed; <0 on select failure.
 * @note Matches NNSDK sample: select then query IO / run.
 */
int Nnsdk2SelectSubgraphIfNeeded(const AML_NN* aml_nn, void* context,
                                 int subgraph_idx, int subgraph_num);

}  // namespace tflite
#endif  // NNSDK_FUNC_H__
