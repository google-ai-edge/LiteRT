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
 * @file    node_binary_info.h
 * @module  aml_compiler_plugin
 * @brief   Map LiteRT op options into C IR node params (TFLite-aligned values).
 ******************************************************************************/

#ifndef NODE_BINARY_INFO_H
#define NODE_BINARY_INFO_H

#include "compiler_core_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_op_options.h"

/**
 * @brief Parse LiteRT op attributes into @p node via AmlNodeSetParamI/F.
 * @param node     Handle from AmlNodeCreate (Node_type already set).
 * @param op_code  LiteRtOpCode (numeric TFLite BuiltinOperator).
 * @param c_op     LiteRT op handle for option queries.
 */
LiteRtStatus ParseOpParams(AmlNode node, int op_code, LiteRtOp c_op);

#endif  // NODE_BINARY_INFO_H
