// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_LITERT_MODEL_H_
#define ODML_LITERT_LITERT_C_LITERT_MODEL_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// LiteRtTensor + Types
//

// Get the string name associated with this tensor. This is an optional
// attribute and if not set will return a zero-length string.
// The returned string pointer is owned by the LiteRtModel to which the given
// Tensor belongs. It becomes invalid when the LiteRtModel is destroyed.
LiteRtStatus LiteRtGetTensorName(LiteRtTensor tensor, const char** name);

// Get the index associated with this tensor.
LiteRtStatus LiteRtGetTensorIndex(LiteRtTensor tensor, uint32_t* tensor_index);

// Get type identifier from tensor.
LiteRtStatus LiteRtGetTensorTypeId(LiteRtTensor tensor,
                                   LiteRtTensorTypeId* type_id);

// Get unranked tensor type info, return bad status if not unranked.
LiteRtStatus LiteRtGetUnrankedTensorType(
    LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type);

// Get ranked tensor type info, return bad status if not ranked.
LiteRtStatus LiteRtGetRankedTensorType(
    LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type);

// QUANTIZATION

// Get the identifier for the type of quantization for a given tensor.
LiteRtStatus LiteRtGetQuantizationTypeId(LiteRtTensor tensor,
                                         LiteRtQuantizationTypeId* q_type_id);

// Get the per-tensor quantization information for a given tensor if it has it.
LiteRtStatus LiteRtGetPerTensorQuantization(
    LiteRtTensor tensor, LiteRtQuantizationPerTensor* per_tensor_quantization);

// Get the per-channel quantization information for a given tensor if it has it.
LiteRtStatus LiteRtGetPerChannelQuantization(
    LiteRtTensor tensor,
    LiteRtQuantizationPerChannel* per_channel_quantization);

// EDGES

// Get all the ops that reference given tensor, and at what operand index.
LiteRtStatus LiteRtGetNumTensorUses(LiteRtTensor tensor,
                                    LiteRtParamIndex* num_uses);
LiteRtStatus LiteRtGetTensorUse(LiteRtTensor tensor, LiteRtParamIndex use_index,
                                LiteRtOp* user,
                                LiteRtParamIndex* user_arg_index);

// Get the op that defines this tensor and the corresponding output index. If
// tensor is a subgraph input, has_defining_op will be false.
LiteRtStatus LiteRtGetTensorDefiningOp(LiteRtTensor tensor,
                                       bool* has_defining_op,
                                       LiteRtTensorDefiningOp* defining_op);

// WEIGHTS (constant data)

// Get static weights associated with a given tensor. All tensors have weights,
// null weights have size = 0;
//
// Note: The returned LiteRtWeights is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetTensorWeights(LiteRtTensor tensor,
                                    LiteRtWeights* weights);

//
// LiteRtWeights
//

// Get opaque array from given tensor weights.
LiteRtStatus LiteRtGetWeightsBytes(LiteRtWeights weights, const void** addr,
                                   size_t* size);

// Get the buffer id associated with the weights. Buffer id managed internally
// by the buffer manager. Buffer id starts from 1.
LiteRtStatus LiteRtGetWeightsBufferId(LiteRtWeights weights,
                                      int32_t* buffer_id);
//
// LiteRtOp
//

// Get code corresponding to operation type for given op.
LiteRtStatus LiteRtGetOpCode(LiteRtOp op, LiteRtOpCode* code);

// Get custom code for given op, returns error if op is not a custom op.
LiteRtStatus LiteRtGetCustomCode(LiteRtOp op, const char** code);

// Get input tensors of given op.
LiteRtStatus LiteRtGetNumOpInputs(LiteRtOp op, LiteRtParamIndex* num_inputs);
LiteRtStatus LiteRtGetOpInput(LiteRtOp op, LiteRtParamIndex input_index,
                              LiteRtTensor* input);

// Get output tensors of given op.
LiteRtStatus LiteRtGetNumOpOutputs(LiteRtOp op, LiteRtParamIndex* num_outputs);
LiteRtStatus LiteRtGetOpOutput(LiteRtOp op, LiteRtParamIndex output_index,
                               LiteRtTensor* output);

//
// LiteRtSubgraph
//

// Get input tensors for given subgraph.
//
// Note: The returned LiteRtTensor is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetNumSubgraphInputs(LiteRtSubgraph subgraph,
                                        LiteRtParamIndex* num_inputs);
LiteRtStatus LiteRtGetSubgraphInput(LiteRtSubgraph subgraph,
                                    LiteRtParamIndex input_index,
                                    LiteRtTensor* input);

// Get output tensors for given subgraph.
//
// Note: The returned LiteRtTensor is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetNumSubgraphOutputs(LiteRtSubgraph subgraph,
                                         LiteRtParamIndex* num_outputs);
LiteRtStatus LiteRtGetSubgraphOutput(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex output_index,
                                     LiteRtTensor* output);

// Get all ops in given subgraph in a topological order.
LiteRtStatus LiteRtGetNumSubgraphOps(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex* num_ops);
LiteRtStatus LiteRtGetSubgraphOp(LiteRtSubgraph subgraph,
                                 LiteRtParamIndex op_index, LiteRtOp* op);

//
// LiteRtSignature
//

// Default signature key. This is the key that is used if models don't
// define any signatures.
LiteRtStatus LiteRtGetDefaultSignatureKey(const char** signature_key);

// Get the signature key string defined in the model.
// The returned string pointer is owned by the LiteRtModel to which the given
// Signature belongs. It becomes invalid when the LiteRtModel is destroyed.
LiteRtStatus LiteRtGetSignatureKey(LiteRtSignature signature,
                                   const char** signature_key);

// Get the associated subgraph for the given signature.
LiteRtStatus LiteRtGetSignatureSubgraph(LiteRtSignature signature,
                                        LiteRtSubgraph* subgraph);

// Get the number of inputs for the given signature.
LiteRtStatus LiteRtGetNumSignatureInputs(LiteRtSignature signature,
                                         LiteRtParamIndex* num_inputs);

// Get the name of the i-th of input tensor name for the given signature.
// The returned string pointer is owned by the LiteRtModel to which the given
// Signature belongs. It becomes invalid when the LiteRtModel is destroyed.
LiteRtStatus LiteRtGetSignatureInputName(LiteRtSignature signature,
                                         LiteRtParamIndex input_idx,
                                         const char** input_name);

// Get the input tensor for the given signature and input name.
//
// Note: The returned LiteRtTensor is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetSignatureInputTensor(LiteRtSignature signature,
                                           const char* input_name,
                                           LiteRtTensor* tensor);

// Get the input tensor for the given signature and input index.
LiteRtStatus LiteRtGetSignatureInputTensorByIndex(LiteRtSignature signature,
                                                  LiteRtParamIndex input_idx,
                                                  LiteRtTensor* tensor);

// Get the number of outputs for the given signature.
LiteRtStatus LiteRtGetNumSignatureOutputs(LiteRtSignature signature,
                                          LiteRtParamIndex* num_outputs);

// Get the name of the i-th of output tensor name for the given signature.
// The returned string pointer is owned by the LiteRtModel to which the given
// Signature belongs. It becomes invalid when the LiteRtModel is destroyed.
LiteRtStatus LiteRtGetSignatureOutputName(LiteRtSignature signature,
                                          LiteRtParamIndex output_idx,
                                          const char** output_name);

// Get the output tensor for the given signature and output name.
//
// Note: The returned LiteRtTensor is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetSignatureOutputTensor(LiteRtSignature signature,
                                            const char* output_name,
                                            LiteRtTensor* tensor);

// Get the output tensor for the given signature and output index.
LiteRtStatus LiteRtGetSignatureOutputTensorByIndex(LiteRtSignature signature,
                                                   LiteRtParamIndex output_idx,
                                                   LiteRtTensor* tensor);

//
// LiteRtModel
//

LiteRtStatus LiteRtCreateModelFromFile(const char* filename,
                                       LiteRtModel* model);
// The caller must ensure that the buffer remains valid for the lifetime of
// the model.
LiteRtStatus LiteRtCreateModelFromBuffer(const void* buffer_addr,
                                         size_t buffer_size,
                                         LiteRtModel* model);

// Get the metadata buffer associated with given key if it exists.
LiteRtStatus LiteRtGetModelMetadata(LiteRtModel model, const char* metadata_key,
                                    const void** metadata_buffer,
                                    size_t* metadata_buffer_size);

// Add metadata to the model. LiteRtModel copies key and metadata internally. It
// doesn't take ownership of the key and metadata, so users can free them after
// this API call.
LiteRtStatus LiteRtAddModelMetadata(LiteRtModel model, const char* metadata_key,
                                    const void* metadata_buffer,
                                    size_t metadata_buffer_size);

// Get the index of the entry subgraph.
// TODO: b/365299994 - Figure out signatures.
LiteRtStatus LiteRtGetMainModelSubgraphIndex(
    LiteRtModel model, LiteRtParamIndex* main_subgraph_index);

// Get number of subgraphs in model.
LiteRtStatus LiteRtGetNumModelSubgraphs(LiteRtModel model,
                                        LiteRtParamIndex* num_subgraphs);

// Get subgraph at given index in model.
//
// Note: The returned LiteRtSubgraph is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetModelSubgraph(LiteRtModel model,
                                    LiteRtParamIndex subgraph_index,
                                    LiteRtSubgraph* subgraph);

// Get the number of signatures defined in the model.
LiteRtStatus LiteRtGetNumModelSignatures(LiteRtModel model,
                                         LiteRtParamIndex* num_signatures);

// Get the signature at the given index in the model
//
// Note: The returned LiteRtSignature is only valid during the LiteRtModel's
// lifetime.
LiteRtStatus LiteRtGetModelSignature(LiteRtModel model,
                                     LiteRtParamIndex signature_index,
                                     LiteRtSignature* signature);

// Destroy the given model, freeing any memory it owns.
void LiteRtDestroyModel(LiteRtModel model);

//
// Utility Types
//

// An append only list of ops.
LiteRtStatus LiteRtPushOp(LiteRtOpList op_list, LiteRtOp op,
                          LiteRtParamIndex partition_index);

//
// Serialization related functions
//

// Serializes model to valid tflite flatbuffer bytes with signatures.
//
// This destroys the model before it returns unless destroy_model is false.
// Caller takes ownership of `buf`. Flatbuffers are packed into their arrays
// back to front, so the valid flatbuffer is buf[offset, size]. See the above
// options for more details.
LiteRtStatus LiteRtSerializeModelWithSignatures(
    LiteRtModel model, uint8_t** buf, size_t* size, size_t* offset,
    bool destroy_model, char** signatures, LiteRtParamIndex num_signatures,
    LiteRtModelSerializationOptions options);

// Serializes model to valid tflite flatbuffer bytes.
//
// This destroys the model before it returns unless destroy_model is false.
// Caller takes ownership of `buf`. Flatbuffers are packed into their arrays
// back to front, so the valid flatbuffer is buf[offset, size]. See the above
// options for more details.
LiteRtStatus LiteRtSerializeModel(LiteRtModel model, uint8_t** buf,
                                  size_t* size, size_t* offset,
                                  bool destroy_model,
                                  LiteRtModelSerializationOptions options);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_MODEL_H_
