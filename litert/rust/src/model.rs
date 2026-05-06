// Copyright 2026 Google LLC.
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

#![allow(non_upper_case_globals)]

use crate::bindings::*;
use crate::call_check_status;
use crate::error::{Error, ErrorCause};
use crate::helper_funs::c_str_to_str;
use crate::ElementType;
use std::ffi::{c_char, c_void, CString};
use std::marker::PhantomData;

/// `Model` is a wrapper around the LiteRtModel C struct.
/// Usually represents a model loaded from a file.
pub struct Model {
    pub(crate) raw_model: LiteRtModel,
}

/// `Subgraph` is a wrapper around the LiteRtSubgraph C struct.
/// It represents a subgraph of a model.
pub struct Subgraph<'a> {
    raw_subgraph: LiteRtSubgraph,
    _phantom: PhantomData<&'a LiteRtSubgraph>,
}

/// `Signature` is a wrapper around the LiteRtSignature C struct.
/// It represents a signature of a model.
pub struct Signature<'a> {
    raw_signature: LiteRtSignature,
    _phantom: PhantomData<&'a LiteRtSignature>,
}

enum InputOutputNamesIteratorKind {
    Input,
    Output,
}

/// An iterator over the input or output names of a signature.
pub struct InputOutputNamesIterator<'a> {
    signature: &'a Signature<'a>,
    index: LiteRtParamIndex,
    total_num_names: LiteRtParamIndex,
    kind: InputOutputNamesIteratorKind,
}

impl Signature<'_> {
    /// Returns the key of the signature.
    pub fn key(&self) -> Result<&str, Error> {
        let mut key: *const c_char = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointers.
            unsafe { LiteRtGetSignatureKey(self.raw_signature, &mut key) },
            ErrorCause::GetSignatureKey
        );
        // SAFETY: We assume that if C API returns OK then the output is valid.
        unsafe { c_str_to_str(key) }
    }

    /// Returns the subgraph associated with the signature.
    pub fn subgraph(&self) -> Result<Subgraph<'_>, Error> {
        let mut raw_subgraph_ptr: LiteRtSubgraph = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointers.
            unsafe { LiteRtGetSignatureSubgraph(self.raw_signature, &mut raw_subgraph_ptr) },
            ErrorCause::GetSignatureSubgraph
        );
        Ok(Subgraph { raw_subgraph: raw_subgraph_ptr, _phantom: PhantomData {} })
    }

    /// Returns the number of inputs of the signature.
    pub fn num_inputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_inputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumSignatureInputs(self.raw_signature, &mut num_inputs) },
            ErrorCause::GetNumSignatureInputs
        );
        Ok(num_inputs)
    }

    /// Returns an iterator over the input names of the signature.
    pub fn input_names(&self) -> Result<InputOutputNamesIterator<'_>, Error> {
        let num_inputs = self.num_inputs()?;
        Ok(InputOutputNamesIterator {
            signature: self,
            index: 0,
            total_num_names: num_inputs,
            kind: InputOutputNamesIteratorKind::Input,
        })
    }

    /// Returns the number of outputs of the signature.
    pub fn num_outputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_outputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumSignatureOutputs(self.raw_signature, &mut num_outputs) },
            ErrorCause::GetNumSignatureOutputs
        );
        Ok(num_outputs)
    }

    /// Returns an iterator over the output names of the signature.
    pub fn output_names(&self) -> Result<InputOutputNamesIterator<'_>, Error> {
        let num_outputs = self.num_outputs()?;
        Ok(InputOutputNamesIterator {
            signature: self,
            index: 0,
            total_num_names: num_outputs,
            kind: InputOutputNamesIteratorKind::Output,
        })
    }
}

impl<'a> Iterator for InputOutputNamesIterator<'a> {
    type Item = Result<&'a str, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total_num_names {
            return None;
        }
        let mut name: *const c_char = std::ptr::null_mut();
        // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper function.
        // We assume that the output is valid if the return status is OK or don't use the output pointers.
        unsafe {
            let status = match self.kind {
                InputOutputNamesIteratorKind::Input => {
                    LiteRtGetSignatureInputName(self.signature.raw_signature, self.index, &mut name)
                }
                InputOutputNamesIteratorKind::Output => LiteRtGetSignatureOutputName(
                    self.signature.raw_signature,
                    self.index,
                    &mut name,
                ),
            };
            if status != LiteRtStatus_kLiteRtStatusOk {
                return Some(Err(Error::new(ErrorCause::GetSignatureInputName, status)));
            }
        }
        self.index += 1;
        // SAFETY: We assume that if C API returns OK then the output is valid.
        let name_str = unsafe { c_str_to_str(name) };
        match name_str {
            Ok(name) => Some(Ok(name)),
            Err(e) => Some(Err(e)),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_num_names, Some(self.total_num_names))
    }
}

/// An iterator over the signatures of a model.
pub struct SignatureIterator<'a> {
    model: &'a Model,
    index: LiteRtParamIndex,
    total_num_signatures: LiteRtParamIndex,
}

impl<'a> Iterator for SignatureIterator<'a> {
    type Item = Result<Signature<'a>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total_num_signatures {
            return None;
        }
        let mut raw_signature_ptr: LiteRtSignature = std::ptr::null_mut();
        // SAFETY: self.model.raw_model is always valid as it's initialized by a wrapper function.
        // self.index is always valid, it is explicitly limited to the valid range.
        // We assume that the output is valid if the return status is OK or don't use the output pointers.
        unsafe {
            let status =
                LiteRtGetModelSignature(self.model.raw_model, self.index, &mut raw_signature_ptr);
            self.index += 1;
            if status != LiteRtStatus_kLiteRtStatusOk {
                return Some(Err(Error::new(ErrorCause::GetSignature, status)));
            }
        }
        Some(Ok(Signature { raw_signature: raw_signature_ptr, _phantom: PhantomData {} }))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_num_signatures, Some(self.total_num_signatures))
    }
}

/// `Tensor` is a wrapper around the LiteRtTensor C struct.
/// It represents a tensor in a model.
pub struct Tensor<'a> {
    raw_tensor: LiteRtTensor,
    _phantom: PhantomData<&'a LiteRtTensor>,
}

impl<'a> Tensor<'a> {
    fn type_id(&self) -> Result<LiteRtTensorTypeId, Error> {
        let mut raw_tensor_type = LiteRtTensorTypeId_kLiteRtRankedTensorType;
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe { LiteRtGetTensorTypeId(self.raw_tensor, &mut raw_tensor_type) },
            ErrorCause::GetTensorTypeId
        );
        Ok(raw_tensor_type)
    }

    /// Returns the unranked tensor type.
    pub fn unranked_tensor_type(&self) -> Result<LiteRtUnrankedTensorType, Error> {
        let mut raw_tensor_type = LiteRtUnrankedTensorType { element_type: 0 };
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe { LiteRtGetUnrankedTensorType(self.raw_tensor, &mut raw_tensor_type) },
            ErrorCause::GetUnrankedTensorType
        );
        Ok(raw_tensor_type)
    }

    /// Returns the ranked tensor type.
    pub fn ranked_tensor_type(&self) -> Result<LiteRtRankedTensorType, Error> {
        let mut raw_tensor_type = LiteRtRankedTensorType::default();
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe { LiteRtGetRankedTensorType(self.raw_tensor, &mut raw_tensor_type) },
            ErrorCause::GetRankedTensorType
        );
        Ok(raw_tensor_type)
    }

    /// Returns the element type of the tensor.
    pub fn element_type(&self) -> Result<ElementType, Error> {
        match self.type_id()? {
            LiteRtTensorTypeId_kLiteRtRankedTensorType => {
                let rtt = self.ranked_tensor_type()?;
                let rtt_element_type = ElementType::from_c_enum(rtt.element_type)?;
                Ok(rtt_element_type)
            }
            LiteRtTensorTypeId_kLiteRtUnrankedTensorType => {
                // TODO(mgubin): Add support for layout.
                let utt = self.unranked_tensor_type()?;
                let utt_element_type = ElementType::from_c_enum(utt.element_type)?;
                Ok(utt_element_type)
            }
            _ => Err(Error::new(
                ErrorCause::InvalidTensorTypeId,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )),
        }
    }

    /// Returns the name of the tensor.
    pub fn name(&self) -> Result<&str, Error> {
        let mut name: *const c_char = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe { LiteRtGetTensorName(self.raw_tensor, &mut name) },
            ErrorCause::GetTensorName
        );
        // SAFETY: We assume that if C API returns OK then the output is valid.
        unsafe { c_str_to_str(name) }
    }
}

impl<'a> Subgraph<'a> {
    /// Returns the number of inputs of the subgraph.
    pub fn num_inputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_inputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumSubgraphInputs(self.raw_subgraph, &mut num_inputs) },
            ErrorCause::GetNumSubgraphInputs
        );
        Ok(num_inputs)
    }

    /// Returns the number of outputs of the subgraph.
    pub fn num_outputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_outputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumSubgraphOutputs(self.raw_subgraph, &mut num_outputs) },
            ErrorCause::GetNumSubgraphOutputs
        );
        Ok(num_outputs)
    }

    fn input_tensor(&self, tensor_index: LiteRtParamIndex) -> Result<Tensor<'_>, Error> {
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointers.
            unsafe { LiteRtGetSubgraphInput(self.raw_subgraph, tensor_index, &mut raw_tensor_ptr) },
            ErrorCause::GetSubgraphInput
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the input tensor with the given name.
    pub fn input_tensor_by_name(&self, tensor_name: &str) -> Result<Tensor<'_>, Error> {
        let num_inputs = self.num_inputs()?;
        for i in 0..num_inputs {
            let tensor = self.input_tensor(i)?;
            if tensor.name()? == tensor_name {
                return Ok(tensor);
            }
        }
        return Err(Error::new(
            ErrorCause::SubgraphInputTensorByNameNotFound,
            LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
        ));
    }

    fn output_tensor(&self, tensor_index: LiteRtParamIndex) -> Result<Tensor<'_>, Error> {
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe {
                LiteRtGetSubgraphOutput(self.raw_subgraph, tensor_index, &mut raw_tensor_ptr)
            },
            ErrorCause::GetSubgraphOutput
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the output tensor with the given name.
    pub fn output_tensor_by_name(&self, tensor_name: &str) -> Result<Tensor<'_>, Error> {
        let num_outputs = self.num_outputs()?;
        for i in 0..num_outputs {
            let tensor = self.output_tensor(i)?;
            if tensor.name()? == tensor_name {
                return Ok(tensor);
            }
        }
        return Err(Error::new(
            ErrorCause::SubgraphOutputTensorByNameNotFound,
            LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
        ));
    }
}

impl Model {
    /// Creates a model from a file path.
    pub fn create_model_from_file(path: &str) -> Result<Self, Error> {
        let path_c_string =
            CString::new(path).expect("CString::new failed: string contains null bytes");
        let c_ptr: *const c_char = path_c_string.as_ptr();
        let mut raw_model_ptr: LiteRtModel = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: c_ptr is a valid pointer to the memory buffer provided by safe Rust code.
            unsafe { LiteRtCreateModelFromFile(c_ptr, &mut raw_model_ptr) },
            ErrorCause::CreateModelFromFile
        );
        Ok(Model { raw_model: raw_model_ptr })
    }

    /// Creates a model from a memory buffer.
    pub fn create_model_from_buffer(buffer: &mut [u8]) -> Result<Self, Error> {
        let mut raw_model_ptr: LiteRtModel = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: buffer is a valid pointer to the memory buffer provided by safe Rust code.
            unsafe {
                LiteRtCreateModelFromBuffer(
                    buffer.as_ptr() as *const c_void,
                    buffer.len(),
                    &mut raw_model_ptr,
                )
            },
            ErrorCause::CreateModelFromBuffer
        );
        Ok(Model { raw_model: raw_model_ptr })
    }

    /// Returns the number of subgraphs in the model.
    pub fn num_subgraphs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_subgraphs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_model is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumModelSubgraphs(self.raw_model, &mut num_subgraphs) },
            ErrorCause::GetNumModelSubgraphs
        );
        Ok(num_subgraphs)
    }

    /// Returns the number of signatures in the model.
    pub fn num_signatures(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_signatures: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_model is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe { LiteRtGetNumModelSignatures(self.raw_model, &mut num_signatures) },
            ErrorCause::GetNumModelSignatures
        );
        Ok(num_signatures)
    }

    /// Returns an iterator over the signatures of the model.
    pub fn signatures(&self) -> Result<SignatureIterator<'_>, Error> {
        Ok(SignatureIterator {
            model: self,
            index: 0,
            total_num_signatures: self.num_signatures()?,
        })
    }

    /// Returns the signature at the given index.
    pub fn signature(&self, index: LiteRtParamIndex) -> Result<Signature<'_>, Error> {
        let mut raw_signature_ptr: LiteRtSignature = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_model is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointers.
            unsafe { LiteRtGetModelSignature(self.raw_model, index, &mut raw_signature_ptr) },
            ErrorCause::GetModelSignature
        );
        Ok(Signature { raw_signature: raw_signature_ptr, _phantom: PhantomData{} })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: self.raw_model is always valid, it's guaranteed to be initialized by
        // create* function.
        unsafe {
            LiteRtDestroyModel(self.raw_model);
        }
    }
}
