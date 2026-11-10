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
use crate::environment::Environment;
use crate::error::{Error, ErrorCause};
use crate::helper_funs::c_str_to_str;
use crate::ElementType;
use std::borrow::Cow;
use std::ffi::{c_char, c_void, CString};
use std::marker::PhantomData;

/// `Model` is a wrapper around the LiteRtModel C struct.
/// Usually represents a model loaded from a file.
pub struct Model<'a> {
    pub(crate) raw_model: LiteRtModel,
    // If the model is created from a buffer, it can own or borrow the buffer.
    // This field is used to keep the buffer alive for the lifetime of the model.
    _buffer: Option<Cow<'a, [u8]>>,
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

impl<'a> Signature<'a> {
    /// Returns the key string identifying this signature in the model.
    ///
    /// # Returns
    ///
    /// * `Ok(&str)` - The signature key string owned by the parent [`Model`].
    /// * `Err(Error)` - If querying the signature key fails or the C string is not valid UTF-8.
    pub fn key(&self) -> Result<&'a str, Error> {
        let mut key: *const c_char = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper
            // function. We assume that the output is valid if the return status is OK
            // or don't use the output pointers.
            unsafe { LiteRtGetSignatureKey(self.raw_signature, &mut key) },
            ErrorCause::GetSignatureKey
        );
        // SAFETY: We assume that if C API returns OK then the output is valid.
        unsafe { c_str_to_str(key) }
    }

    /// Returns the [`Subgraph`] associated with this signature.
    ///
    /// # Returns
    ///
    /// * `Ok(Subgraph)` - A wrapper around the underlying subgraph for this signature.
    /// * `Err(Error)` - If retrieving the signature's subgraph fails.
    pub fn subgraph(&self) -> Result<Subgraph<'a>, Error> {
        let mut raw_subgraph_ptr: LiteRtSubgraph = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper
            // function. We assume that the output is valid if the return status is OK
            // or don't use the output pointers.
            unsafe { LiteRtGetSignatureSubgraph(self.raw_signature, &mut raw_subgraph_ptr) },
            ErrorCause::GetSignatureSubgraph
        );
        Ok(Subgraph { raw_subgraph: raw_subgraph_ptr, _phantom: PhantomData {} })
    }

    /// Returns the number of inputs defined in this signature.
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtParamIndex)` - The total number of input tensors in the signature.
    /// * `Err(Error)` - If querying the number of signature inputs fails.
    pub fn num_inputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_inputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper
            // function.
            unsafe { LiteRtGetNumSignatureInputs(self.raw_signature, &mut num_inputs) },
            ErrorCause::GetNumSignatureInputs
        );
        Ok(num_inputs)
    }

    /// Returns an iterator over the input argument names of this signature.
    ///
    /// # Returns
    ///
    /// * `Ok(InputOutputNamesIterator)` - An iterator yielding `Result<&str, Error>` for each input
    ///   name in signature order.
    /// * `Err(Error)` - If querying the number of signature inputs fails.
    pub fn input_names(&self) -> Result<InputOutputNamesIterator<'_>, Error> {
        let num_inputs = self.num_inputs()?;
        Ok(InputOutputNamesIterator {
            signature: self,
            index: 0,
            total_num_names: num_inputs,
            kind: InputOutputNamesIteratorKind::Input,
        })
    }

    /// Returns the number of outputs defined in this signature.
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtParamIndex)` - The total number of output tensors in the signature.
    /// * `Err(Error)` - If querying the number of signature outputs fails.
    pub fn num_outputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_outputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper
            // function.
            unsafe { LiteRtGetNumSignatureOutputs(self.raw_signature, &mut num_outputs) },
            ErrorCause::GetNumSignatureOutputs
        );
        Ok(num_outputs)
    }

    /// Returns an iterator over the output argument names of this signature.
    ///
    /// # Returns
    ///
    /// * `Ok(InputOutputNamesIterator)` - An iterator yielding `Result<&str, Error>` for each
    ///   output name in signature order.
    /// * `Err(Error)` - If querying the number of signature outputs fails.
    pub fn output_names(&self) -> Result<InputOutputNamesIterator<'_>, Error> {
        let num_outputs = self.num_outputs()?;
        Ok(InputOutputNamesIterator {
            signature: self,
            index: 0,
            total_num_names: num_outputs,
            kind: InputOutputNamesIteratorKind::Output,
        })
    }

    /// Returns the zero-based input index corresponding to the given signature input name.
    ///
    /// # Arguments
    ///
    /// * `name` - The signature input name to look up.
    ///
    /// # Returns
    ///
    /// * `Ok(InputOutputIndex)` - The index of the matching input in this signature.
    /// * `Err(Error)` - If no input with `name` exists in the signature or reading input names
    ///   fails.
    pub fn input_index_by_name(&self, name: &str) -> Result<InputOutputIndex, Error> {
        for (i, input_name) in self.input_names()?.enumerate() {
            if input_name? == name {
                return Ok(InputOutputIndex::new(i));
            }
        }
        Err(Error::new(ErrorCause::SignatureInputNotFound, LiteRtStatus_kLiteRtStatusErrorNotFound))
    }

    /// Returns the zero-based output index corresponding to the given signature output name.
    ///
    /// # Arguments
    ///
    /// * `name` - The signature output name to look up.
    ///
    /// # Returns
    ///
    /// * `Ok(InputOutputIndex)` - The index of the matching output in this signature.
    /// * `Err(Error)` - If no output with `name` exists in the signature or reading output names
    ///   fails.
    pub fn output_index_by_name(&self, name: &str) -> Result<InputOutputIndex, Error> {
        for (i, output_name) in self.output_names()?.enumerate() {
            if output_name? == name {
                return Ok(InputOutputIndex::new(i));
            }
        }
        Err(Error::new(
            ErrorCause::SignatureOutputNotFound,
            LiteRtStatus_kLiteRtStatusErrorNotFound,
        ))
    }

    /// Returns the input [`Tensor`] at the given signature input index.
    ///
    /// # Arguments
    ///
    /// * `input_index` - The zero-based index of the input tensor in the signature (or a type
    ///   convertible into [`InputOutputIndex`]).
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The input [`Tensor`] at `input_index`.
    /// * `Err(Error)` - If `input_index` is out of bounds or retrieving the tensor fails.
    pub fn input_tensor_by_index(
        &self,
        input_index: impl Into<InputOutputIndex>,
    ) -> Result<Tensor<'a>, Error> {
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper
            // function. We assume that the output is valid if the return status is OK or don't use
            // the output pointer.
            unsafe {
                LiteRtGetSignatureInputTensorByIndex(
                    self.raw_signature,
                    input_index.into().value(),
                    &mut raw_tensor_ptr,
                )
            },
            ErrorCause::GetSignatureInputTensor
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the input [`Tensor`] with the given signature input name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the input in the signature.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The input [`Tensor`] matching `name`.
    /// * `Err(Error)` - If `name` contains an interior NUL byte, no input with `name` exists in the
    ///   signature, or retrieving the tensor fails.
    pub fn input_tensor_by_name(&self, name: &str) -> Result<Tensor<'a>, Error> {
        let name_c_string = CString::new(name).map_err(|_| {
            Error::new(
                ErrorCause::InvalidStringEncoding,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )
        })?;
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature and name_c_string are valid pointers.
            unsafe {
                LiteRtGetSignatureInputTensor(
                    self.raw_signature,
                    name_c_string.as_ptr(),
                    &mut raw_tensor_ptr,
                )
            },
            ErrorCause::GetSignatureInputTensor
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the output [`Tensor`] at the given signature output index.
    ///
    /// # Arguments
    ///
    /// * `output_index` - The zero-based index of the output tensor in the signature (or a type
    ///   convertible into [`InputOutputIndex`]).
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The output [`Tensor`] at `output_index`.
    /// * `Err(Error)` - If `output_index` is out of bounds or retrieving the tensor fails.
    pub fn output_tensor_by_index(
        &self,
        output_index: impl Into<InputOutputIndex>,
    ) -> Result<Tensor<'a>, Error> {
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper
            // function. We assume that the output is valid if the return status is OK or don't use
            // the output pointer.
            unsafe {
                LiteRtGetSignatureOutputTensorByIndex(
                    self.raw_signature,
                    output_index.into().value(),
                    &mut raw_tensor_ptr,
                )
            },
            ErrorCause::GetSignatureOutputTensor
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the output [`Tensor`] with the given signature output name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the output in the signature.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The output [`Tensor`] matching `name`.
    /// * `Err(Error)` - If `name` contains an interior NUL byte, no output with `name` exists in
    ///   the signature, or retrieving the tensor fails.
    pub fn output_tensor_by_name(&self, name: &str) -> Result<Tensor<'a>, Error> {
        let name_c_string = CString::new(name).map_err(|_| {
            Error::new(
                ErrorCause::InvalidStringEncoding,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )
        })?;
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_signature and name_c_string are valid pointers.
            unsafe {
                LiteRtGetSignatureOutputTensor(
                    self.raw_signature,
                    name_c_string.as_ptr(),
                    &mut raw_tensor_ptr,
                )
            },
            ErrorCause::GetSignatureOutputTensor
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }
}

impl<'a> Iterator for InputOutputNamesIterator<'a> {
    type Item = Result<&'a str, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total_num_names {
            return None;
        }
        let current_index = self.index;
        self.index += 1;
        let mut name: *const c_char = std::ptr::null_mut();
        // SAFETY: self.raw_signature is always valid as it's initialized by a wrapper function.
        // We assume that the output is valid if the return status is OK or don't use the output
        // pointers.
        unsafe {
            let (status, cause) = match self.kind {
                InputOutputNamesIteratorKind::Input => (
                    LiteRtGetSignatureInputName(
                        self.signature.raw_signature,
                        current_index,
                        &mut name,
                    ),
                    ErrorCause::GetSignatureInputName,
                ),
                InputOutputNamesIteratorKind::Output => (
                    LiteRtGetSignatureOutputName(
                        self.signature.raw_signature,
                        current_index,
                        &mut name,
                    ),
                    ErrorCause::GetSignatureOutputName,
                ),
            };
            if status != LiteRtStatus_kLiteRtStatusOk {
                return Some(Err(Error::new(cause, status)));
            }
        }
        // SAFETY: We assume that if C API returns OK then the output is valid.
        let name_str = unsafe { c_str_to_str(name) };
        match name_str {
            Ok(name) => Some(Ok(name)),
            Err(e) => Some(Err(e)),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_num_names.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

/// An iterator over the signatures of a model.
pub struct SignatureIterator<'a> {
    model: &'a Model<'a>,
    index: LiteRtParamIndex,
    total_num_signatures: SignatureIndex,
}

impl<'a> Iterator for SignatureIterator<'a> {
    type Item = Result<Signature<'a>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total_num_signatures.value() {
            return None;
        }
        let index = SignatureIndex::new(self.index);
        self.index += 1;
        Some(self.model.signature(index))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_num_signatures.value().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

/// `Tensor` is a wrapper around the LiteRtTensor C struct.
/// It represents a tensor in a model.
pub struct Tensor<'a> {
    raw_tensor: LiteRtTensor,
    _phantom: PhantomData<&'a LiteRtTensor>,
}

impl<'a> Tensor<'a> {
    /// Returns the underlying C tensor type ID (ranked or unranked).
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtTensorTypeId)` - The tensor type identifier (`kLiteRtRankedTensorType` or
    ///   `kLiteRtUnrankedTensorType`).
    /// * `Err(Error)` - If querying the tensor type ID fails.
    fn type_id(&self) -> Result<LiteRtTensorTypeId, Error> {
        let mut raw_tensor_type = LiteRtTensorTypeId_kLiteRtRankedTensorType;
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointer.
            unsafe { LiteRtGetTensorTypeId(self.raw_tensor, &mut raw_tensor_type) },
            ErrorCause::GetTensorTypeId
        );
        Ok(raw_tensor_type)
    }

    /// Returns the unranked tensor type descriptor for this tensor.
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtUnrankedTensorType)` - The C struct describing the unranked tensor's element
    ///   type.
    /// * `Err(Error)` - If this tensor is not an unranked tensor or querying its type fails.
    pub fn unranked_tensor_type(&self) -> Result<LiteRtUnrankedTensorType, Error> {
        let mut raw_tensor_type = LiteRtUnrankedTensorType { element_type: 0 };
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointer.
            unsafe { LiteRtGetUnrankedTensorType(self.raw_tensor, &mut raw_tensor_type) },
            ErrorCause::GetUnrankedTensorType
        );
        Ok(raw_tensor_type)
    }

    /// Returns the ranked tensor type descriptor (element type and layout/dimensions) for this
    /// tensor.
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtRankedTensorType)` - The C struct describing the ranked tensor's element type
    ///   and layout.
    /// * `Err(Error)` - If this tensor is not a ranked tensor or querying its type fails.
    pub fn ranked_tensor_type(&self) -> Result<LiteRtRankedTensorType, Error> {
        let mut raw_tensor_type = LiteRtRankedTensorType::default();
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointer.
            unsafe { LiteRtGetRankedTensorType(self.raw_tensor, &mut raw_tensor_type) },
            ErrorCause::GetRankedTensorType
        );
        Ok(raw_tensor_type)
    }

    /// Returns the [`ElementType`] of this tensor.
    ///
    /// # Returns
    ///
    /// * `Ok(ElementType)` - The data type of the tensor's elements.
    /// * `Err(Error)` - If the tensor's type ID or element type is unrecognized or cannot be
    ///   queried.
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

    /// Returns the internal name of the tensor in the subgraph.
    ///
    /// # Returns
    ///
    /// * `Ok(&str)` - The tensor's name string owned by the parent [`Model`].
    /// * `Err(Error)` - If querying the tensor name fails or the C string is not valid UTF-8.
    pub fn name(&self) -> Result<&'a str, Error> {
        let mut name: *const c_char = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_tensor is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointer.
            unsafe { LiteRtGetTensorName(self.raw_tensor, &mut name) },
            ErrorCause::GetTensorName
        );
        // SAFETY: We assume that if C API returns OK then the output is valid.
        unsafe { c_str_to_str(name) }
    }
}

/// A wrapper around `LiteRtParamIndex` representing the index of an input or output in a
/// signature or subgraph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct InputOutputIndex(LiteRtParamIndex);

impl InputOutputIndex {
    /// Creates a new [`InputOutputIndex`] from a raw `LiteRtParamIndex`.
    ///
    /// # Arguments
    ///
    /// * `index` - The zero-based parameter index.
    ///
    /// # Returns
    ///
    /// A new `InputOutputIndex` wrapping `index`.
    #[must_use]
    pub const fn new(index: LiteRtParamIndex) -> Self {
        InputOutputIndex(index)
    }

    /// Returns the underlying `LiteRtParamIndex` value.
    ///
    /// # Returns
    ///
    /// The zero-based `LiteRtParamIndex` value.
    #[must_use]
    pub const fn value(self) -> LiteRtParamIndex {
        self.0
    }
}

impl From<LiteRtParamIndex> for InputOutputIndex {
    fn from(index: LiteRtParamIndex) -> Self {
        InputOutputIndex(index)
    }
}

impl<'a> Subgraph<'a> {
    /// Returns the number of input tensors in the subgraph.
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtParamIndex)` - The number of inputs of the subgraph.
    /// * `Err(Error)` - If querying the subgraph input count fails.
    pub fn num_inputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_inputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumSubgraphInputs(self.raw_subgraph, &mut num_inputs) },
            ErrorCause::GetNumSubgraphInputs
        );
        Ok(num_inputs)
    }

    /// Returns the number of output tensors in the subgraph.
    ///
    /// # Returns
    ///
    /// * `Ok(LiteRtParamIndex)` - The number of outputs of the subgraph.
    /// * `Err(Error)` - If querying the subgraph output count fails.
    pub fn num_outputs(&self) -> Result<LiteRtParamIndex, Error> {
        let mut num_outputs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumSubgraphOutputs(self.raw_subgraph, &mut num_outputs) },
            ErrorCause::GetNumSubgraphOutputs
        );
        Ok(num_outputs)
    }

    /// Returns the subgraph input [`Tensor`] at the specified index.
    ///
    /// # Arguments
    ///
    /// * `tensor_index` - The zero-based index of the input tensor in the subgraph (or a type
    ///   convertible into [`InputOutputIndex`]).
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The subgraph input [`Tensor`] at `tensor_index`.
    /// * `Err(Error)` - If `tensor_index` is out of bounds or retrieving the tensor fails.
    pub fn input_tensor(
        &self,
        tensor_index: impl Into<InputOutputIndex>,
    ) -> Result<Tensor<'a>, Error> {
        let tensor_index = tensor_index.into().value();
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointers.
            unsafe { LiteRtGetSubgraphInput(self.raw_subgraph, tensor_index, &mut raw_tensor_ptr) },
            ErrorCause::GetSubgraphInput
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the subgraph input [`Tensor`] with the given internal tensor name.
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - The internal subgraph tensor name to search for.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The subgraph input [`Tensor`] whose [`Tensor::name`] matches `tensor_name`.
    /// * `Err(Error)` - If no subgraph input tensor matches `tensor_name` or querying tensors
    ///   fails.
    pub fn input_tensor_by_name(&self, tensor_name: &str) -> Result<Tensor<'a>, Error> {
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

    /// Returns the subgraph output [`Tensor`] at the specified index.
    ///
    /// # Arguments
    ///
    /// * `tensor_index` - The zero-based index of the output tensor in the subgraph (or a type
    ///   convertible into [`InputOutputIndex`]).
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The subgraph output [`Tensor`] at `tensor_index`.
    /// * `Err(Error)` - If `tensor_index` is out of bounds or retrieving the tensor fails.
    pub fn output_tensor(
        &self,
        tensor_index: impl Into<InputOutputIndex>,
    ) -> Result<Tensor<'a>, Error> {
        let tensor_index = tensor_index.into().value();
        let mut raw_tensor_ptr: LiteRtTensor = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_subgraph is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointer.
            unsafe {
                LiteRtGetSubgraphOutput(self.raw_subgraph, tensor_index, &mut raw_tensor_ptr)
            },
            ErrorCause::GetSubgraphOutput
        );
        Ok(Tensor { raw_tensor: raw_tensor_ptr, _phantom: PhantomData {} })
    }

    /// Returns the subgraph output [`Tensor`] with the given internal tensor name.
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - The internal subgraph tensor name to search for.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - The subgraph output [`Tensor`] whose [`Tensor::name`] matches
    ///   `tensor_name`.
    /// * `Err(Error)` - If no subgraph output tensor matches `tensor_name` or querying tensors
    ///   fails.
    pub fn output_tensor_by_name(&self, tensor_name: &str) -> Result<Tensor<'a>, Error> {
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

/// A wrapper around `LiteRtParamIndex` representing the index of a subgraph in a model.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubgraphIndex(LiteRtParamIndex);

impl SubgraphIndex {
    /// Creates a new [`SubgraphIndex`] from a raw `LiteRtParamIndex`.
    ///
    /// # Arguments
    ///
    /// * `index` - The zero-based subgraph index.
    ///
    /// # Returns
    ///
    /// A new `SubgraphIndex` wrapping `index`.
    #[must_use]
    pub const fn new(index: LiteRtParamIndex) -> Self {
        SubgraphIndex(index)
    }

    /// Returns the underlying `LiteRtParamIndex` value.
    ///
    /// # Returns
    ///
    /// The zero-based `LiteRtParamIndex` value.
    #[must_use]
    pub const fn value(self) -> LiteRtParamIndex {
        self.0
    }
}

impl From<LiteRtParamIndex> for SubgraphIndex {
    fn from(index: LiteRtParamIndex) -> Self {
        SubgraphIndex(index)
    }
}

/// A wrapper around `LiteRtParamIndex` representing the index of a signature in a model.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SignatureIndex(LiteRtParamIndex);

impl SignatureIndex {
    /// The default signature index (0).
    pub const DEFAULT: Self = Self(0);

    /// Creates a new [`SignatureIndex`] from a raw `LiteRtParamIndex`.
    ///
    /// # Arguments
    ///
    /// * `index` - The zero-based signature index.
    ///
    /// # Returns
    ///
    /// A new `SignatureIndex` wrapping `index`.
    #[must_use]
    pub const fn new(index: LiteRtParamIndex) -> Self {
        SignatureIndex(index)
    }

    /// Returns the underlying `LiteRtParamIndex` value.
    ///
    /// # Returns
    ///
    /// The zero-based `LiteRtParamIndex` value.
    #[must_use]
    pub const fn value(self) -> LiteRtParamIndex {
        self.0
    }
}

impl From<LiteRtParamIndex> for SignatureIndex {
    fn from(index: LiteRtParamIndex) -> Self {
        SignatureIndex(index)
    }
}

impl Model<'static> {
    /// Loads and creates a [`Model`] from a file path.
    ///
    /// # Arguments
    ///
    /// * `environment` - The LiteRT [`Environment`] used to initialize the model.
    /// * `path` - Path to the `.tflite` model file on disk.
    ///
    /// # Returns
    ///
    /// * `Ok(Model)` - The loaded [`Model`].
    /// * `Err(Error)` - If `path` contains an interior NUL byte, or reading/parsing the model file
    ///   fails.
    pub fn create_model_from_file(environment: &Environment, path: &str) -> Result<Self, Error> {
        let path_c_string = CString::new(path).map_err(|_| {
            Error::new(
                ErrorCause::InvalidStringEncoding,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )
        })?;
        let c_ptr: *const c_char = path_c_string.as_ptr();
        let mut raw_model_ptr: LiteRtModel = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: c_ptr is a valid pointer to the memory buffer provided by safe Rust code.
            unsafe {
                LiteRtCreateModelFromFile(environment.raw_environment, c_ptr, &mut raw_model_ptr)
            },
            ErrorCause::CreateModelFromFile
        );
        Ok(Model { raw_model: raw_model_ptr, _buffer: None })
    }
}

impl<'a> Model<'a> {
    /// Creates a [`Model`] from an in-memory buffer.
    ///
    /// Accepts any type convertible into `Cow<'a, [u8]>`, such as a borrowed byte slice
    /// (`&'a [u8]`) or an owned byte vector (`Vec<u8>`).
    /// - When created from a borrowed slice (`&'a [u8]`), the returned [`Model<'a>`] is tied to the
    ///   lifetime `'a` of the slice without copying.
    /// - When created from an owned `Vec<u8>`, the [`Model<'static>`] takes ownership of the buffer
    ///   and keeps it alive until the model is dropped.
    ///
    /// # Arguments
    ///
    /// * `environment` - The LiteRT [`Environment`] used to initialize the model.
    /// * `buffer` - Borrowed byte slice (`&'a [u8]`) or owned byte vector (`Vec<u8>`) containing
    ///   the FlatBuffer model data.
    ///
    /// # Returns
    ///
    /// * `Ok(Model)` - The created [`Model`].
    /// * `Err(Error)` - If parsing or initializing the model from the buffer fails.
    ///
    /// # Examples
    ///
    /// Creating a model from a borrowed slice (`&[u8]`):
    ///
    /// ```no_run
    /// use litert::{EnvironmentBuilder, Model};
    ///
    /// # fn main() -> Result<(), litert::Error> {
    /// let environment = EnvironmentBuilder::build_default()?;
    ///
    /// let bytes: Vec<u8> = std::fs::read("/path/to/model.tflite").expect("Failed to read model file");
    /// let slice: &[u8] = &bytes;
    ///
    /// // Borrowed zero-copy: `model` cannot outlive `slice`.
    /// let model = Model::create_model_from_buffer(&environment, slice)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Creating a model from an owned `Vec<u8>`:
    ///
    /// ```no_run
    /// use litert::{EnvironmentBuilder, Model};
    ///
    /// # fn main() -> Result<(), litert::Error> {
    /// let environment = EnvironmentBuilder::build_default()?;
    ///
    /// let bytes: Vec<u8> = std::fs::read("/path/to/model.tflite").expect("Failed to read model file");
    ///
    /// // Ownership of `bytes` is moved into `model` (`Model<'static>`).
    /// let model: Model<'static> = Model::create_model_from_buffer(&environment, bytes)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_model_from_buffer(
        environment: &Environment,
        buffer: impl Into<Cow<'a, [u8]>>,
    ) -> Result<Self, Error> {
        let mut raw_model_ptr: LiteRtModel = std::ptr::null_mut();
        let buffer = buffer.into();
        call_check_status!(
            // SAFETY: buffer is a valid pointer to the memory buffer provided by safe Rust code.
            unsafe {
                LiteRtCreateModelFromBuffer(
                    environment.raw_environment,
                    buffer.as_ptr() as *const c_void,
                    buffer.len(),
                    &mut raw_model_ptr,
                )
            },
            ErrorCause::CreateModelFromBuffer
        );
        Ok(Model { raw_model: raw_model_ptr, _buffer: Some(buffer) })
    }

    /// Returns the number of subgraphs in the model.
    ///
    /// # Returns
    ///
    /// * `Ok(SubgraphIndex)` - A [`SubgraphIndex`] wrapping the total number of subgraphs.
    /// * `Err(Error)` - If querying the number of model subgraphs fails.
    pub fn num_subgraphs(&self) -> Result<SubgraphIndex, Error> {
        let mut num_subgraphs: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_model is always valid as it's initialized by a wrapper function.
            unsafe { LiteRtGetNumModelSubgraphs(self.raw_model, &mut num_subgraphs) },
            ErrorCause::GetNumModelSubgraphs
        );
        Ok(SubgraphIndex::new(num_subgraphs))
    }

    /// Returns the number of signatures defined in the model.
    ///
    /// # Returns
    ///
    /// * `Ok(SignatureIndex)` - A [`SignatureIndex`] wrapping the total number of signatures.
    /// * `Err(Error)` - If querying the number of model signatures fails.
    pub fn num_signatures(&self) -> Result<SignatureIndex, Error> {
        let mut num_signatures: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY: self.raw_model is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointer.
            unsafe { LiteRtGetNumModelSignatures(self.raw_model, &mut num_signatures) },
            ErrorCause::GetNumModelSignatures
        );
        Ok(SignatureIndex::new(num_signatures))
    }

    /// Returns an iterator over all [`Signature`]s of the model.
    ///
    /// # Returns
    ///
    /// * `Ok(SignatureIterator)` - An iterator yielding `Result<Signature, Error>` for each
    ///   signature in the model.
    /// * `Err(Error)` - If querying the number of model signatures fails.
    pub fn signatures(&self) -> Result<SignatureIterator<'_>, Error> {
        Ok(SignatureIterator {
            model: self,
            index: 0,
            total_num_signatures: self.num_signatures()?,
        })
    }

    /// Returns the [`Signature`] at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The [`SignatureIndex`] of the signature to retrieve.
    ///
    /// # Returns
    ///
    /// * `Ok(Signature)` - The [`Signature`] at `index`.
    /// * `Err(Error)` - If `index` is out of bounds or retrieving the signature fails.
    pub fn signature(&self, index: SignatureIndex) -> Result<Signature<'_>, Error> {
        let mut raw_signature_ptr: LiteRtSignature = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_model is always valid as it's initialized by a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the
            // output pointers.
            unsafe {
                LiteRtGetModelSignature(self.raw_model, index.value(), &mut raw_signature_ptr)
            },
            ErrorCause::GetModelSignature
        );
        Ok(Signature { raw_signature: raw_signature_ptr, _phantom: PhantomData {} })
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // SAFETY: self.raw_model is always valid, it's guaranteed to be initialized by
        // create* function.
        unsafe {
            LiteRtDestroyModel(self.raw_model);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::EnvironmentBuilder;

    const TEST_MODEL_PATHS: &[&str] = &[
        "third_party/odml/litert/litert/test/testdata/simple_model.tflite",
        "../test/testdata/simple_model.tflite",
    ];

    fn find_test_model_path() -> Option<&'static str> {
        TEST_MODEL_PATHS.iter().copied().find(|p| std::path::Path::new(p).exists())
    }

    #[test]
    fn test_input_output_index() {
        let idx = InputOutputIndex::new(3);
        assert_eq!(idx.value(), 3);
        let from_idx: InputOutputIndex = 5.into();
        assert_eq!(from_idx.value(), 5);

        let sg_idx = SubgraphIndex::new(2);
        assert_eq!(sg_idx.value(), 2);
        let from_sg: SubgraphIndex = 4.into();
        assert_eq!(from_sg.value(), 4);

        let sig_idx = SignatureIndex::new(1);
        assert_eq!(sig_idx.value(), 1);
        assert_eq!(SignatureIndex::DEFAULT.value(), 0);
        let from_sig: SignatureIndex = 6.into();
        assert_eq!(from_sig.value(), 6);
    }

    #[test]
    fn test_create_model_errors() {
        let env = EnvironmentBuilder::build_default().expect("Failed to create environment");
        let nul_err = Model::create_model_from_file(&env, "bad\0path.tflite");
        assert!(nul_err.is_err());
        assert_eq!(nul_err.err().unwrap().cause(), ErrorCause::InvalidStringEncoding);

        let missing_err = Model::create_model_from_file(&env, "/nonexistent/model.tflite");
        assert!(missing_err.is_err());
        assert_eq!(missing_err.err().unwrap().cause(), ErrorCause::CreateModelFromFile);

        let bad_buf: &[u8] = &[0, 1, 2, 3];
        let buf_err = Model::create_model_from_buffer(&env, bad_buf);
        assert!(buf_err.is_err());
        assert_eq!(buf_err.err().unwrap().cause(), ErrorCause::CreateModelFromBuffer);
    }

    #[test]
    fn test_model_from_file_and_buffer_and_signatures() {
        let Some(model_path) = find_test_model_path() else {
            return;
        };
        let env = EnvironmentBuilder::build_default().expect("Failed to create environment");

        // 1. Load from file
        let model_from_file =
            Model::create_model_from_file(&env, model_path).expect("Failed to load model file");
        assert_eq!(model_from_file.num_subgraphs().unwrap().value(), 1);
        assert_eq!(model_from_file.num_signatures().unwrap().value(), 1);

        // 2. Load from borrowed slice (&[u8]) and owned Vec<u8>
        let bytes = std::fs::read(model_path).expect("Failed to read model bytes");
        {
            let model_from_slice =
                Model::create_model_from_buffer(&env, bytes.as_slice()).expect("Failed from slice");
            assert_eq!(model_from_slice.num_signatures().unwrap().value(), 1);
        }

        let model: Model<'static> =
            Model::create_model_from_buffer(&env, bytes).expect("Failed from owned Vec");

        // 3. Test SignatureIterator & size_hint
        let mut sig_iter = model.signatures().expect("Failed to get signatures iterator");
        assert_eq!(sig_iter.size_hint(), (1, Some(1)));
        let sig = sig_iter.next().expect("Expected 1 signature").expect("Signature error");
        assert_eq!(sig_iter.size_hint(), (0, Some(0)));
        assert!(sig_iter.next().is_none());

        assert!(!sig.key().unwrap().is_empty());
        assert_eq!(sig.num_inputs().unwrap(), 2);
        assert_eq!(sig.num_outputs().unwrap(), 1);

        // 4. Test InputOutputNamesIterator & index/tensor lookups
        let mut in_names_iter = sig.input_names().unwrap();
        assert_eq!(in_names_iter.size_hint(), (2, Some(2)));
        let in0_name = in_names_iter.next().unwrap().unwrap();
        assert_eq!(in_names_iter.size_hint(), (1, Some(1)));
        let in1_name = in_names_iter.next().unwrap().unwrap();
        assert_eq!(in_names_iter.size_hint(), (0, Some(0)));
        assert!(in_names_iter.next().is_none());

        let out_names: Vec<_> = sig.output_names().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(out_names.len(), 1);
        let out0_name = out_names[0];

        assert_eq!(sig.input_index_by_name(in0_name).unwrap().value(), 0);
        assert_eq!(sig.input_index_by_name(in1_name).unwrap().value(), 1);
        assert_eq!(
            sig.input_index_by_name("nonexistent").unwrap_err().cause(),
            ErrorCause::SignatureInputNotFound
        );

        assert_eq!(sig.output_index_by_name(out0_name).unwrap().value(), 0);
        assert_eq!(
            sig.output_index_by_name("nonexistent").unwrap_err().cause(),
            ErrorCause::SignatureOutputNotFound
        );

        // 5. Test input_tensor_by_index / input_tensor_by_name / output_tensor_by_index /
        //    output_tensor_by_name
        let in0_by_idx = sig.input_tensor_by_index(0).unwrap();
        let in0_by_name = sig.input_tensor_by_name(in0_name).unwrap();
        assert_eq!(in0_by_idx.name().unwrap(), in0_by_name.name().unwrap());
        assert_eq!(in0_by_idx.element_type().unwrap(), ElementType::Float32);
        assert_eq!(in0_by_idx.ranked_tensor_type().unwrap().layout.rank(), 1);

        assert_eq!(
            sig.input_tensor_by_index(99).err().unwrap().cause(),
            ErrorCause::GetSignatureInputTensor
        );
        assert_eq!(
            sig.input_tensor_by_name("bad\0name").err().unwrap().cause(),
            ErrorCause::InvalidStringEncoding
        );
        assert_eq!(
            sig.input_tensor_by_name("nonexistent").err().unwrap().cause(),
            ErrorCause::GetSignatureInputTensor
        );

        let out0_by_idx = sig.output_tensor_by_index(0).unwrap();
        let out0_by_name = sig.output_tensor_by_name(out0_name).unwrap();
        assert_eq!(out0_by_idx.name().unwrap(), out0_by_name.name().unwrap());
        assert_eq!(out0_by_idx.element_type().unwrap(), ElementType::Float32);

        assert_eq!(
            sig.output_tensor_by_index(99).err().unwrap().cause(),
            ErrorCause::GetSignatureOutputTensor
        );
        assert_eq!(
            sig.output_tensor_by_name("bad\0name").err().unwrap().cause(),
            ErrorCause::InvalidStringEncoding
        );

        // 6. Test Subgraph methods
        let subgraph = sig.subgraph().unwrap();
        assert_eq!(subgraph.num_inputs().unwrap(), 2);
        assert_eq!(subgraph.num_outputs().unwrap(), 1);
        let sg_in0 = subgraph.input_tensor(0).unwrap();
        assert_eq!(
            subgraph.input_tensor_by_name(sg_in0.name().unwrap()).unwrap().name().unwrap(),
            sg_in0.name().unwrap()
        );
        let sg_out0 = subgraph.output_tensor(0).unwrap();
        assert_eq!(
            subgraph.output_tensor_by_name(sg_out0.name().unwrap()).unwrap().name().unwrap(),
            sg_out0.name().unwrap()
        );
    }
}
