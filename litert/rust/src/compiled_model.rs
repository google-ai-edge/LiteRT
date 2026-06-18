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

//! The compiled model is the result of compiling a model with specific options.
//! It can be used to run inference on the model.
use crate::bindings::*;
use crate::call_check_status;
use crate::environment::Environment;
use crate::error::{Error, ErrorCause};
use crate::model::{Model, Tensor};
use crate::tensor_buffer::{TensorBuffer, TensorBufferRequirements};

/// Options for compiling a model.
pub struct Options {
    raw_options: LiteRtOptions,
}

/// Hardware accelerators that can be used for inference.
pub enum LiteRtHwAccelerator {
    None,
    Cpu,
    Gpu,
    Npu,
}

impl LiteRtHwAccelerator {
    pub fn to_c_enum(&self) -> LiteRtHwAccelerators {
        match self {
            Self::None => LiteRtHwAccelerators_kLiteRtHwAcceleratorNone,
            Self::Cpu => LiteRtHwAccelerators_kLiteRtHwAcceleratorCpu,
            Self::Gpu => LiteRtHwAccelerators_kLiteRtHwAcceleratorGpu,
            Self::Npu => LiteRtHwAccelerators_kLiteRtHwAcceleratorNpu,
        }
    }
}

impl Options {
    /// Creates a new set of options with default values.
    pub fn default() -> Result<Self, Error> {
        let mut raw_options_ptr: *mut LiteRtOptionsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: The raw_options_ptr is initialized to null_mut before and points to a valid object
            // after the function call, if it's successful.
            unsafe { LiteRtCreateOptions(&mut raw_options_ptr) },
            ErrorCause::CreateOptions
        );
        Ok(Self {
            raw_options: raw_options_ptr,
        })
    }

    /// Creates a new set of options with the specified hardware accelerator.
    pub fn create_with_accelerator(accelerator: LiteRtHwAccelerator) -> Result<Self, Error> {
        let accelerator_c_enum = accelerator.to_c_enum();
        let options = Self::default()?;
        call_check_status!(
            // SAFETY: options.raw_options is valid because it's created by calling the default() function.
            // accelerator_c_enum is valid because it's created by calling the to_c_enum() function.
            unsafe {
                LiteRtSetOptionsHardwareAccelerators(options.raw_options, accelerator_c_enum)
            },
            ErrorCause::SetOptionsHardwareAccelerators
        );
        Ok(options)
    }
}

impl Drop for Options {
    fn drop(&mut self) {
        // SAFETY: self.raw_options is valid because it's created by calling the default() function.
        unsafe {
            LiteRtDestroyOptions(self.raw_options);
        }
    }
}

/// A compiled model that can be used to run inference.
pub struct CompiledModel {
    pub(crate) raw_compiled_model: LiteRtCompiledModel,
}

impl CompiledModel {
    /// Creates a new compiled model.
    pub fn create(
        environment: &Environment,
        model: &Model,
        options: &Options,
    ) -> Result<Self, Error> {
        let mut raw_compiled_model_ptr: *mut LiteRtCompiledModelT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: All input pointers are initalized before and point to valid objects.
            unsafe {
                LiteRtCreateCompiledModel(
                    environment.raw_environment,
                    model.raw_model,
                    options.raw_options,
                    &mut raw_compiled_model_ptr,
                )
            },
            ErrorCause::CreateCompiledModel
        );
        Ok(CompiledModel {
            raw_compiled_model: raw_compiled_model_ptr,
        })
    }

    fn input_buffer_requirements(
        &self,
        signature_index: LiteRtParamIndex,
        input_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let mut requirements_ptr: *mut LiteRtTensorBufferRequirementsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the create() function.
            unsafe {
                LiteRtGetCompiledModelInputBufferRequirements(
                    self.raw_compiled_model,
                    signature_index,
                    input_index,
                    &mut requirements_ptr,
                )
            },
            ErrorCause::GetCompiledModelInputBufferRequirements
        );
        Ok(TensorBufferRequirements::new(requirements_ptr))
    }

    fn output_buffer_requirements(
        &self,
        signature_index: LiteRtParamIndex,
        output_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let mut requirements_ptr: *mut LiteRtTensorBufferRequirementsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the create() function.
            unsafe {
                LiteRtGetCompiledModelOutputBufferRequirements(
                    self.raw_compiled_model,
                    signature_index,
                    output_index,
                    &mut requirements_ptr,
                )
            },
            ErrorCause::GetCompiledModelOutputBufferRequirements
        );
        Ok(TensorBufferRequirements::new(requirements_ptr))
    }

    /// Creates a set of input tensor buffers for the specified signature.
    pub fn create_input_tensor_buffers(
        &self,
        environment: &Environment,
        model: &Model,
        signature_index: LiteRtParamIndex,
    ) -> Result<Vec<TensorBuffer<'_>>, Error> {
        let signature = model.signature(signature_index)?;
        let subgraph = signature.subgraph()?;
        let mut result = Vec::with_capacity(signature.num_inputs()?);
        for (i, input_name) in signature.input_names()?.enumerate() {
            let input_requirements = self.input_buffer_requirements(signature_index, i)?;
            let tensor = subgraph.input_tensor_by_name(input_name?)?;
            let buffer =
                CompiledModel::create_buffer_impl(environment, &input_requirements, &tensor)?;
            result.push(buffer);
        }
        Ok(result)
    }

    /// Creates a set of output tensor buffers for the specified signature.
    pub fn create_output_tensor_buffers(
        &self,
        environment: &Environment,
        model: &Model,
        signature_index: LiteRtParamIndex,
    ) -> Result<Vec<TensorBuffer<'_>>, Error> {
        let signature = model.signature(signature_index)?;
        let subgraph = signature.subgraph()?;
        let mut result = Vec::with_capacity(signature.num_outputs()?);
        for (i, output_name) in signature.output_names()?.enumerate() {
            let output_requirements = self.output_buffer_requirements(signature_index, i)?;
            let tensor = subgraph.output_tensor_by_name(output_name?)?;
            let buffer =
                CompiledModel::create_buffer_impl(environment, &output_requirements, &tensor)?;
            result.push(buffer);
        }
        Ok(result)
    }

    fn create_buffer_impl<'a>(
        environment: &Environment,
        requirements: &TensorBufferRequirements,
        tensor: &Tensor,
    ) -> Result<TensorBuffer<'a>, Error> {
        let supported_types = requirements.supported_types()?;
        // For simplicity we just pick the first supported tensor buffer type.
        let Some(buffer_type) = supported_types.get(0) else {
            return Err(Error::new(
                ErrorCause::InputDoesntSupportAnyTensorBufferTypes,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        };
        let tensor_type = tensor.ranked_tensor_type()?;
        let element_type = tensor.element_type()?;
        let buffer_size = requirements.buffer_size()?;
        TensorBuffer::new(
            environment,
            &tensor_type,
            buffer_type,
            buffer_size,
            element_type,
        )
    }

    /// Runs inference on the compiled model.
    pub fn run(
        &self,
        signature_index: LiteRtParamIndex,
        input: &[TensorBuffer<'_>],
        output: &[TensorBuffer<'_>],
    ) -> Result<(), Error> {
        let mut input_ptrs: Vec<_> = input
            .iter()
            .map(|tensor| tensor.raw_tensor_buffer)
            .collect();
        let mut output_ptrs: Vec<_> = output
            .iter()
            .map(|tensor| tensor.raw_tensor_buffer)
            .collect();
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the create() function.
            // input_ptrs and output_ptrs are valid because they are created in the function.
            unsafe {
                LiteRtRunCompiledModel(
                    self.raw_compiled_model,
                    signature_index,
                    input_ptrs.len(),
                    input_ptrs.as_mut_ptr(),
                    output_ptrs.len(),
                    output_ptrs.as_mut_ptr(),
                )
            },
            ErrorCause::RunCompiledModel
        );
        Ok(())
    }
}
