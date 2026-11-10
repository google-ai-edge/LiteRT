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
use crate::model::{InputOutputIndex, Model, SignatureIndex, Tensor};
use crate::tensor_buffer::{TensorBuffer, TensorBufferRequirements};

/// Options for compiling a model.
pub struct Options {
    raw_options: LiteRtOptions,
}

/// Hardware accelerators that can be used for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteRtHwAccelerator {
    None,
    Cpu,
    Gpu,
    Npu,
}

impl LiteRtHwAccelerator {
    /// Converts the hardware accelerator enum variant into its corresponding LiteRT C enum value.
    ///
    /// # Returns
    ///
    /// The `LiteRtHwAccelerators` C enum constant matching `self`.
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
    /// Creates a new set of compilation options with default values.
    ///
    /// # Returns
    ///
    /// * `Ok(Options)` - A newly initialized `Options` instance with default settings.
    /// * `Err(Error)` - If the underlying LiteRT C API fails to create the options object.
    pub fn default() -> Result<Self, Error> {
        let mut raw_options_ptr: *mut LiteRtOptionsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: The raw_options_ptr is initialized to null_mut before and points to a valid
            // object after the function call, if it's successful.
            unsafe { LiteRtCreateOptions(&mut raw_options_ptr) },
            ErrorCause::CreateOptions
        );
        Ok(Self { raw_options: raw_options_ptr })
    }

    /// Creates a new set of compilation options configured with the specified hardware accelerator.
    ///
    /// # Arguments
    ///
    /// * `accelerator` - The hardware accelerator (`LiteRtHwAccelerator`) to target during
    ///   compilation and inference.
    ///
    /// # Returns
    ///
    /// * `Ok(Options)` - A newly initialized `Options` instance configured for `accelerator`.
    /// * `Err(Error)` - If creating the options or setting the hardware accelerator fails.
    pub fn create_with_accelerator(accelerator: LiteRtHwAccelerator) -> Result<Self, Error> {
        let accelerator_c_enum = accelerator.to_c_enum();
        let options = Self::default()?;
        call_check_status!(
            // SAFETY: options.raw_options is valid because it's created by calling the default()
            // function. accelerator_c_enum is valid because it's created by calling
            // the to_c_enum() function.
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
    /// Compiles a [`Model`] within the given [`Environment`] using the specified [`Options`].
    ///
    /// # Arguments
    ///
    /// * `environment` - The LiteRT [`Environment`] used to manage runtime resources and
    ///   accelerators. Must outlive the `CompiledModel` and any executions running on it.
    /// * `model` - The [`Model`] to compile. Must outlive the `CompiledModel`.
    /// * `options` - Compilation [`Options`] specifying hardware accelerators and other settings.
    ///
    /// # Returns
    ///
    /// * `Ok(CompiledModel)` - The compiled model ready for buffer creation and inference.
    /// * `Err(Error)` - If model compilation fails in the underlying LiteRT runtime.
    pub fn create(
        environment: &Environment,
        model: &Model<'_>,
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
        Ok(CompiledModel { raw_compiled_model: raw_compiled_model_ptr })
    }

    /// Queries the buffer requirements for a specific input tensor in a signature.
    ///
    /// # Arguments
    ///
    /// * `signature_index` - The index of the model signature.
    /// * `input_index` - The zero-based index of the input tensor within the signature.
    ///
    /// # Returns
    ///
    /// * `Ok(TensorBufferRequirements)` - The buffer requirements (supported buffer types, size,
    ///   alignment) for the specified input tensor.
    /// * `Err(Error)` - If the signature or input index is invalid or querying requirements fails.
    fn input_buffer_requirements(
        &self,
        signature_index: SignatureIndex,
        input_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let mut requirements_ptr: *mut LiteRtTensorBufferRequirementsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the
            // create() function.
            unsafe {
                LiteRtGetCompiledModelInputBufferRequirements(
                    self.raw_compiled_model,
                    signature_index.value(),
                    input_index,
                    &mut requirements_ptr,
                )
            },
            ErrorCause::GetCompiledModelInputBufferRequirements
        );
        Ok(TensorBufferRequirements::new(requirements_ptr))
    }

    /// Queries the buffer requirements for a specific output tensor in a signature.
    ///
    /// # Arguments
    ///
    /// * `signature_index` - The index of the model signature.
    /// * `output_index` - The zero-based index of the output tensor within the signature.
    ///
    /// # Returns
    ///
    /// * `Ok(TensorBufferRequirements)` - The buffer requirements (supported buffer types, size,
    ///   alignment) for the specified output tensor.
    /// * `Err(Error)` - If the signature or output index is invalid or querying requirements fails.
    fn output_buffer_requirements(
        &self,
        signature_index: SignatureIndex,
        output_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let mut requirements_ptr: *mut LiteRtTensorBufferRequirementsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the
            // create() function.
            unsafe {
                LiteRtGetCompiledModelOutputBufferRequirements(
                    self.raw_compiled_model,
                    signature_index.value(),
                    output_index,
                    &mut requirements_ptr,
                )
            },
            ErrorCause::GetCompiledModelOutputBufferRequirements
        );
        Ok(TensorBufferRequirements::new(requirements_ptr))
    }

    /// Creates a set of managed input tensor buffers for all inputs of the specified signature.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to allocate the managed buffers.
    /// * `model` - The [`Model`] from which this compiled model was created.
    /// * `signature_index` - The index of the model signature for which to allocate input buffers.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TensorBuffer>)` - A vector of allocated input [`TensorBuffer`]s ordered by the
    ///   signature's input indices.
    /// * `Err(Error)` - If the signature index is invalid, buffer requirements cannot be queried,
    ///   or buffer allocation fails.
    pub fn create_input_tensor_buffers(
        &self,
        environment: &Environment,
        model: &Model<'_>,
        signature_index: SignatureIndex,
    ) -> Result<Vec<TensorBuffer<'_>>, Error> {
        let signature = model.signature(signature_index)?;
        let num_inputs = signature.num_inputs()?;
        let mut result = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let input_requirements = self.input_buffer_requirements(signature_index, i)?;
            let tensor = signature.input_tensor_by_index(i)?;
            let buffer =
                CompiledModel::create_buffer_impl(environment, &input_requirements, &tensor)?;
            result.push(buffer);
        }
        Ok(result)
    }

    /// Creates a set of managed output tensor buffers for all outputs of the specified signature.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to allocate the managed buffers.
    /// * `model` - The [`Model`] from which this compiled model was created.
    /// * `signature_index` - The index of the model signature for which to allocate output buffers.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TensorBuffer>)` - A vector of allocated output [`TensorBuffer`]s ordered by the
    ///   signature's output indices.
    /// * `Err(Error)` - If the signature index is invalid, buffer requirements cannot be queried,
    ///   or buffer allocation fails.
    pub fn create_output_tensor_buffers(
        &self,
        environment: &Environment,
        model: &Model<'_>,
        signature_index: SignatureIndex,
    ) -> Result<Vec<TensorBuffer<'_>>, Error> {
        let signature = model.signature(signature_index)?;
        let num_outputs = signature.num_outputs()?;
        let mut result = Vec::with_capacity(num_outputs);
        for i in 0..num_outputs {
            let output_requirements = self.output_buffer_requirements(signature_index, i)?;
            let tensor = signature.output_tensor_by_index(i)?;
            let buffer =
                CompiledModel::create_buffer_impl(environment, &output_requirements, &tensor)?;
            result.push(buffer);
        }
        Ok(result)
    }

    /// Allocates a single managed [`TensorBuffer`] given its requirements and tensor metadata.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to allocate the managed buffer.
    /// * `requirements` - The [`TensorBufferRequirements`] specifying supported buffer types and
    ///   byte size.
    /// * `tensor` - The model [`Tensor`] providing shape/layout and element type metadata.
    ///
    /// # Returns
    ///
    /// * `Ok(TensorBuffer)` - The newly allocated [`TensorBuffer`].
    /// * `Err(Error)` - If no supported buffer types exist, tensor metadata cannot be retrieved, or
    ///   buffer creation fails.
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
        TensorBuffer::new(environment, &tensor_type, buffer_type, buffer_size, element_type)
    }

    /// Resizes the specified input tensor to support dynamic shapes at runtime.
    ///
    /// After resizing, previously obtained buffer requirements may be invalidated and callers
    /// should re-create or re-query buffer requirements if needed.
    ///
    /// # Arguments
    ///
    /// * `signature_index` - The index of the model signature containing the input tensor.
    /// * `input_index` - The zero-based index of the input tensor within the signature (or a type
    ///   convertible into [`InputOutputIndex`]).
    /// * `dimensions` - A slice of new dimension sizes for the input tensor.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the input tensor was successfully resized.
    /// * `Err(Error)` - If the parameters are invalid, dynamic shapes are unsupported, or resizing
    ///   fails.
    pub fn resize_input_tensor_buffer(
        &self,
        signature_index: SignatureIndex,
        input_index: impl Into<InputOutputIndex>,
        dimensions: &[i32],
    ) -> Result<(), Error> {
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the
            // create() function. dimensions is a valid slice provided by the caller.
            unsafe {
                LiteRtCompiledModelResizeInputTensor(
                    self.raw_compiled_model,
                    signature_index.value(),
                    input_index.into().value(),
                    dimensions.as_ptr(),
                    dimensions.len(),
                )
            },
            ErrorCause::ResizeCompiledModelInputBuffer
        );
        Ok(())
    }

    /// Runs inference synchronously on the compiled model for the specified signature.
    ///
    /// # Arguments
    ///
    /// * `signature_index` - The index of the model signature to execute.
    /// * `input` - A slice of input [`TensorBuffer`]s populated with input data, ordered by the
    ///   signature's inputs.
    /// * `output` - A slice of output [`TensorBuffer`]s to receive the inference results, ordered
    ///   by the signature's outputs.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If inference completed successfully.
    /// * `Err(Error)` - If model execution fails.
    pub fn run(
        &self,
        signature_index: SignatureIndex,
        input: &[TensorBuffer<'_>],
        output: &[TensorBuffer<'_>],
    ) -> Result<(), Error> {
        let mut input_ptrs: Vec<_> = input.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        let mut output_ptrs: Vec<_> =
            output.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the
            // create() function. input_ptrs and output_ptrs are valid because they are
            // created in the function.
            unsafe {
                LiteRtRunCompiledModel(
                    self.raw_compiled_model,
                    signature_index.value(),
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

    /// Runs inference asynchronously if supported by the model and hardware accelerator.
    ///
    /// If asynchronous execution is supported, parts or all of the model may execute in parallel
    /// and synchronization events will be attached to the output tensor buffers (use
    /// [`TensorBuffer::read_async`] to await and read results). Otherwise, the model executes
    /// synchronously before returning.
    ///
    /// # Arguments
    ///
    /// * `signature_index` - The index of the model signature to execute.
    /// * `input` - A slice of input [`TensorBuffer`]s populated with input data.
    /// * `output` - A slice of output [`TensorBuffer`]s to receive the inference results.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If inference was scheduled to run asynchronously.
    /// * `Ok(false)` - If inference was executed synchronously.
    /// * `Err(Error)` - If model execution fails.
    #[cfg(async_support)]
    pub fn run_async(
        &self,
        signature_index: SignatureIndex,
        input: &[TensorBuffer<'_>],
        output: &[TensorBuffer<'_>],
    ) -> Result<bool, Error> {
        let mut input_ptrs: Vec<_> = input.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        let mut output_ptrs: Vec<_> =
            output.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        let mut ran_in_parallel_thread: bool = false;
        call_check_status!(
            // SAFETY: self.raw_compiled_model is valid because it's created by calling the
            // create() function. input_ptrs, output_ptrs, and ran_in_parallel_thread
            // are valid because they are created in the function.
            unsafe {
                LiteRtRunCompiledModelAsync(
                    self.raw_compiled_model,
                    signature_index.value(),
                    input_ptrs.len(),
                    input_ptrs.as_mut_ptr(),
                    output_ptrs.len(),
                    output_ptrs.as_mut_ptr(),
                    &mut ran_in_parallel_thread,
                )
            },
            ErrorCause::RunCompiledModel
        );
        Ok(ran_in_parallel_thread)
    }

    /// Returns `true` if the model is fully accelerated for all the selected HW accelerator(s).
    ///
    /// For example, if both GPU and NPU are selected and the model is only delegated to GPU,
    /// this method will still return `true`.
    ///
    /// # Returns
    ///
    /// * `Ok(bool)` - `true` if the model is fully accelerated across the selected hardware
    ///   accelerators, `false` otherwise.
    /// * `Err(Error)` - If querying the acceleration status fails.
    pub fn is_fully_accelerated(&self) -> Result<bool, Error> {
        let mut fully_accelerated = false;
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is guaranteed non-null and valid for the lifetime of
            //   `self`, managed by `CompiledModel` and only destroyed in `Drop`.
            // - `&mut fully_accelerated` is a valid, aligned, non-null pointer to local stack
            //   memory with exclusive write permission for a single `bool`.
            unsafe {
                LiteRtCompiledModelIsFullyAccelerated(
                    self.raw_compiled_model,
                    &mut fully_accelerated,
                )
            },
            ErrorCause::IsCompiledModelFullyAccelerated
        );
        Ok(fully_accelerated)
    }

    /// Returns `true` if the model is fully accelerated on non-CPU accelerators
    /// (e.g. 100% delegated on GPU or NPU with zero CPU execution).
    ///
    /// # Returns
    ///
    /// * `Ok(bool)` - `true` if 100% of the model is delegated to non-CPU hardware accelerators,
    ///   `false` otherwise.
    /// * `Err(Error)` - If querying the non-CPU acceleration status fails.
    pub fn is_non_cpu_fully_accelerated(&self) -> Result<bool, Error> {
        let mut non_cpu_fully_accelerated = false;
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is guaranteed non-null and valid for the lifetime of
            //   `self`, managed by `CompiledModel` and only destroyed in `Drop`.
            // - `&mut non_cpu_fully_accelerated` is a valid, aligned, non-null pointer to local
            //   stack memory with exclusive write permission for a single `bool`.
            unsafe {
                LiteRtCompiledModelIsNonCpuFullyAccelerated(
                    self.raw_compiled_model,
                    &mut non_cpu_fully_accelerated,
                )
            },
            ErrorCause::IsCompiledModelNonCpuFullyAccelerated
        );
        Ok(non_cpu_fully_accelerated)
    }
}

impl Drop for CompiledModel {
    fn drop(&mut self) {
        // SAFETY: self.raw_compiled_model is always valid, it's guaranteed to be initialized by
        // create function.
        unsafe {
            LiteRtDestroyCompiledModel(self.raw_compiled_model);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accelerator_to_c_enum() {
        assert_eq!(
            LiteRtHwAccelerator::None.to_c_enum(),
            LiteRtHwAccelerators_kLiteRtHwAcceleratorNone
        );
        assert_eq!(
            LiteRtHwAccelerator::Cpu.to_c_enum(),
            LiteRtHwAccelerators_kLiteRtHwAcceleratorCpu
        );
        assert_eq!(
            LiteRtHwAccelerator::Gpu.to_c_enum(),
            LiteRtHwAccelerators_kLiteRtHwAcceleratorGpu
        );
        assert_eq!(
            LiteRtHwAccelerator::Npu.to_c_enum(),
            LiteRtHwAccelerators_kLiteRtHwAcceleratorNpu
        );
    }

    #[test]
    fn test_options_create_with_accelerator() {
        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu);
        assert!(options.is_ok());
    }

    #[test]
    fn test_options_default() {
        let options = Options::default();
        assert!(options.is_ok());
    }

    #[test]
    fn test_compiled_model_end_to_end() {
        let test_paths = [
            "third_party/odml/litert/litert/test/testdata/simple_model.tflite",
            "../test/testdata/simple_model.tflite",
        ];
        let Some(&model_path) = test_paths.iter().find(|p| std::path::Path::new(p).exists()) else {
            return;
        };

        let env = crate::environment::EnvironmentBuilder::build_default()
            .expect("Failed to create environment");
        let model =
            Model::create_model_from_file(&env, model_path).expect("Failed to load test model");
        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
            .expect("Failed to create options");
        let compiled_model =
            CompiledModel::create(&env, &model, &options).expect("Failed to compile model");

        assert!(compiled_model.is_fully_accelerated().is_ok());
        assert!(compiled_model.is_non_cpu_fully_accelerated().is_ok());

        let sig_index = SignatureIndex::DEFAULT;
        let inputs = compiled_model
            .create_input_tensor_buffers(&env, &model, sig_index)
            .expect("Failed to create input buffers");
        let outputs = compiled_model
            .create_output_tensor_buffers(&env, &model, sig_index)
            .expect("Failed to create output buffers");

        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        inputs[0].write(&[1.0f32, 2.0f32]).expect("Failed to write input 0");
        inputs[1].write(&[10.0f32, 20.0f32]).expect("Failed to write input 1");

        compiled_model.run(sig_index, &inputs, &outputs).expect("Failed to run compiled model");

        let mut output_data = [0.0f32; 2];
        outputs[0].read(&mut output_data).expect("Failed to read output 0");
        assert!((output_data[0] - 11.0).abs() < 1e-4);
        assert!((output_data[1] - 22.0).abs() < 1e-4);
    }
}
