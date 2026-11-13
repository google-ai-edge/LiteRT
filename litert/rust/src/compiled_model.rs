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
use crate::tensor_buffer::{ElementType, TensorBuffer, TensorBufferRequirements};
use std::sync::{Mutex, MutexGuard};

/// Options for compiling a model.
pub struct Options {
    /// Underlying C `LiteRtOptions` handle owned by this struct.
    raw_options: LiteRtOptions,
}

/// Hardware accelerators that can be used for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteRtHwAccelerator {
    /// No hardware acceleration.
    None,
    /// CPU execution.
    Cpu,
    /// GPU acceleration.
    Gpu,
    /// NPU (Neural Processing Unit) acceleration.
    Npu,
}

impl LiteRtHwAccelerator {
    /// Converts the hardware accelerator enum variant into its corresponding `LiteRT` C enum value.
    ///
    /// # Returns
    ///
    /// The `LiteRtHwAccelerators` C enum constant matching `self`.
    #[allow(clippy::wrong_self_convention)]
    #[must_use]
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
    /// A newly initialized [`Options`] instance with default settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::CreateOptions`) if the underlying `LiteRT` C API fails to
    /// create the options object.
    #[allow(clippy::should_implement_trait)]
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
    /// * `accelerator` - The hardware accelerator ([`LiteRtHwAccelerator`]) to target during
    ///   compilation and inference.
    ///
    /// # Returns
    ///
    /// A newly initialized [`Options`] instance configured for `accelerator`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if creating the options or setting the hardware accelerator fails.
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
    /// Underlying C `LiteRtCompiledModel` handle owned by this struct.
    pub(crate) raw_compiled_model: LiteRtCompiledModel,
    /// Serializes all C API operations on `raw_compiled_model` so `CompiledModel` is `Sync`
    /// (`LiteRtCompiledModelT` in C++ owns a single `tflite::Interpreter` and unguarded caches).
    mutex: Mutex<()>,
}

impl CompiledModel {
    /// Acquires the internal mutex serializing C API access to `self.raw_compiled_model`.
    pub(crate) fn lock_model(&self) -> MutexGuard<'_, ()> {
        self.mutex.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Compiles a [`Model`] within the given [`Environment`] using the specified [`Options`].
    ///
    /// Takes `model: &mut Model<'_>` because the underlying `LiteRT` C++ runtime
    /// (`LiteRtCreateCompiledModel`) may mutate `LiteRtModelT` in place during compilation
    /// (for example, resolving external weights, replacing magic numbers, or applying compiler
    /// plugins). Requiring `&mut Model<'_>` allows all read-only inspection methods on [`Model`]
    /// (`signatures`, `subgraphs`, etc.) to remain lock-free while statically preventing data races
    /// between compilation and concurrent model inspection.
    ///
    /// # Arguments
    ///
    /// * `environment` - The `LiteRT` [`Environment`] used to manage runtime resources and
    ///   accelerators. Must outlive the `CompiledModel` and any executions running on it.
    /// * `model` - Mutable reference to the [`Model`] to compile. Must outlive the `CompiledModel`.
    /// * `options` - Compilation [`Options`] specifying hardware accelerators and other settings.
    ///
    /// # Returns
    ///
    /// The compiled [`CompiledModel`] ready for buffer creation and inference.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::CreateCompiledModel`) if model compilation fails in the
    /// underlying `LiteRT` runtime.
    pub fn create(
        environment: &Environment,
        model: &mut Model<'_>,
        options: &Options,
    ) -> Result<Self, Error> {
        let mut raw_compiled_model_ptr: *mut LiteRtCompiledModelT = std::ptr::null_mut();
        let _env_guard = environment.lock_env();
        call_check_status!(
            // SAFETY:
            // - `environment.raw_environment`, `model.raw_model`, and `options.raw_options` are
            //   valid, non-null handles owned by their respective wrapper structs.
            // - `_env_guard` serializes C API operations on `environment.raw_environment`, and
            //   `model: &mut Model<'_>` guarantees exclusive access to `model.raw_model` during
            //   compilation, while `LiteRtCreateCompiledModel` in C++ also holds a process-wide
            //   `compiled_model_create_mutex`.
            // - `&mut raw_compiled_model_ptr` is a valid, aligned, non-null stack out-pointer.
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
        Ok(CompiledModel { raw_compiled_model: raw_compiled_model_ptr, mutex: Mutex::new(()) })
    }

    /// Returns `true` if this compiled model was created with `environment`.
    pub(crate) fn has_same_environment(&self, environment: &Environment) -> Result<bool, Error> {
        let mut compiled_env: LiteRtEnvironment = std::ptr::null_mut();
        let _guard = self.lock_model();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null `LiteRtCompiledModel` handle owned
            //   by `self`, and `_guard` serializes concurrent C API calls on the compiled model.
            // - `&mut compiled_env` is a valid, aligned, non-null stack out-pointer.
            unsafe {
                LiteRtGetCompiledModelEnvironment(self.raw_compiled_model, &mut compiled_env)
            },
            ErrorCause::GetCompiledModelEnvironment
        );
        Ok(compiled_env == environment.raw_environment)
    }

    /// Queries the current runtime layout of the input tensor at `input_index` in
    /// `signature_index`.
    #[allow(dead_code)]
    pub(crate) fn input_tensor_layout(
        &self,
        signature_index: SignatureIndex,
        input_index: LiteRtParamIndex,
    ) -> Result<LiteRtLayout, Error> {
        let _guard = self.lock_model();
        self.input_tensor_layout_locked(signature_index, input_index)
    }

    /// Queries the current runtime layout of the input tensor at `input_index` in
    /// `signature_index` while the caller already holds `self.lock_model()`.
    pub(crate) fn input_tensor_layout_locked(
        &self,
        signature_index: SignatureIndex,
        input_index: LiteRtParamIndex,
    ) -> Result<LiteRtLayout, Error> {
        let mut runtime_layout = LiteRtLayout::default();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null `LiteRtCompiledModel` handle owned
            //   by `self`, and the caller holds `self.lock_model()`.
            // - `&mut runtime_layout` is a valid, aligned, non-null stack out-pointer.
            unsafe {
                LiteRtGetCompiledModelInputTensorLayout(
                    self.raw_compiled_model,
                    signature_index.value(),
                    input_index,
                    &mut runtime_layout,
                )
            },
            ErrorCause::ResizeCompiledModelInputBuffer
        );
        Ok(runtime_layout)
    }

    /// Queries the current runtime layouts of all `num_outputs` output tensors in
    /// `signature_index` (updating tensor allocations in the underlying interpreter).
    #[allow(dead_code)]
    pub(crate) fn output_tensor_layouts(
        &self,
        signature_index: SignatureIndex,
        num_outputs: usize,
    ) -> Result<Vec<LiteRtLayout>, Error> {
        let _guard = self.lock_model();
        self.output_tensor_layouts_locked(signature_index, num_outputs)
    }

    /// Queries the current runtime layouts of all `num_outputs` output tensors in
    /// `signature_index` while the caller already holds `self.lock_model()`.
    pub(crate) fn output_tensor_layouts_locked(
        &self,
        signature_index: SignatureIndex,
        num_outputs: usize,
    ) -> Result<Vec<LiteRtLayout>, Error> {
        if num_outputs == 0 {
            return Ok(Vec::new());
        }
        let mut output_layouts = vec![LiteRtLayout::default(); num_outputs];
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null `LiteRtCompiledModel` handle owned
            //   by `self`, and the caller holds `self.lock_model()`.
            // - `output_layouts` is a pre-allocated mutable slice of `num_outputs > 0` initialized
            //   `LiteRtLayout` structs, so `output_layouts.as_mut_ptr()` is non-null, aligned, and
            //   valid for writing `num_outputs` elements.
            unsafe {
                LiteRtGetCompiledModelOutputTensorLayouts(
                    self.raw_compiled_model,
                    signature_index.value(),
                    num_outputs,
                    output_layouts.as_mut_ptr(),
                    true,
                )
            },
            ErrorCause::ResizeCompiledModelInputBuffer
        );
        Ok(output_layouts)
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
    #[allow(dead_code)]
    pub(crate) fn input_buffer_requirements(
        &self,
        signature_index: SignatureIndex,
        input_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let _guard = self.lock_model();
        self.input_buffer_requirements_locked(signature_index, input_index)
    }

    /// Queries the buffer requirements for `input_index` while the caller already holds
    /// `self.lock_model()`.
    pub(crate) fn input_buffer_requirements_locked(
        &self,
        signature_index: SignatureIndex,
        input_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let mut requirements_ptr: *mut LiteRtTensorBufferRequirementsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of `self`,
            //   and the caller holds `self.lock_model()`.
            // - `&mut requirements_ptr` is a valid, aligned, non-null stack out-pointer.
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
        // SAFETY: `requirements_ptr` is a valid handle returned by
        // `LiteRtGetCompiledModelInputBufferRequirements`, and the caller holds `self.lock_model()`
        // while `from_raw` eagerly copies the buffer size and supported buffer types.
        unsafe { TensorBufferRequirements::from_raw(requirements_ptr) }
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
    #[allow(dead_code)]
    pub(crate) fn output_buffer_requirements(
        &self,
        signature_index: SignatureIndex,
        output_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let _guard = self.lock_model();
        self.output_buffer_requirements_locked(signature_index, output_index)
    }

    /// Queries the buffer requirements for `output_index` while the caller already holds
    /// `self.lock_model()`.
    pub(crate) fn output_buffer_requirements_locked(
        &self,
        signature_index: SignatureIndex,
        output_index: LiteRtParamIndex,
    ) -> Result<TensorBufferRequirements<'_>, Error> {
        let mut requirements_ptr: *mut LiteRtTensorBufferRequirementsT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of `self`,
            //   and the caller holds `self.lock_model()`.
            // - `&mut requirements_ptr` is a valid, aligned, non-null stack out-pointer.
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
        // SAFETY: `requirements_ptr` is a valid handle returned by
        // `LiteRtGetCompiledModelOutputBufferRequirements`, and the caller holds
        // `self.lock_model()` while `from_raw` eagerly copies the buffer size and supported buffer
        // types.
        unsafe { TensorBufferRequirements::from_raw(requirements_ptr) }
    }

    /// Creates a set of managed input tensor buffers for all inputs of the specified signature.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to allocate the managed buffers. Must be the same
    ///   environment that created `self` and must outlive the returned buffers.
    /// * `model` - The [`Model`] from which this compiled model was created.
    /// * `signature_index` - The index of the model signature for which to allocate input buffers.
    ///
    /// # Returns
    ///
    /// A vector of allocated input [`TensorBuffer`]s ordered by the signature's input indices.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `environment` does not match `self`'s environment, the signature index
    /// is invalid, buffer requirements cannot be queried, or buffer allocation fails.
    pub fn create_input_tensor_buffers<'a>(
        &'a self,
        environment: &'a Environment,
        model: &Model<'_>,
        signature_index: SignatureIndex,
    ) -> Result<Vec<TensorBuffer<'a>>, Error> {
        if !self.has_same_environment(environment)? {
            return Err(Error::new(
                ErrorCause::CreateManagedTensorBuffer,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        let signature = model.signature(signature_index)?;
        let num_inputs = signature.num_inputs()?;
        let _guard = self.lock_model();
        let mut result = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let tensor = signature.input_tensor_by_index(i)?;
            let buffer = self.create_input_buffer_for_tensor_locked(
                environment,
                signature_index,
                i,
                &tensor,
            )?;
            result.push(buffer);
        }
        Ok(result)
    }

    /// Creates a set of managed output tensor buffers for all outputs of the specified signature.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to allocate the managed buffers. Must be the same
    ///   environment that created `self` and must outlive the returned buffers.
    /// * `model` - The [`Model`] from which this compiled model was created.
    /// * `signature_index` - The index of the model signature for which to allocate output buffers.
    ///
    /// # Returns
    ///
    /// A vector of allocated output [`TensorBuffer`]s ordered by the signature's output indices.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `environment` does not match `self`'s environment, the signature index
    /// is invalid, buffer requirements cannot be queried, or buffer allocation fails.
    pub fn create_output_tensor_buffers<'a>(
        &'a self,
        environment: &'a Environment,
        model: &Model<'_>,
        signature_index: SignatureIndex,
    ) -> Result<Vec<TensorBuffer<'a>>, Error> {
        if !self.has_same_environment(environment)? {
            return Err(Error::new(
                ErrorCause::CreateManagedTensorBuffer,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        let signature = model.signature(signature_index)?;
        let num_outputs = signature.num_outputs()?;
        let _guard = self.lock_model();
        let mut runtime_layouts: Option<Vec<LiteRtLayout>> = None;
        let mut result = Vec::with_capacity(num_outputs);
        for i in 0..num_outputs {
            let tensor = signature.output_tensor_by_index(i)?;
            let buffer = self.create_output_buffer_for_tensor_locked(
                environment,
                signature_index,
                i,
                num_outputs,
                &tensor,
                &mut runtime_layouts,
            )?;
            result.push(buffer);
        }
        Ok(result)
    }

    /// Allocates a managed input [`TensorBuffer`] for `input_index` while the caller already holds
    /// `self.lock_model()`, querying the compiled model's runtime input layout if the static tensor
    /// shape contains dynamic (`< 0`) dimensions.
    pub(crate) fn create_input_buffer_for_tensor_locked<'a>(
        &self,
        environment: &Environment,
        signature_index: SignatureIndex,
        input_index: LiteRtParamIndex,
        tensor: &Tensor<'_>,
    ) -> Result<TensorBuffer<'a>, Error> {
        let mut tensor_type = tensor.ranked_tensor_type()?;
        let rank = usize::try_from(tensor_type.layout.rank())
            .unwrap_or(0)
            .min(tensor_type.layout.dimensions.len());
        if tensor_type.layout.dimensions[..rank].iter().any(|&d| d < 0) {
            let mut runtime_layout = tensor_type.layout;
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of `self`,
            //   and the caller holds `self.lock_model()`.
            // - `&mut runtime_layout` is a valid, aligned, non-null stack out-pointer.
            let status = unsafe {
                LiteRtGetCompiledModelInputTensorLayout(
                    self.raw_compiled_model,
                    signature_index.value(),
                    input_index,
                    &mut runtime_layout,
                )
            };
            if status == LiteRtStatus_kLiteRtStatusOk {
                tensor_type.layout = runtime_layout;
            }
            let new_rank = usize::try_from(tensor_type.layout.rank())
                .unwrap_or(0)
                .min(tensor_type.layout.dimensions.len());
            for dim in &mut tensor_type.layout.dimensions[..new_rank] {
                if *dim < 0 {
                    *dim = 1;
                }
            }
        }
        let requirements = self.input_buffer_requirements_locked(signature_index, input_index)?;
        let element_type = tensor.element_type()?;
        Self::create_buffer_with_type(environment, &requirements, &tensor_type, element_type)
    }

    /// Allocates a managed output [`TensorBuffer`] for `output_index` while the caller already
    /// holds `self.lock_model()`, lazily querying and caching the compiled model's runtime output
    /// layouts if the static tensor shape contains dynamic (`< 0`) dimensions.
    pub(crate) fn create_output_buffer_for_tensor_locked<'a>(
        &self,
        environment: &Environment,
        signature_index: SignatureIndex,
        output_index: LiteRtParamIndex,
        num_outputs: usize,
        tensor: &Tensor<'_>,
        cached_layouts: &mut Option<Vec<LiteRtLayout>>,
    ) -> Result<TensorBuffer<'a>, Error> {
        let mut tensor_type = tensor.ranked_tensor_type()?;
        let rank = usize::try_from(tensor_type.layout.rank())
            .unwrap_or(0)
            .min(tensor_type.layout.dimensions.len());
        if tensor_type.layout.dimensions[..rank].iter().any(|&d| d < 0) {
            if cached_layouts.is_none() && num_outputs > 0 {
                let mut layouts = vec![LiteRtLayout::default(); num_outputs];
                // SAFETY:
                // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of
                //   `self`, and the caller holds `self.lock_model()`.
                // - `layouts` has `num_outputs > 0` initialized `LiteRtLayout` elements, so
                //   `layouts.as_mut_ptr()` is non-null, aligned, and valid for writing
                //   `num_outputs` elements.
                let status = unsafe {
                    LiteRtGetCompiledModelOutputTensorLayouts(
                        self.raw_compiled_model,
                        signature_index.value(),
                        num_outputs,
                        layouts.as_mut_ptr(),
                        true,
                    )
                };
                if status == LiteRtStatus_kLiteRtStatusOk {
                    *cached_layouts = Some(layouts);
                }
            }
            if let Some(layouts) = cached_layouts.as_ref() {
                if let Some(&runtime_layout) = layouts.get(output_index) {
                    tensor_type.layout = runtime_layout;
                }
            }
            let new_rank = usize::try_from(tensor_type.layout.rank())
                .unwrap_or(0)
                .min(tensor_type.layout.dimensions.len());
            for dim in &mut tensor_type.layout.dimensions[..new_rank] {
                if *dim < 0 {
                    *dim = 1;
                }
            }
        }
        let requirements = self.output_buffer_requirements_locked(signature_index, output_index)?;
        let element_type = tensor.element_type()?;
        Self::create_buffer_with_type(environment, &requirements, &tensor_type, element_type)
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
    #[allow(dead_code)]
    pub(crate) fn create_buffer_impl<'a>(
        environment: &Environment,
        requirements: &TensorBufferRequirements,
        tensor: &Tensor,
    ) -> Result<TensorBuffer<'a>, Error> {
        let tensor_type = tensor.ranked_tensor_type()?;
        let element_type = tensor.element_type()?;
        Self::create_buffer_with_type(environment, requirements, &tensor_type, element_type)
    }

    /// Allocates a managed [`TensorBuffer`] using the first supported buffer type from
    /// `requirements` and the provided ranked tensor type and element type.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to allocate the managed buffer.
    /// * `requirements` - The [`TensorBufferRequirements`] specifying supported buffer types and
    ///   byte size.
    /// * `tensor_type` - The resolved `LiteRtRankedTensorType` describing layout and element type.
    /// * `element_type` - The [`ElementType`] of the tensor buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(TensorBuffer)` - The newly allocated [`TensorBuffer`].
    /// * `Err(Error)` - If `requirements` supports no buffer types or buffer creation fails.
    pub(crate) fn create_buffer_with_type<'a>(
        environment: &Environment,
        requirements: &TensorBufferRequirements,
        tensor_type: &LiteRtRankedTensorType,
        element_type: ElementType,
    ) -> Result<TensorBuffer<'a>, Error> {
        let supported_types = requirements.supported_types()?;
        // For simplicity we just pick the first supported tensor buffer type.
        let Some(buffer_type) = supported_types.first() else {
            return Err(Error::new(
                ErrorCause::InputDoesntSupportAnyTensorBufferTypes,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        };
        let buffer_size = requirements.buffer_size()?;
        TensorBuffer::new(environment, tensor_type, buffer_type, buffer_size, element_type)
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
    /// `Ok(())` if the input tensor was successfully resized.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::ResizeCompiledModelInputBuffer`) if the parameters are
    /// invalid, dynamic shapes are unsupported, or resizing fails.
    pub fn resize_input_tensor_buffer(
        &self,
        signature_index: SignatureIndex,
        input_index: impl Into<InputOutputIndex>,
        dimensions: &[i32],
    ) -> Result<(), Error> {
        let _guard = self.lock_model();
        self.resize_input_tensor_buffer_locked(signature_index, input_index, dimensions)
    }

    /// Resizes the specified input tensor while the caller already holds `self.lock_model()`.
    pub(crate) fn resize_input_tensor_buffer_locked(
        &self,
        signature_index: SignatureIndex,
        input_index: impl Into<InputOutputIndex>,
        dimensions: &[i32],
    ) -> Result<(), Error> {
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of `self`,
            //   and the caller holds `self.lock_model()`.
            // - `dimensions.as_ptr()` is a valid pointer to `dimensions.len()` `i32` elements.
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
    /// `Ok(())` if inference completed successfully.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::RunCompiledModel`) if model execution fails, or if any
    /// participating input or output buffer currently has an active host-memory lock guard or is in
    /// use by another concurrent run.
    pub fn run(
        &self,
        signature_index: SignatureIndex,
        input: &[TensorBuffer<'_>],
        output: &[TensorBuffer<'_>],
    ) -> Result<(), Error> {
        self.run_with_iter(signature_index, input.iter(), output.iter())
    }

    /// Runs inference synchronously on the compiled model using iterators of borrowed
    /// [`TensorBuffer`] references.
    ///
    /// Acquires `self.lock_model()` first and then acquires a non-blocking
    /// [`crate::tensor_buffer::TensorBufferRunGuard`] on each unique participating input and output
    /// buffer before invoking the C API so that:
    /// 1. Calling `run` while any [`crate::tensor_buffer_lock::TensorBufferReadLock`] or
    ///    [`crate::tensor_buffer_lock::TensorBufferWriteLock`] guard (or another concurrent model's
    ///    `run`) is using an input or output buffer fails immediately with
    ///    [`ErrorCause::RunCompiledModel`] rather than racing on `LiteRtTensorBufferT::is_locked_`
    ///    or mutating memory behind an active `&[T]`/`&mut [T]` slice.
    /// 2. Concurrent calls to [`TensorBuffer::lock_read`] or [`TensorBuffer::lock_write`] on
    ///    another thread while `LiteRtRunCompiledModel` is executing are rejected until `run`
    ///    finishes and drops its run guards.
    pub(crate) fn run_with_iter<'a, 'b>(
        &self,
        signature_index: SignatureIndex,
        input: impl IntoIterator<Item = &'a TensorBuffer<'a>>,
        output: impl IntoIterator<Item = &'b TensorBuffer<'b>>,
    ) -> Result<(), Error> {
        let input_refs: Vec<&'a TensorBuffer<'a>> = input.into_iter().collect();
        let output_refs: Vec<&'b TensorBuffer<'b>> = output.into_iter().collect();
        let _guard = self.lock_model();
        let mut _run_guards = Vec::with_capacity(input_refs.len() + output_refs.len());
        let mut seen_raw_ptrs = Vec::with_capacity(input_refs.len() + output_refs.len());
        for &buf in &input_refs {
            if !seen_raw_ptrs.contains(&buf.raw_tensor_buffer) {
                seen_raw_ptrs.push(buf.raw_tensor_buffer);
                _run_guards.push(buf.acquire_for_run()?);
            }
        }
        for &buf in &output_refs {
            if !seen_raw_ptrs.contains(&buf.raw_tensor_buffer) {
                seen_raw_ptrs.push(buf.raw_tensor_buffer);
                _run_guards.push(buf.acquire_for_run()?);
            }
        }
        let mut input_ptrs: Vec<_> =
            input_refs.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        let mut output_ptrs: Vec<_> =
            output_refs.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of `self`,
            //   and `_guard` serializes concurrent C API calls on the compiled model.
            // - `_run_guards` holds exclusive `BufferAccessState::InUseByRun` on every unique
            //   buffer in `input_refs` and `output_refs` for the entire duration of
            //   `LiteRtRunCompiledModel`, guaranteeing that no `TensorBufferReadLock`,
            //   `TensorBufferWriteLock`, or concurrent `CompiledModel::run` on another model can
            //   access any participating buffer during execution, and preventing any concurrent
            //   access to `LiteRtTensorBufferT::is_locked_` in C++.
            // - `LiteRtCompiledModelT::Run` in C++ (`litert/runtime/compiled_model.cc`) explicitly
            //   deduplicates repeated `LiteRtTensorBuffer` pointers in `locked_buffers`.
            // - `input_ptrs` and `output_ptrs` are valid contiguous arrays of live
            //   `LiteRtTensorBuffer` handles borrowed from `input_refs` and `output_refs`.
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
    /// In the underlying `LiteRT` C++ runtime (`LiteRtCompiledModelT::Run`), buffer registration,
    /// tensor allocation, and `tflite::Interpreter::Invoke()` all execute synchronously on the
    /// calling thread while `self`'s internal model lock is held; when asynchronous execution is
    /// supported by a hardware accelerator delegate (e.g. GPU or NPU), the delegate attaches a
    /// synchronization event (`LiteRtEvent`, such as a sync fence) to the output tensor buffers
    /// instead of blocking on hardware queue completion at the end of `Invoke()` (use
    /// [`TensorBuffer::read_async`] to await and read results). Otherwise, the model executes
    /// completely synchronously before returning.
    ///
    /// # Arguments
    ///
    /// * `signature_index` - The index of the model signature to execute.
    /// * `input` - A slice of input [`TensorBuffer`]s populated with input data.
    /// * `output` - A slice of output [`TensorBuffer`]s to receive the inference results.
    ///
    /// # Returns
    ///
    /// `true` if synchronization events were attached to the output buffers for asynchronous
    /// hardware execution, or `false` if inference completed synchronously.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::RunCompiledModel`) if model execution fails, or if any
    /// participating input or output buffer currently has an active host-memory lock guard or is in
    /// use by another concurrent run.
    #[cfg(async_support)]
    pub fn run_async(
        &self,
        signature_index: SignatureIndex,
        input: &[TensorBuffer<'_>],
        output: &[TensorBuffer<'_>],
    ) -> Result<bool, Error> {
        self.run_async_with_iter(signature_index, input.iter(), output.iter())
    }

    /// Runs inference asynchronously on the compiled model using iterators of borrowed
    /// [`TensorBuffer`] references.
    #[cfg(async_support)]
    pub(crate) fn run_async_with_iter<'a, 'b>(
        &self,
        signature_index: SignatureIndex,
        input: impl IntoIterator<Item = &'a TensorBuffer<'a>>,
        output: impl IntoIterator<Item = &'b TensorBuffer<'b>>,
    ) -> Result<bool, Error> {
        let input_refs: Vec<&'a TensorBuffer<'a>> = input.into_iter().collect();
        let output_refs: Vec<&'b TensorBuffer<'b>> = output.into_iter().collect();
        let _guard = self.lock_model();
        let mut _run_guards = Vec::with_capacity(input_refs.len() + output_refs.len());
        let mut seen_raw_ptrs = Vec::with_capacity(input_refs.len() + output_refs.len());
        for &buf in &input_refs {
            if !seen_raw_ptrs.contains(&buf.raw_tensor_buffer) {
                seen_raw_ptrs.push(buf.raw_tensor_buffer);
                _run_guards.push(buf.acquire_for_run()?);
            }
        }
        for &buf in &output_refs {
            if !seen_raw_ptrs.contains(&buf.raw_tensor_buffer) {
                seen_raw_ptrs.push(buf.raw_tensor_buffer);
                _run_guards.push(buf.acquire_for_run()?);
            }
        }
        let mut input_ptrs: Vec<_> =
            input_refs.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        let mut output_ptrs: Vec<_> =
            output_refs.iter().map(|tensor| tensor.raw_tensor_buffer).collect();
        let mut has_pending_output_events: bool = false;
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is a valid, non-null handle for the lifetime of `self`,
            //   and `_guard` serializes concurrent C API calls on the compiled model.
            // - In C++ (`LiteRtCompiledModelT::Run`), buffer registration, `AllocateTensors`,
            //   `interp_->Invoke()`, and host-buffer unlocking all complete synchronously on the
            //   calling thread before `LiteRtRunCompiledModelAsync` returns; asynchronous
            //   accelerator delegates attach `LiteRtEvent` sync fences to the output buffers
            //   (which `LiteRtTensorBufferT::Lock` automatically waits on before mapping memory).
            // - `_run_guards` holds exclusive `BufferAccessState::InUseByRun` on every unique
            //   buffer in `input_refs` and `output_refs` for the duration of
            //   `LiteRtRunCompiledModelAsync`, preventing any concurrent `lock_read` /
            //   `lock_write` or concurrent `run` on another model from racing on
            //   `LiteRtTensorBufferT::is_locked_`.
            // - `input_ptrs`, `output_ptrs`, and `&mut has_pending_output_events` are valid
            //   pointers.
            unsafe {
                LiteRtRunCompiledModelAsync(
                    self.raw_compiled_model,
                    signature_index.value(),
                    input_ptrs.len(),
                    input_ptrs.as_mut_ptr(),
                    output_ptrs.len(),
                    output_ptrs.as_mut_ptr(),
                    &mut has_pending_output_events,
                )
            },
            ErrorCause::RunCompiledModel
        );
        Ok(has_pending_output_events)
    }

    /// Returns `true` if the model is fully accelerated for all the selected HW accelerator(s).
    ///
    /// For example, if both GPU and NPU are selected and the model is only delegated to GPU,
    /// this method will still return `true`.
    ///
    /// # Returns
    ///
    /// `true` if the model is fully accelerated across the selected hardware accelerators, `false`
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::IsCompiledModelFullyAccelerated`) if querying the
    /// acceleration status fails.
    pub fn is_fully_accelerated(&self) -> Result<bool, Error> {
        let mut fully_accelerated = false;
        let _guard = self.lock_model();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is guaranteed non-null and valid for the lifetime of
            //   `self`, managed by `CompiledModel` and only destroyed in `Drop`. `_guard`
            //   serializes concurrent C API calls on the compiled model.
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
    /// `true` if 100% of the model is delegated to non-CPU hardware accelerators, `false`
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::IsCompiledModelNonCpuFullyAccelerated`) if querying the
    /// non-CPU acceleration status fails.
    pub fn is_non_cpu_fully_accelerated(&self) -> Result<bool, Error> {
        let mut non_cpu_fully_accelerated = false;
        let _guard = self.lock_model();
        call_check_status!(
            // SAFETY:
            // - `self.raw_compiled_model` is guaranteed non-null and valid for the lifetime of
            //   `self`, managed by `CompiledModel` and only destroyed in `Drop`. `_guard`
            //   serializes concurrent C API calls on the compiled model.
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
        // SAFETY:
        // - `self.raw_compiled_model` is a valid, non-null handle initialized by
        //   `LiteRtCreateCompiledModel` and uniquely owned by `self`.
        unsafe {
            LiteRtDestroyCompiledModel(self.raw_compiled_model);
        }
    }
}

// SAFETY: `CompiledModel` owns its `LiteRtCompiledModel` handle and can be safely moved across
// threads.
unsafe impl Send for CompiledModel {}

// SAFETY: All C API operations on `self.raw_compiled_model` are serialized by `self.mutex`, making
// concurrent access across threads via `&CompiledModel` race-free.
unsafe impl Sync for CompiledModel {}

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
        let mut model =
            Model::create_model_from_file(&env, model_path).expect("Failed to load test model");
        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
            .expect("Failed to create options");
        let compiled_model =
            CompiledModel::create(&env, &mut model, &options).expect("Failed to compile model");

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

        // Attempting to run while an input or output buffer has an active host-memory lock guard
        // must fail cleanly with `ErrorCause::RunCompiledModel` rather than racing or aliasing.
        {
            let _out_read_guard = outputs[0].lock_read::<f32>().expect("Failed to lock output 0");
            let busy_err = compiled_model
                .run(sig_index, &inputs, &outputs)
                .err()
                .expect("Expected error when running with locked output buffer");
            assert_eq!(busy_err.cause(), ErrorCause::RunCompiledModel);
        }

        compiled_model.run(sig_index, &inputs, &outputs).expect("Failed to run compiled model");

        let mut output_data = [0.0f32; 2];
        outputs[0].read(&mut output_data).expect("Failed to read output 0");
        assert!((output_data[0] - 11.0).abs() < 1e-4);
        assert!((output_data[1] - 22.0).abs() < 1e-4);
    }
}
