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

//! Stateful per-signature execution runner (`SignatureRunner`), in-flight async guard
//! (`AsyncExecution`), and tensor selector (`TensorId`).

use crate::bindings::{
    LiteRtParamIndex, LiteRtStatus_kLiteRtStatusErrorIndexOOB,
    LiteRtStatus_kLiteRtStatusErrorNotFound,
};
use crate::compiled_model::CompiledModel;
use crate::error::{Error, ErrorCause};
use crate::model::{InputOutputIndex, SignatureIndex};
use crate::model_executor::ExecutorInner;
use crate::tensor_buffer::TensorBuffer;
use crate::tensor_buffer_lock::{TensorBufferReadLock, TensorBufferWriteLock, TensorElement};
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;

/// Identifies an input or output tensor within a signature by 0-based index or tensor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorId<'a> {
    /// Selects an input or output tensor by its 0-based [`InputOutputIndex`].
    Index(InputOutputIndex),
    /// Selects an input or output tensor by its tensor name string.
    Name(&'a str),
}

impl From<InputOutputIndex> for TensorId<'_> {
    /// Converts an [`InputOutputIndex`] into [`TensorId::Index`].
    fn from(index: InputOutputIndex) -> Self {
        Self::Index(index)
    }
}

impl From<LiteRtParamIndex> for TensorId<'_> {
    /// Converts a 0-based `LiteRtParamIndex` (`usize`) into [`TensorId::Index`].
    fn from(index: LiteRtParamIndex) -> Self {
        Self::Index(InputOutputIndex::from(index))
    }
}

impl<'a> From<&'a str> for TensorId<'a> {
    /// Converts a tensor name string slice into [`TensorId::Name`].
    fn from(name: &'a str) -> Self {
        Self::Name(name)
    }
}

impl<'a> From<&'a String> for TensorId<'a> {
    /// Converts a borrowed `String` tensor name into [`TensorId::Name`].
    fn from(name: &'a String) -> Self {
        Self::Name(name.as_str())
    }
}

/// Represents a single input or output buffer slot in a [`SignatureRunner`].
pub(crate) enum BufferSlot<'model, 'ext> {
    /// Pre-allocated managed `TensorBuffer` owned by the `SignatureRunner`.
    Managed(TensorBuffer<'model>),
    /// External/hardware `TensorBuffer` borrowed from the caller, along with the original
    /// managed buffer so the slot can be restored via `unbind_input_buffer` or
    /// `unbind_output_buffer`.
    External {
        /// External or hardware-backed tensor buffer borrowed from the caller.
        borrowed: &'ext TensorBuffer<'ext>,
        /// Pre-allocated managed buffer preserved so the slot can be restored via `unbind`.
        managed_backup: TensorBuffer<'model>,
    },
    /// Transient placeholder used only during in-place slot transitions.
    Transitioning,
}

impl<'model, 'ext> BufferSlot<'model, 'ext> {
    /// Returns a reference to the currently active [`TensorBuffer`] (either managed or external).
    #[must_use]
    pub(crate) fn active_buffer(&self) -> &TensorBuffer<'_> {
        match self {
            Self::Managed(buf) => buf,
            Self::External { borrowed, .. } => borrowed,
            Self::Transitioning => unreachable!("BufferSlot is never left in Transitioning state"),
        }
    }

    /// Binds an external `buffer` to this slot while preserving the original managed buffer as
    /// backup.
    pub(crate) fn bind(&mut self, buffer: &'ext TensorBuffer<'ext>) {
        *self = match std::mem::replace(self, Self::Transitioning) {
            Self::Managed(managed) => Self::External { borrowed: buffer, managed_backup: managed },
            Self::External { managed_backup, .. } => {
                Self::External { borrowed: buffer, managed_backup }
            }
            Self::Transitioning => unreachable!("BufferSlot is never left in Transitioning state"),
        };
    }

    /// Unbinds any external buffer and restores the slot's pre-allocated managed [`TensorBuffer`].
    pub(crate) fn unbind(&mut self) {
        *self = match std::mem::replace(self, Self::Transitioning) {
            Self::Managed(managed) => Self::Managed(managed),
            Self::External { managed_backup, .. } => Self::Managed(managed_backup),
            Self::Transitioning => unreachable!("BufferSlot is never left in Transitioning state"),
        };
    }

    /// Replaces the slot with `new_managed`, discarding the previous buffer and unbinding any
    /// external buffer.
    pub(crate) fn replace_managed(&mut self, new_managed: TensorBuffer<'model>) {
        *self = Self::Managed(new_managed);
    }
}

/// Pairs an input or output tensor's signature name with its [`BufferSlot`].
pub(crate) struct NamedBufferSlot<'model, 'ext> {
    /// Tensor name from the model's [`crate::model::Signature`].
    pub(crate) name: String,
    /// Buffer storage (managed or externally bound) for this signature tensor.
    pub(crate) slot: BufferSlot<'model, 'ext>,
}

/// Stateful, per-signature execution runner that owns pre-allocated input/output [`TensorBuffer`]s.
///
/// Not `Sync`: concurrent threads should each create their own `SignatureRunner` from a shared
/// [`crate::model_executor::LiteRTModelExecutor`].
///
/// Note on field declaration order: `inputs` and `outputs` are declared before `executor` so that
/// all managed [`TensorBuffer`]s are destroyed before the backing `CompiledModel`, `Model`, and
/// `Environment` inside `executor`.
pub struct SignatureRunner<'model, 'ext> {
    /// Zero-based index of the signature in the [`crate::model::Model`].
    pub(crate) signature_index: SignatureIndex,
    /// Pre-allocated input buffer slots paired with their signature input names.
    pub(crate) inputs: Vec<NamedBufferSlot<'model, 'ext>>,
    /// Pre-allocated output buffer slots paired with their signature output names.
    pub(crate) outputs: Vec<NamedBufferSlot<'model, 'ext>>,
    /// Shared owner of the underlying `CompiledModel`, `Model`, and `Environment`.
    pub(crate) executor: Arc<ExecutorInner<'model>>,
    /// Marker making `SignatureRunner` `Send + !Sync` in safe Rust (`Cell<()>` is `Send + !Sync`).
    pub(crate) _not_sync: PhantomData<Cell<()>>,
}

impl<'model, 'ext> SignatureRunner<'model, 'ext> {
    /// Resolves an input tensor name to its zero-based [`InputOutputIndex`] via a fast contiguous
    /// slice scan over `self.inputs`.
    ///
    /// # Arguments
    ///
    /// * `name` - The input tensor name in the signature.
    ///
    /// # Returns
    ///
    /// The 0-based [`InputOutputIndex`], or [`ErrorCause::SignatureInputNotFound`] if `name` does
    /// not match any input tensor in this signature.
    pub fn input_index(&self, name: &str) -> Result<InputOutputIndex, Error> {
        self.inputs
            .iter()
            .position(|entry| entry.name == name)
            .map(InputOutputIndex::new)
            .ok_or_else(|| {
                Error::new(
                    ErrorCause::SignatureInputNotFound,
                    LiteRtStatus_kLiteRtStatusErrorNotFound,
                )
            })
    }

    /// Resolves an output tensor name to its zero-based [`InputOutputIndex`] via a fast contiguous
    /// slice scan over `self.outputs`.
    ///
    /// # Arguments
    ///
    /// * `name` - The output tensor name in the signature.
    ///
    /// # Returns
    ///
    /// The 0-based [`InputOutputIndex`], or [`ErrorCause::SignatureOutputNotFound`] if `name` does
    /// not match any output tensor in this signature.
    pub fn output_index(&self, name: &str) -> Result<InputOutputIndex, Error> {
        self.outputs
            .iter()
            .position(|entry| entry.name == name)
            .map(InputOutputIndex::new)
            .ok_or_else(|| {
                Error::new(
                    ErrorCause::SignatureOutputNotFound,
                    LiteRtStatus_kLiteRtStatusErrorNotFound,
                )
            })
    }

    /// Resolves a [`TensorId`] (index or name) to a validated 0-based index into `self.inputs`.
    pub(crate) fn resolve_input_index(&self, id: TensorId<'_>) -> Result<usize, Error> {
        match id {
            TensorId::Index(idx) => {
                let i = idx.value();
                if i < self.inputs.len() {
                    Ok(i)
                } else {
                    Err(Error::new(
                        ErrorCause::SignatureInputNotFound,
                        LiteRtStatus_kLiteRtStatusErrorIndexOOB,
                    ))
                }
            }
            TensorId::Name(name) => Ok(self.input_index(name)?.value()),
        }
    }

    /// Resolves a [`TensorId`] (index or name) to a validated 0-based index into `self.outputs`.
    pub(crate) fn resolve_output_index(&self, id: TensorId<'_>) -> Result<usize, Error> {
        match id {
            TensorId::Index(idx) => {
                let i = idx.value();
                if i < self.outputs.len() {
                    Ok(i)
                } else {
                    Err(Error::new(
                        ErrorCause::SignatureOutputNotFound,
                        LiteRtStatus_kLiteRtStatusErrorIndexOOB,
                    ))
                }
            }
            TensorId::Name(name) => Ok(self.output_index(name)?.value()),
        }
    }

    /// Returns a reference to the active input [`TensorBuffer`] for `id` (name or index).
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// A shared reference to the active [`TensorBuffer`] for `id`, or an [`Error`] if `id` is not
    /// found.
    pub fn input_buffer<'s>(
        &self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<&TensorBuffer<'_>, Error> {
        let idx = self.resolve_input_index(id.into())?;
        Ok(self.inputs[idx].slot.active_buffer())
    }

    /// Returns a reference to the active output [`TensorBuffer`] for `id` (name or index).
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// A shared reference to the active [`TensorBuffer`] for `id`, or an [`Error`] if `id` is not
    /// found.
    pub fn output_buffer<'s>(
        &self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<&TensorBuffer<'_>, Error> {
        let idx = self.resolve_output_index(id.into())?;
        Ok(self.outputs[idx].slot.active_buffer())
    }

    /// Copies `data` into the input tensor buffer identified by `id` (name or index).
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `data` - Slice of elements to copy into the input buffer.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found, `T` is incompatible
    /// with the tensor's element type, or `data` exceeds the buffer capacity.
    pub fn set_input<'s, T: TensorElement>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        data: &[T],
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_input_index(id.into())?;
        self.inputs[idx].slot.active_buffer().write(data)?;
        Ok(self)
    }

    /// Locks the input tensor buffer identified by `id` in write mode and invokes `f` with a
    /// mutable typed slice `&mut [T]` for zero-copy in-place initialization.
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `f` - Closure receiving the locked `&mut [T]` slice.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found, `T` is incompatible,
    /// or locking fails.
    pub fn set_input_with<'s, T: TensorElement>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        f: impl FnOnce(&mut [T]),
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_input_index(id.into())?;
        let mut lock = self.inputs[idx].slot.active_buffer().lock_write::<T>()?;
        f(&mut lock);
        drop(lock);
        Ok(self)
    }

    /// Locks the input tensor buffer identified by `id` for writing and returns an RAII write
    /// guard.
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// A [`TensorBufferWriteLock`] dereferencing to `&mut [T]`, or an [`Error`] if `id` is not
    /// found, `T` is incompatible, or the buffer is already locked.
    pub fn lock_input<'s, T: TensorElement>(
        &mut self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<TensorBufferWriteLock<'_, T>, Error> {
        let idx = self.resolve_input_index(id.into())?;
        self.inputs[idx].slot.active_buffer().lock_write::<T>()
    }

    /// Binds an external or hardware-backed [`TensorBuffer`] to the input slot identified by `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `buffer` - External [`TensorBuffer`] to bind to this input slot.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found.
    pub fn bind_input_buffer<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        buffer: &'ext TensorBuffer<'ext>,
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_input_index(id.into())?;
        self.inputs[idx].slot.bind(buffer);
        Ok(self)
    }

    /// Restores the input slot identified by `id` back to its pre-allocated managed
    /// [`TensorBuffer`].
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found.
    pub fn unbind_input_buffer<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_input_index(id.into())?;
        self.inputs[idx].slot.unbind();
        Ok(self)
    }

    /// Binds an external or hardware-backed [`TensorBuffer`] to the output slot identified by `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `buffer` - External [`TensorBuffer`] to bind to this output slot.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found.
    pub fn bind_output_buffer<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        buffer: &'ext TensorBuffer<'ext>,
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_output_index(id.into())?;
        self.outputs[idx].slot.bind(buffer);
        Ok(self)
    }

    /// Restores the output slot identified by `id` back to its pre-allocated managed
    /// [`TensorBuffer`].
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found.
    pub fn unbind_output_buffer<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_output_index(id.into())?;
        self.outputs[idx].slot.unbind();
        Ok(self)
    }

    /// Resizes a dynamic input tensor and re-allocates its managed [`TensorBuffer`] (and any
    /// affected output [`TensorBuffer`]s) atomically.
    ///
    /// Note: if an external buffer was previously bound to the resized input or any output slot,
    /// `resize_input` unbinds it and switches the slot to the newly allocated managed buffer
    /// matching the resized layout.
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `dimensions` - New concrete dimensions for the input tensor.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found, resizing fails, or
    /// buffer re-allocation fails.
    pub fn resize_input<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        dimensions: &[i32],
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_input_index(id.into())?;
        self.executor.compiled_model.resize_input_tensor_buffer(
            self.signature_index,
            InputOutputIndex::new(idx),
            dimensions,
        )?;

        let signature = self.executor.model.signature(self.signature_index)?;
        let input_req =
            self.executor.compiled_model.input_buffer_requirements(self.signature_index, idx)?;
        let tensor = signature.input_tensor_by_index(idx)?;
        let mut tensor_type = tensor.ranked_tensor_type()?;
        tensor_type.layout =
            self.executor.compiled_model.input_tensor_layout(self.signature_index, idx)?;
        let new_rank = usize::try_from(tensor_type.layout.rank())
            .unwrap_or(0)
            .min(tensor_type.layout.dimensions.len());
        for dim in &mut tensor_type.layout.dimensions[..new_rank] {
            if *dim < 0 {
                *dim = 1;
            }
        }
        let new_input_buf = CompiledModel::create_buffer_with_type(
            &self.executor.environment,
            &input_req,
            &tensor_type,
            tensor.element_type()?,
        )?;

        let num_outputs = self.outputs.len();
        let mut new_output_bufs = Vec::with_capacity(num_outputs);
        if num_outputs > 0 {
            let output_layouts = self
                .executor
                .compiled_model
                .output_tensor_layouts(self.signature_index, num_outputs)?;
            for (out_idx, out_layout) in output_layouts.into_iter().enumerate() {
                let out_req = self
                    .executor
                    .compiled_model
                    .output_buffer_requirements(self.signature_index, out_idx)?;
                let out_tensor = signature.output_tensor_by_index(out_idx)?;
                let mut out_tensor_type = out_tensor.ranked_tensor_type()?;
                out_tensor_type.layout = out_layout;
                let out_rank = usize::try_from(out_tensor_type.layout.rank())
                    .unwrap_or(0)
                    .min(out_tensor_type.layout.dimensions.len());
                for dim in &mut out_tensor_type.layout.dimensions[..out_rank] {
                    if *dim < 0 {
                        *dim = 1;
                    }
                }
                let new_out_buf = CompiledModel::create_buffer_with_type(
                    &self.executor.environment,
                    &out_req,
                    &out_tensor_type,
                    out_tensor.element_type()?,
                )?;
                new_output_bufs.push(new_out_buf);
            }
        }

        // Commit all new buffers atomically after all fallible allocations succeed.
        self.inputs[idx].slot.replace_managed(new_input_buf);
        for (out_idx, new_out_buf) in new_output_bufs.into_iter().enumerate() {
            self.outputs[out_idx].slot.replace_managed(new_out_buf);
        }

        Ok(self)
    }

    /// Executes inference synchronously for this signature using the current input/output buffers.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if inference execution fails.
    pub fn run(&mut self) -> Result<&mut Self, Error> {
        self.executor.compiled_model.run_with_iter(
            self.signature_index,
            self.inputs.iter().map(|entry| entry.slot.active_buffer()),
            self.outputs.iter().map(|entry| entry.slot.active_buffer()),
        )?;
        Ok(self)
    }

    /// Executes inference asynchronously (if supported by the accelerator) and returns an
    /// [`AsyncExecution`] guard that borrows `self` mutably until completion.
    ///
    /// # Returns
    ///
    /// An [`AsyncExecution`] guard holding `&mut self` while execution is in flight, or an
    /// [`Error`] if launching asynchronous execution fails.
    #[cfg(async_support)]
    pub fn run_async(&mut self) -> Result<AsyncExecution<'_, 'model, 'ext>, Error> {
        let ran_in_parallel = self.executor.compiled_model.run_async_with_iter(
            self.signature_index,
            self.inputs.iter().map(|entry| entry.slot.active_buffer()),
            self.outputs.iter().map(|entry| entry.slot.active_buffer()),
        )?;
        Ok(AsyncExecution { runner: self, ran_in_parallel })
    }

    /// Copies data from the output tensor buffer identified by `id` into `dst`.
    /// Returns `&mut Self` so multiple `.get_output(...)` calls can be chained.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `dst` - Destination slice to receive the output elements (must have `dst.len() >=` tensor
    ///   element count).
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found, `T` is incompatible,
    /// or `dst` is smaller than the output tensor (`ErrorCause::ReadBufferTooSmall`).
    pub fn get_output<'s, T: TensorElement>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        dst: &mut [T],
    ) -> Result<&mut Self, Error> {
        let idx = self.resolve_output_index(id.into())?;
        self.outputs[idx].slot.active_buffer().read(dst)?;
        Ok(self)
    }

    /// Allocates and returns a `Vec<T>` containing the output tensor data for `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// A newly allocated `Vec<T>` containing the output elements, or an [`Error`] if `id` is not
    /// found, `T` is incompatible, or locking fails.
    pub fn output_vec<'s, T: TensorElement>(
        &self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<Vec<T>, Error> {
        let lock = self.lock_output::<T>(id)?;
        Ok(lock.to_vec())
    }

    /// Locks the output tensor buffer identified by `id` in read mode and returns a zero-copy
    /// RAII guard implementing `Deref<Target = [T]>`.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// A [`TensorBufferReadLock`] dereferencing to `&[T]`, or an [`Error`] if `id` is not found,
    /// `T` is incompatible, or the buffer is already locked.
    pub fn lock_output<'s, T: TensorElement>(
        &self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<TensorBufferReadLock<'_, T>, Error> {
        let idx = self.resolve_output_index(id.into())?;
        self.outputs[idx].slot.active_buffer().lock_read::<T>()
    }
}

/// Guard representing an in-flight asynchronous execution on a [`SignatureRunner`].
///
/// Holds `&'runner mut SignatureRunner<'model, 'ext>`, statically preventing input mutation,
/// buffer unbinding, or concurrent runs until awaited or dropped.
#[cfg(async_support)]
#[must_use = "dropping AsyncExecution immediately blocks until execution completes; await outputs"]
pub struct AsyncExecution<'runner, 'model, 'ext> {
    /// Mutable borrow of the [`SignatureRunner`] whose asynchronous execution is in flight.
    pub(crate) runner: &'runner mut SignatureRunner<'model, 'ext>,
    /// Whether the `LiteRT` runtime scheduled execution on a parallel thread or hardware queue.
    pub(crate) ran_in_parallel: bool,
}

#[cfg(async_support)]
impl<'runner, 'model, 'ext> AsyncExecution<'runner, 'model, 'ext> {
    /// Returns `true` if the accelerator scheduled execution asynchronously on a separate thread
    /// or hardware queue; `false` if execution completed synchronously during `run_async()`.
    ///
    /// # Returns
    ///
    /// `true` if asynchronous execution is running in parallel; `false` otherwise.
    #[must_use]
    pub const fn is_parallel(&self) -> bool {
        self.ran_in_parallel
    }

    /// Waits for the synchronization event (if any) attached to the output tensor `id` and
    /// copies the output data into `dst`, returning the number of elements copied.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `dst` - Destination slice to receive the output elements.
    ///
    /// # Returns
    ///
    /// The number of elements copied into `dst`, or an [`Error`] if `id` is not found, waiting on
    /// the event fails, `T` is incompatible, or `dst` is too small.
    pub async fn get_output<'s, T: TensorElement>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        dst: &mut [T],
    ) -> Result<usize, Error> {
        let idx = self.runner.resolve_output_index(id.into())?;
        let buf = self.runner.outputs[idx].slot.active_buffer();
        buf.read_async(dst).await
    }

    /// Waits for the synchronization event (if any) attached to the output tensor `id` and
    /// returns the output data as a newly allocated `Vec<T>`.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    ///
    /// # Returns
    ///
    /// A `Vec<T>` containing the output elements, or an [`Error`] if `id` is not found, waiting on
    /// the event fails, or `T` is incompatible.
    pub async fn output_vec<'s, T: TensorElement>(
        &mut self,
        id: impl Into<TensorId<'s>>,
    ) -> Result<Vec<T>, Error> {
        let idx = self.runner.resolve_output_index(id.into())?;
        let buf = self.runner.outputs[idx].slot.active_buffer();
        buf.wait_event().await?;
        let lock = buf.lock_read::<T>()?;
        Ok(lock.to_vec())
    }
}

#[cfg(async_support)]
impl Drop for AsyncExecution<'_, '_, '_> {
    /// Blocks until any pending synchronization events on the signature's output buffers signal
    /// and clears them so the [`SignatureRunner`] is safe to reuse immediately after drop.
    fn drop(&mut self) {
        for output in &self.runner.outputs {
            let _ = output.slot.active_buffer().wait_event_blocking();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled_model::{LiteRtHwAccelerator, Options};
    use crate::environment::EnvironmentBuilder;
    use crate::model::Model;
    use crate::model_executor::{LiteRTModelExecutor, SignatureId};
    use std::path::Path;

    /// Verifies `Default` and `From` conversions for [`SignatureId`] and [`TensorId`].
    #[test]
    fn test_selectors() {
        let sig_id_default = SignatureId::default();
        assert_eq!(sig_id_default, SignatureId::Index(SignatureIndex::DEFAULT));
        assert_eq!(
            SignatureId::from(SignatureIndex::new(2)),
            SignatureId::Index(SignatureIndex::new(2))
        );
        assert_eq!(SignatureId::from(2usize), SignatureId::Index(SignatureIndex::new(2)));
        assert_eq!(SignatureId::from("serving_default"), SignatureId::Key("serving_default"));
        let sig_string = String::from("serving_default");
        assert_eq!(SignatureId::from(&sig_string), SignatureId::Key("serving_default"));

        assert_eq!(
            TensorId::from(InputOutputIndex::new(1)),
            TensorId::Index(InputOutputIndex::new(1))
        );
        assert_eq!(TensorId::from(1usize), TensorId::Index(InputOutputIndex::new(1)));
        assert_eq!(TensorId::from("arg0"), TensorId::Name("arg0"));
        let tensor_string = String::from("arg0");
        assert_eq!(TensorId::from(&tensor_string), TensorId::Name("arg0"));
    }

    /// Exercises end-to-end synchronous, loop, zero-copy lock, external buffer bind/unbind,
    /// multithreaded, `from_buffer`, `run_async`, and error-path execution on `SignatureRunner`.
    #[test]
    fn test_model_executor_and_signature_runner_end_to_end() {
        /// Compile-time assertion that `T` implements `Send + Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        /// Compile-time assertion that `T` implements `Send`.
        fn assert_send<T: Send>() {}
        assert_send_sync::<LiteRTModelExecutor<'static>>();
        assert_send::<SignatureRunner<'static, 'static>>();
        #[cfg(async_support)]
        assert_send::<AsyncExecution<'static, 'static, 'static>>();

        let test_paths = [
            "third_party/odml/litert/litert/test/testdata/simple_model.tflite",
            "../test/testdata/simple_model.tflite",
        ];
        let Some(&model_path) = test_paths.iter().find(|p| Path::new(p).exists()) else {
            return;
        };

        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
            .expect("Failed to create options");
        let executor = LiteRTModelExecutor::from_file(model_path, options)
            .expect("Failed to create LiteRTModelExecutor");

        // Discover signature key and input/output names from the model.
        let sig =
            executor.model().signature(SignatureIndex::DEFAULT).expect("Failed to get signature");
        let sig_key = sig.key().expect("Failed to get signature key").to_string();
        let input_names: Vec<String> = sig
            .input_names()
            .expect("Failed to get input names")
            .map(|r| r.expect("Failed to read input name").to_string())
            .collect();
        let output_names: Vec<String> = sig
            .output_names()
            .expect("Failed to get output names")
            .map(|r| r.expect("Failed to read output name").to_string())
            .collect();
        assert_eq!(input_names.len(), 2);
        assert_eq!(output_names.len(), 1);

        // 1. One-shot execution with method chaining & temporary drop-order safety.
        let mut one_shot_out = [0.0f32; 2];
        LiteRTModelExecutor::from_file(
            model_path,
            Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
                .expect("Failed to create options"),
        )
        .expect("Failed to create executor")
        .signature_runner(SignatureId::default())
        .expect("Failed to create signature runner")
        .set_input(0, &[3.0f32, 4.0f32])
        .expect("Failed to set input 0")
        .set_input(1, &[30.0f32, 40.0f32])
        .expect("Failed to set input 1")
        .run()
        .expect("Failed to run model")
        .get_output(0, &mut one_shot_out)
        .expect("Failed to get output 0");
        assert!((one_shot_out[0] - 33.0).abs() < 1e-4);
        assert!((one_shot_out[1] - 44.0).abs() < 1e-4);

        // 2. Stateful runner lookup by signature key & tensor names, plus loop reuse.
        let mut runner = executor
            .signature_runner(sig_key.as_str())
            .expect("Failed to create runner by signature key");
        assert_eq!(
            runner.input_index(&input_names[0]).expect("Failed to resolve input 0").value(),
            0
        );
        assert_eq!(
            runner.input_index(&input_names[1]).expect("Failed to resolve input 1").value(),
            1
        );
        assert_eq!(
            runner.output_index(&output_names[0]).expect("Failed to resolve output 0").value(),
            0
        );
        assert!(runner.input_index("nonexistent").is_err());
        assert!(runner.output_index("nonexistent").is_err());
        assert!(runner.input_buffer(99).is_err());
        assert!(runner.output_buffer(99).is_err());

        for step in 1..=3 {
            let a = step as f32;
            runner
                .set_input(&input_names[0], &[a, a * 2.0])
                .expect("Failed to set input 0")
                .set_input_with(&input_names[1], |slice: &mut [f32]| {
                    slice[0] = 100.0;
                    slice[1] = 200.0;
                })
                .expect("Failed to set input 1 with closure")
                .run()
                .expect("Failed to run model");

            let lock = runner
                .lock_output::<f32>(&output_names[0])
                .expect("Failed to lock output buffer for reading");
            assert!((lock[0] - (100.0 + a)).abs() < 1e-4);
            assert!((lock[1] - (200.0 + a * 2.0)).abs() < 1e-4);
        }

        let vec_out = runner.output_vec::<f32>(0).expect("Failed to read output_vec");
        assert_eq!(vec_out.len(), 2);
        assert!((vec_out[0] - 103.0).abs() < 1e-4);
        assert!((vec_out[1] - 206.0).abs() < 1e-4);

        // 3. Test lock_input directly.
        {
            let mut in_lock =
                runner.lock_input::<f32>(0).expect("Failed to lock input buffer for writing");
            in_lock[0] = 7.0;
            in_lock[1] = 8.0;
        }
        runner
            .set_input(1, &[1.0f32, 2.0f32])
            .expect("Failed to set input 1")
            .run()
            .expect("Failed to run model");
        let vec_out2 = runner.output_vec::<f32>(0).expect("Failed to read output_vec");
        assert!((vec_out2[0] - 8.0).abs() < 1e-4);
        assert!((vec_out2[1] - 10.0).abs() < 1e-4);

        // 4. Test external buffer bind and unbind.
        let ext_inputs = executor
            .compiled_model()
            .create_input_tensor_buffers(
                executor.environment(),
                executor.model(),
                SignatureIndex::DEFAULT,
            )
            .expect("Failed to create external input buffers");
        let ext_outputs = executor
            .compiled_model()
            .create_output_tensor_buffers(
                executor.environment(),
                executor.model(),
                SignatureIndex::DEFAULT,
            )
            .expect("Failed to create external output buffers");
        ext_inputs[0].write(&[50.0f32, 60.0f32]).expect("Failed to write external input 0");
        {
            let mut bound_runner =
                executor.signature_runner(0).expect("Failed to create bound runner");
            bound_runner
                .bind_input_buffer(0, &ext_inputs[0])
                .expect("Failed to bind external input buffer")
                .set_input(1, &[5.0f32, 6.0f32])
                .expect("Failed to set input 1")
                .bind_output_buffer(0, &ext_outputs[0])
                .expect("Failed to bind external output buffer")
                .run()
                .expect("Failed to run with bound buffers");

            let mut ext_out_data = [0.0f32; 2];
            ext_outputs[0].read(&mut ext_out_data).expect("Failed to read external output 0");
            assert!((ext_out_data[0] - 55.0).abs() < 1e-4);
            assert!((ext_out_data[1] - 66.0).abs() < 1e-4);

            // Unbind and verify managed buffers work again.
            bound_runner
                .unbind_input_buffer(0)
                .expect("Failed to unbind input buffer")
                .unbind_output_buffer(0)
                .expect("Failed to unbind output buffer")
                .set_input(0, &[9.0f32, 10.0f32])
                .expect("Failed to set restored input 0")
                .run()
                .expect("Failed to run with restored buffers");
            let restored_out =
                bound_runner.output_vec::<f32>(0).expect("Failed to read restored output");
            assert!((restored_out[0] - 14.0).abs() < 1e-4);
            assert!((restored_out[1] - 16.0).abs() < 1e-4);
        }

        // 5. Test multithreaded execution using cloned `LiteRTModelExecutor`.
        let mut handles = Vec::new();
        for tid in 0..4 {
            let exec_clone = executor.clone();
            handles.push(std::thread::spawn(move || {
                let mut thread_runner =
                    exec_clone.signature_runner(0).expect("Failed to create thread runner");
                let x = (tid + 1) as f32;
                thread_runner
                    .set_input(0, &[x, x * 10.0])
                    .expect("Failed to set thread input 0")
                    .set_input(1, &[100.0f32, 200.0f32])
                    .expect("Failed to set thread input 1")
                    .run()
                    .expect("Failed to run thread model");
                let out = thread_runner.output_vec::<f32>(0).expect("Failed to read thread output");
                assert!((out[0] - (100.0 + x)).abs() < 1e-4);
                assert!((out[1] - (200.0 + x * 10.0)).abs() < 1e-4);
            }));
        }
        for h in handles {
            h.join().expect("Worker thread panicked");
        }

        // 6. Test `from_buffer` with borrowed slice (`&[u8]`) and owned `Vec<u8>`.
        let model_bytes = std::fs::read(model_path).expect("Failed to read model file");
        {
            let slice_exec = LiteRTModelExecutor::from_buffer(
                model_bytes.as_slice(),
                Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
                    .expect("Failed to create options"),
            )
            .expect("Failed to create executor from slice");
            let mut slice_runner =
                slice_exec.signature_runner(0).expect("Failed to create slice runner");
            slice_runner
                .set_input(0, &[2.0f32, 3.0f32])
                .expect("Failed to set slice input 0")
                .set_input(1, &[4.0f32, 5.0f32])
                .expect("Failed to set slice input 1")
                .run()
                .expect("Failed to run slice runner");
            assert_eq!(
                slice_runner.output_vec::<f32>(0).expect("Failed to read slice output"),
                vec![6.0f32, 8.0f32]
            );
        }
        let owned_exec: LiteRTModelExecutor<'static> = LiteRTModelExecutor::from_buffer(
            model_bytes,
            Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
                .expect("Failed to create options"),
        )
        .expect("Failed to create executor from owned Vec");
        let mut owned_runner =
            owned_exec.signature_runner(0).expect("Failed to create owned runner");
        owned_runner
            .set_input(0, &[12.0f32, 13.0f32])
            .expect("Failed to set owned input 0")
            .set_input(1, &[14.0f32, 15.0f32])
            .expect("Failed to set owned input 1")
            .run()
            .expect("Failed to run owned runner");
        assert_eq!(
            owned_runner.output_vec::<f32>(0).expect("Failed to read owned output"),
            vec![26.0f32, 28.0f32]
        );

        // 7. Test `run_async` and `AsyncExecution` when `async_support` is enabled.
        #[cfg(async_support)]
        {
            use std::future::Future;
            use std::pin::Pin;
            use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

            /// Minimal single-threaded spin-executor for polling futures in unit tests.
            fn block_on<F: Future>(mut f: F) -> F::Output {
                const VTABLE: RawWakerVTable = RawWakerVTable::new(
                    |_| RawWaker::new(std::ptr::null(), &VTABLE),
                    |_| {},
                    |_| {},
                    |_| {},
                );
                let raw_waker = RawWaker::new(std::ptr::null(), &VTABLE);
                // SAFETY: `raw_waker` is a valid no-op RawWaker.
                let waker = unsafe { Waker::from_raw(raw_waker) };
                let mut cx = Context::from_waker(&waker);
                // SAFETY: `f` is pinned on the stack for the duration of `block_on`.
                let mut f = unsafe { Pin::new_unchecked(&mut f) };
                loop {
                    match f.as_mut().poll(&mut cx) {
                        Poll::Ready(res) => return res,
                        Poll::Pending => std::thread::yield_now(),
                    }
                }
            }

            owned_runner
                .set_input(0, &[20.0f32, 30.0f32])
                .expect("Failed to set async input 0")
                .set_input(1, &[1.5f32, 2.5f32])
                .expect("Failed to set async input 1");
            let mut async_exec = owned_runner.run_async().expect("Failed to start async execution");
            let _ = async_exec.is_parallel();
            let mut async_out = [0.0f32; 2];
            let copied = block_on(async_exec.get_output(0, &mut async_out))
                .expect("Failed to await async output");
            assert_eq!(copied, 2);
            assert!((async_out[0] - 21.5).abs() < 1e-4);
            assert!((async_out[1] - 32.5).abs() < 1e-4);

            let async_vec = block_on(async_exec.output_vec::<f32>(0))
                .expect("Failed to await async output_vec");
            assert_eq!(async_vec, vec![21.5f32, 32.5f32]);
            drop(async_exec);

            // Verify that dropping `AsyncExecution` without polling its outputs still leaves the
            // runner's output buffers clean for subsequent `run()` calls.
            let unpolled_exec =
                owned_runner.run_async().expect("Failed to start unpolled async execution");
            drop(unpolled_exec);
            owned_runner
                .set_input(0, &[1.0f32, 2.0f32])
                .expect("Failed to set input 0 after unpolled drop")
                .set_input(1, &[3.0f32, 4.0f32])
                .expect("Failed to set input 1 after unpolled drop")
                .run()
                .expect("Failed to run after unpolled drop");
            assert_eq!(
                owned_runner.output_vec::<f32>(0).expect("Failed to read output after drop"),
                vec![4.0f32, 6.0f32]
            );
        }

        // 8. Test error paths: invalid signature key/index, mismatched type, oversized input,
        // undersized read buffer, and double-lock rejection.
        assert!(executor.signature_runner("nonexistent_signature").is_err());
        assert!(executor.signature_runner(99usize).is_err());

        let err_type = owned_runner.set_input(0, &[1i32, 2i32]).err().expect("Expected type err");
        assert_eq!(err_type.cause(), ErrorCause::IncompatibleWriteType);

        let err_read_type = owned_runner.output_vec::<i32>(0).expect_err("Expected read type err");
        assert_eq!(err_read_type.cause(), ErrorCause::IncompatibleReadType);

        let err_oversized =
            owned_runner.set_input(0, &[1.0f32, 2.0, 3.0]).err().expect("Expected size err");
        assert_eq!(err_oversized.cause(), ErrorCause::TensorBufferTooSmall);

        let mut tiny_dst = [0.0f32; 1];
        let err_undersized =
            owned_runner.get_output(0, &mut tiny_dst).err().expect("Expected read size err");
        assert_eq!(err_undersized.cause(), ErrorCause::ReadBufferTooSmall);

        {
            let _first_lock =
                owned_runner.lock_output::<f32>(0).expect("Failed to acquire first read lock");
            let second_lock_err =
                owned_runner.lock_output::<f32>(0).err().expect("Expected double-lock err");
            assert_eq!(second_lock_err.cause(), ErrorCause::LockTensorBufferRead);
        }

        // 9. Test `from_parts` error when `compiled_model` comes from a different `Environment`.
        {
            let env1 = EnvironmentBuilder::build_default().expect("Failed to build env1");
            let env2 = EnvironmentBuilder::build_default().expect("Failed to build env2");
            let m1 = Model::create_model_from_file(&env1, model_path).expect("Failed to load m1");
            let cm1 = CompiledModel::create(
                &env1,
                &m1,
                &Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
                    .expect("Failed to create options"),
            )
            .expect("Failed to compile cm1");
            let mismatch_err = LiteRTModelExecutor::from_parts(env2, m1, cm1)
                .err()
                .expect("Expected from_parts mismatch error");
            assert_eq!(mismatch_err.cause(), ErrorCause::GetCompiledModelEnvironment);
        }
    }

    /// Verifies dynamic input tensor resizing and automatic input/output buffer reallocation on
    /// [`SignatureRunner`].
    #[test]
    fn test_dynamic_shape_resize_input() {
        let test_paths = [
            "third_party/odml/litert/litert/test/testdata/dynamic_add_model.tflite",
            "../test/testdata/dynamic_add_model.tflite",
        ];
        let Some(&model_path) = test_paths.iter().find(|p| Path::new(p).exists()) else {
            return;
        };

        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
            .expect("Failed to create options");
        let executor = LiteRTModelExecutor::from_file(model_path, options)
            .expect("Failed to create LiteRTModelExecutor for dynamic model");

        let mut runner = executor
            .signature_runner(SignatureId::default())
            .expect("Failed to create SignatureRunner for dynamic model");

        // Resize both dynamic inputs from `[-1, 2, 3]` to `[2, 2, 3]` (12 elements).
        runner
            .resize_input(0, &[2, 2, 3])
            .expect("Failed to resize input 0")
            .resize_input(1, &[2, 2, 3])
            .expect("Failed to resize input 1");

        assert_eq!(
            runner.input_buffer(0).expect("Failed to get input buffer 0").dimensions(),
            &[2, 2, 3]
        );
        assert_eq!(
            runner.input_buffer(1).expect("Failed to get input buffer 1").dimensions(),
            &[2, 2, 3]
        );
        assert_eq!(
            runner.output_buffer(0).expect("Failed to get output buffer 0").dimensions(),
            &[2, 2, 3]
        );

        let in0: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let in1: Vec<f32> = (0..12).map(|x| (x * 10) as f32).collect();
        runner
            .set_input(0, &in0)
            .expect("Failed to set dynamic input 0")
            .set_input(1, &in1)
            .expect("Failed to set dynamic input 1")
            .run()
            .expect("Failed to run dynamic model");

        let out = runner.output_vec::<f32>(0).expect("Failed to read dynamic output");
        assert_eq!(out.len(), 12);
        for i in 0..12 {
            assert!((out[i] - (in0[i] + in1[i])).abs() < 1e-4);
        }
    }
}
