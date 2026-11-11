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
    LiteRtStatus_kLiteRtStatusErrorInvalidArgument, LiteRtStatus_kLiteRtStatusErrorNotFound,
};
use crate::compiled_model::CompiledModel;
use crate::error::{Error, ErrorCause};
use crate::model::{InputOutputIndex, SignatureIndex};
use crate::model_executor::ExecutorInner;
use crate::tensor_buffer::{ElementType, TensorBuffer};
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

    /// Returns a reference to the slot's managed [`TensorBuffer`] (either currently active or
    /// preserved as backup while an external buffer is bound).
    #[must_use]
    pub(crate) fn managed_buffer(&self) -> &TensorBuffer<'model> {
        match self {
            Self::Managed(buf) => buf,
            Self::External { managed_backup, .. } => managed_backup,
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
/// [`crate::model_executor::LiteRTModelExecutor`]. Note that while each `SignatureRunner` owns
/// independent input and output [`TensorBuffer`]s (allowing parallel input preparation and output
/// reading across threads), the underlying [`CompiledModel`] serializes C API `run()` /
/// `run_async()` invocations with an internal mutex because C++ `LiteRtCompiledModelT` wraps a
/// single `tflite::Interpreter`.
///
/// Note on field declaration order: `outputs` is declared before `inputs`, and both are declared
/// before `executor`, so that on drop any pending output events are awaited and output buffers are
/// destroyed before input buffers, and all managed [`TensorBuffer`]s are destroyed before the
/// backing `CompiledModel`, `Model`, and `Environment` inside `executor`.
pub struct SignatureRunner<'model, 'ext> {
    /// Zero-based index of the signature in the [`crate::model::Model`].
    pub(crate) signature_index: SignatureIndex,
    /// Pre-allocated output buffer slots paired with their signature output names (declared before
    /// `inputs` so output buffers and their events are dropped before input buffers).
    pub(crate) outputs: Vec<NamedBufferSlot<'model, 'ext>>,
    /// Pre-allocated input buffer slots paired with their signature input names.
    pub(crate) inputs: Vec<NamedBufferSlot<'model, 'ext>>,
    /// Shared owner of the underlying `CompiledModel`, `Model`, and `Environment`.
    pub(crate) executor: Arc<ExecutorInner<'model>>,
    /// Whether an earlier `resize_input` call on a multi-input signature deferred output buffer
    /// reallocation because sibling dynamic inputs had not yet been resized to compatible shapes.
    pub(crate) outputs_need_reallocation: bool,
    /// Tracks whether an asynchronous run (`run_async`) attached output buffer events that have
    /// not yet been awaited across all outputs (protecting input and external buffers even if
    /// `AsyncExecution` is leaked via `std::mem::forget`).
    #[cfg(async_support)]
    pub(crate) has_unawaited_output_events: Cell<bool>,
    /// First synchronization error encountered while waiting/clearing output buffer events in
    /// `AsyncExecution::drop`, surfaced on the next runner operation.
    #[cfg(async_support)]
    pub(crate) pending_event_error: Cell<Option<Error>>,
    /// Marker making `SignatureRunner` `Send + !Sync` in safe Rust (`Cell<()>` is `Send + !Sync`).
    pub(crate) _not_sync: PhantomData<Cell<()>>,
}

impl<'model, 'ext> SignatureRunner<'model, 'ext> {
    /// Synchronizes any unawaited asynchronous output events (e.g., if a previous
    /// [`AsyncExecution`] guard was leaked via `std::mem::forget`) and checks any deferred
    /// synchronization error recorded by `AsyncExecution::drop`, keeping the error sticky if
    /// synchronization still fails.
    #[inline]
    pub(crate) fn check_pending_event_error(&self) -> Result<(), Error> {
        #[cfg(async_support)]
        {
            if let Some(prev_err) = self.pending_event_error.take() {
                for output in &self.outputs {
                    if let Err(retry_err) = output.slot.active_buffer().wait_event_blocking() {
                        self.pending_event_error.set(Some(retry_err));
                        return Err(prev_err);
                    }
                }
                self.has_unawaited_output_events.set(false);
                return Err(prev_err);
            }
            if self.has_unawaited_output_events.get() {
                self.wait_all_output_events_blocking()?;
            }
        }
        Ok(())
    }

    /// Blocks until any pending synchronization events on all output buffers have signaled and been
    /// cleared, attempting every output buffer even if an earlier buffer returns an error and
    /// returning the first error encountered (if any).
    #[cfg(async_support)]
    pub(crate) fn wait_all_output_events_blocking(&self) -> Result<(), Error> {
        let mut first_err: Option<Error> = self.pending_event_error.take();
        for output in &self.outputs {
            if let Err(err) = output.slot.active_buffer().wait_event_blocking() {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
        if let Some(err) = first_err {
            self.has_unawaited_output_events.set(true);
            Err(err)
        } else {
            self.has_unawaited_output_events.set(false);
            Ok(())
        }
    }
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
        self.check_pending_event_error()?;
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
        self.check_pending_event_error()?;
        let idx = self.resolve_output_index(id.into())?;
        Ok(self.outputs[idx].slot.active_buffer())
    }

    /// Copies `data` into the input tensor buffer identified by `id` (name or index).
    ///
    /// If `data.len()` is smaller than the buffer's element count, only the first `data.len()`
    /// elements are overwritten and any trailing elements retain their previous contents.
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
        self.check_pending_event_error()?;
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
        self.check_pending_event_error()?;
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
        self.check_pending_event_error()?;
        let idx = self.resolve_input_index(id.into())?;
        self.inputs[idx].slot.active_buffer().lock_write::<T>()
    }

    /// Binds an external or hardware-backed [`TensorBuffer`] to the input slot identified by `id`.
    ///
    /// The external `buffer` must match the input slot's [`ElementType`] and `dimensions()`, and
    /// its `packed_size()` must be at least the slot's managed buffer size.
    ///
    /// # Arguments
    ///
    /// * `id` - Input tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `buffer` - External [`TensorBuffer`] to bind to this input slot.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found, `buffer`'s element
    /// type (`ErrorCause::IncompatibleWriteType`), shape
    /// (`ErrorCause::ResizeCompiledModelInputBuffer`), or byte size
    /// (`ErrorCause::TensorBufferTooSmall`) is incompatible with the slot.
    pub fn bind_input_buffer<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        buffer: &'ext TensorBuffer<'ext>,
    ) -> Result<&mut Self, Error> {
        self.check_pending_event_error()?;
        let idx = self.resolve_input_index(id.into())?;
        let expected = self.inputs[idx].slot.managed_buffer();
        if buffer.element_type() != expected.element_type() {
            return Err(Error::new(
                ErrorCause::IncompatibleWriteType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        if buffer.dimensions() != expected.dimensions() {
            return Err(Error::new(
                ErrorCause::ResizeCompiledModelInputBuffer,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        if buffer.packed_size()? < expected.packed_size()? {
            return Err(Error::new(
                ErrorCause::TensorBufferTooSmall,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
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
        self.check_pending_event_error()?;
        let idx = self.resolve_input_index(id.into())?;
        self.inputs[idx].slot.unbind();
        Ok(self)
    }

    /// Binds an external or hardware-backed [`TensorBuffer`] to the output slot identified by `id`.
    ///
    /// The external `buffer` must match the output slot's [`ElementType`] and `dimensions()`, and
    /// its `packed_size()` must be at least the slot's managed buffer size.
    ///
    /// # Arguments
    ///
    /// * `id` - Output tensor selector (0-based index, [`InputOutputIndex`], or tensor name).
    /// * `buffer` - External [`TensorBuffer`] to bind to this output slot.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] if `id` is not found, `buffer`'s element
    /// type (`ErrorCause::IncompatibleReadType`), shape
    /// (`ErrorCause::ResizeCompiledModelInputBuffer`), or byte size
    /// (`ErrorCause::TensorBufferTooSmall`) is incompatible with the slot.
    pub fn bind_output_buffer<'s>(
        &mut self,
        id: impl Into<TensorId<'s>>,
        buffer: &'ext TensorBuffer<'ext>,
    ) -> Result<&mut Self, Error> {
        self.ensure_outputs_reallocated()?;
        let idx = self.resolve_output_index(id.into())?;
        let expected = self.outputs[idx].slot.managed_buffer();
        if buffer.element_type() != expected.element_type() {
            return Err(Error::new(
                ErrorCause::IncompatibleReadType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        if buffer.dimensions() != expected.dimensions() {
            return Err(Error::new(
                ErrorCause::ResizeCompiledModelInputBuffer,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        if buffer.packed_size()? < expected.packed_size()? {
            return Err(Error::new(
                ErrorCause::TensorBufferTooSmall,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        self.outputs[idx].slot.bind(buffer);
        self.check_pending_event_error()?;
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
        self.check_pending_event_error()?;
        Ok(self)
    }

    /// Resizes a dynamic input tensor and re-allocates its managed [`TensorBuffer`] and the
    /// signature's managed output [`TensorBuffer`]s.
    ///
    /// Note: if an external buffer was previously bound to the resized input or any output slot,
    /// `resize_input` unbinds it and switches the slot to the newly allocated managed buffer
    /// matching the resized layout. When a signature has multiple dynamic inputs that must be
    /// resized together before output shapes can be inferred (e.g. `.resize_input(0,
    /// dims)?.resize_input(1, dims)?`), output buffer reallocation is automatically deferred if an
    /// intermediate `AllocateTensors()` call cannot yet infer output shapes and completed on the
    /// subsequent `resize_input` or before `run` / `run_async`.
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
        self.check_pending_event_error()?;
        let idx = self.resolve_input_index(id.into())?;
        let signature = self.executor.model.signature(self.signature_index)?;
        let tensor = signature.input_tensor_by_index(idx)?;
        let mut tensor_type = tensor.ranked_tensor_type()?;
        let element_type = tensor.element_type()?;
        let num_outputs = self.outputs.len();
        let prev_dims = self.inputs[idx].slot.active_buffer().dimensions().to_vec();

        let (new_input_buf, maybe_new_output_bufs) = {
            let _model_guard = self.executor.compiled_model.lock_model();

            // Synchronize any other input slots' current dimensions into the shared C++
            // interpreter before resizing `idx`, in case another `SignatureRunner` on a different
            // thread resized the shared interpreter to a different shape.
            for (other_idx, entry) in self.inputs.iter().enumerate() {
                if other_idx != idx {
                    let other_dims = entry.slot.active_buffer().dimensions();
                    if let Ok(other_layout) = self
                        .executor
                        .compiled_model
                        .input_tensor_layout_locked(self.signature_index, other_idx)
                    {
                        let other_rank = usize::try_from(other_layout.rank())
                            .unwrap_or(0)
                            .min(other_layout.dimensions.len());
                        if &other_layout.dimensions[..other_rank] != other_dims {
                            let _ = self.executor.compiled_model.resize_input_tensor_buffer_locked(
                                self.signature_index,
                                InputOutputIndex::new(other_idx),
                                other_dims,
                            );
                        }
                    }
                }
            }

            self.executor.compiled_model.resize_input_tensor_buffer_locked(
                self.signature_index,
                InputOutputIndex::new(idx),
                dimensions,
            )?;

            let step_result: Result<
                (TensorBuffer<'model>, Option<Vec<TensorBuffer<'model>>>),
                Error,
            > = (|| {
                tensor_type.layout = self
                    .executor
                    .compiled_model
                    .input_tensor_layout_locked(self.signature_index, idx)?;
                let new_rank = usize::try_from(tensor_type.layout.rank())
                    .unwrap_or(0)
                    .min(tensor_type.layout.dimensions.len());
                for dim in &mut tensor_type.layout.dimensions[..new_rank] {
                    if *dim < 0 {
                        *dim = 1;
                    }
                }

                // Attempt `output_tensor_layouts_locked` (which runs `AllocateTensors()` in
                // C++) before querying buffer requirements so `tensor->bytes` is updated. If
                // the signature has multiple inputs and a sibling dynamic input has not yet
                // been resized to a compatible shape, `AllocateTensors()` may fail on this
                // intermediate step; in that case we still allocate the resized input buffer
                // (sizing CPU host memory from `dimensions` if `tensor->bytes` is stale) and
                // defer output buffer reallocation until the remaining input(s) are resized.
                let output_layouts_res = self
                    .executor
                    .compiled_model
                    .output_tensor_layouts_locked(self.signature_index, num_outputs);

                let mut input_req = self
                    .executor
                    .compiled_model
                    .input_buffer_requirements_locked(self.signature_index, idx)?;
                if let Ok(packed_bytes) = Self::packed_byte_size_for_type(&tensor_type) {
                    if input_req.buffer_size()? < packed_bytes {
                        input_req.set_min_buffer_size(packed_bytes);
                    }
                }
                let new_input_buf = CompiledModel::create_buffer_with_type(
                    &self.executor.environment,
                    &input_req,
                    &tensor_type,
                    element_type,
                )?;

                match output_layouts_res {
                    Ok(output_layouts) => {
                        let mut new_output_bufs = Vec::with_capacity(num_outputs);
                        for (out_idx, out_layout) in output_layouts.into_iter().enumerate() {
                            let out_req = self
                                .executor
                                .compiled_model
                                .output_buffer_requirements_locked(self.signature_index, out_idx)?;
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
                        Ok((new_input_buf, Some(new_output_bufs)))
                    }
                    Err(err) if self.inputs.len() > 1 => Ok((new_input_buf, None)),
                    Err(err) => Err(err),
                }
            })();

            if step_result.is_err() {
                let _ = self.executor.compiled_model.resize_input_tensor_buffer_locked(
                    self.signature_index,
                    InputOutputIndex::new(idx),
                    &prev_dims,
                );
            }
            step_result?
        };

        // Commit all new buffers atomically after all fallible allocations succeed.
        self.inputs[idx].slot.replace_managed(new_input_buf);
        if let Some(new_output_bufs) = maybe_new_output_bufs {
            for (out_idx, new_out_buf) in new_output_bufs.into_iter().enumerate() {
                self.outputs[out_idx].slot.replace_managed(new_out_buf);
            }
            self.outputs_need_reallocation = false;
        } else {
            self.outputs_need_reallocation = true;
        }

        Ok(self)
    }

    /// Computes the packed byte size for a `LiteRtRankedTensorType`.
    fn packed_byte_size_for_type(
        tensor_type: &crate::bindings::LiteRtRankedTensorType,
    ) -> Result<usize, Error> {
        let rank = tensor_type.layout.rank() as usize;
        let dims = tensor_type
            .layout
            .dimensions
            .get(..rank)
            .ok_or_else(|| {
                Error::new(
                    ErrorCause::ResizeCompiledModelInputBuffer,
                    crate::bindings::LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
                )
            })?;
        let mut num_elements: usize = 1;
        for &dim in dims {
            let dim_usize = usize::try_from(dim).map_err(|_| {
                Error::new(
                    ErrorCause::ResizeCompiledModelInputBuffer,
                    crate::bindings::LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
                )
            })?;
            num_elements = num_elements.checked_mul(dim_usize).ok_or_else(|| {
                Error::new(
                    ErrorCause::ResizeCompiledModelInputBuffer,
                    crate::bindings::LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
                )
            })?;
        }
        let elem_bytes = match ElementType::from_c_enum(tensor_type.element_type)? {
            ElementType::Bool | ElementType::Int8 | ElementType::UInt8 => 1,
            ElementType::Int16
            | ElementType::UInt16
            | ElementType::Float16
            | ElementType::BFloat16 => 2,
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => 4,
            ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float64
            | ElementType::Complex64 => 8,
            ElementType::Complex128 => 16,
            _ => {
                return Err(Error::new(
                    ErrorCause::ResizeCompiledModelInputBuffer,
                    crate::bindings::LiteRtStatus_kLiteRtStatusErrorUnsupported,
                ))
            }
        };
        num_elements.checked_mul(elem_bytes).ok_or_else(|| {
            Error::new(
                ErrorCause::ResizeCompiledModelInputBuffer,
                crate::bindings::LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )
        })
    }

    /// Completes any deferred output buffer reallocation from an earlier multi-input
    /// [`Self::resize_input`] call while holding the compiled model lock.
    fn ensure_outputs_reallocated(&mut self) -> Result<(), Error> {
        if !self.outputs_need_reallocation {
            return Ok(());
        }
        let num_outputs = self.outputs.len();
        if num_outputs == 0 {
            self.outputs_need_reallocation = false;
            return Ok(());
        }
        let signature = self.executor.model.signature(self.signature_index)?;
        let new_output_bufs = {
            let _model_guard = self.executor.compiled_model.lock_model();
            for (in_idx, entry) in self.inputs.iter().enumerate() {
                let in_dims = entry.slot.active_buffer().dimensions();
                if let Ok(in_layout) = self
                    .executor
                    .compiled_model
                    .input_tensor_layout_locked(self.signature_index, in_idx)
                {
                    let in_rank = usize::try_from(in_layout.rank())
                        .unwrap_or(0)
                        .min(in_layout.dimensions.len());
                    if &in_layout.dimensions[..in_rank] != in_dims {
                        self.executor.compiled_model.resize_input_tensor_buffer_locked(
                            self.signature_index,
                            InputOutputIndex::new(in_idx),
                            in_dims,
                        )?;
                    }
                }
            }
            let output_layouts = self
                .executor
                .compiled_model
                .output_tensor_layouts_locked(self.signature_index, num_outputs)?;
            let mut bufs = Vec::with_capacity(num_outputs);
            for (out_idx, out_layout) in output_layouts.into_iter().enumerate() {
                let out_req = self
                    .executor
                    .compiled_model
                    .output_buffer_requirements_locked(self.signature_index, out_idx)?;
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
                bufs.push(new_out_buf);
            }
            bufs
        };
        for (out_idx, new_out_buf) in new_output_bufs.into_iter().enumerate() {
            self.outputs[out_idx].slot.replace_managed(new_out_buf);
        }
        self.outputs_need_reallocation = false;
        Ok(())
    }

    /// Executes inference synchronously for this signature using the current input/output buffers.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining, or an [`Error`] (`ErrorCause::RunCompiledModel`) if
    /// inference execution fails or if any participating buffer has an active host-memory lock
    /// guard or is in use by another concurrent run.
    pub fn run(&mut self) -> Result<&mut Self, Error> {
        self.check_pending_event_error()?;
        self.ensure_outputs_reallocated()?;
        self.executor.compiled_model.run_with_iter(
            self.signature_index,
            self.inputs.iter().map(|entry| entry.slot.active_buffer()),
            self.outputs.iter().map(|entry| entry.slot.active_buffer()),
        )?;
        Ok(self)
    }

    /// Executes inference asynchronously (if supported by the hardware accelerator) and returns an
    /// [`AsyncExecution`] guard that borrows `self` mutably until completion.
    ///
    /// In the underlying `LiteRT` C++ runtime, interpreter invocation runs synchronously on the
    /// calling thread; when supported by a hardware accelerator delegate (e.g. GPU or NPU), the
    /// delegate attaches synchronization events (`LiteRtEvent`, such as sync fences) to the output
    /// buffers instead of blocking on hardware queue completion.
    ///
    /// # Returns
    ///
    /// An [`AsyncExecution`] guard holding `&mut self` while output events are pending, or an
    /// [`Error`] (`ErrorCause::RunCompiledModel`) if execution fails or if any participating buffer
    /// has an active host-memory lock guard or is in use by another concurrent run.
    #[cfg(async_support)]
    pub fn run_async(&mut self) -> Result<AsyncExecution<'_, 'model, 'ext>, Error> {
        self.check_pending_event_error()?;
        self.ensure_outputs_reallocated()?;
        let has_pending_output_events = self.executor.compiled_model.run_async_with_iter(
            self.signature_index,
            self.inputs.iter().map(|entry| entry.slot.active_buffer()),
            self.outputs.iter().map(|entry| entry.slot.active_buffer()),
        )?;
        self.has_unawaited_output_events.set(has_pending_output_events);
        Ok(AsyncExecution { runner: self, has_pending_output_events })
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
        self.check_pending_event_error()?;
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
        self.check_pending_event_error()?;
        let idx = self.resolve_output_index(id.into())?;
        self.outputs[idx].slot.active_buffer().lock_read::<T>()
    }
}

/// Guard representing an in-flight asynchronous execution on a [`SignatureRunner`].
///
/// Holds `&'runner mut SignatureRunner<'model, 'ext>`, preventing input mutation, buffer
/// unbinding, or concurrent runs through `runner` during normal use until output events are
/// awaited, [`Self::finish`] is called, or the guard is dropped.
///
/// Note on externally bound buffers and C++ event synchronization:
/// - Externally bound buffers ([`SignatureRunner::bind_input_buffer`] /
///   [`SignatureRunner::bind_output_buffer`]) are borrowed as `&'ext TensorBuffer`, so callers who
///   retain another reference to an external buffer can still call [`TensorBuffer`] methods on it
///   directly.
/// - In the underlying `LiteRT` C++ runtime, `LiteRtTensorBufferT::Lock` automatically waits on any
///   attached `LiteRtEvent` synchronization fence before mapping host memory, and [`TensorBuffer`]
///   also waits on any attached event in `Drop::drop` before calling `LiteRtDestroyTensorBuffer`.
///   Therefore, even if an external buffer is locked directly or if an `AsyncExecution` guard is
///   leaked via `std::mem::forget`, host-memory access and buffer destruction remain memory-safe.
#[cfg(async_support)]
#[must_use = "dropping AsyncExecution immediately blocks until execution completes; await outputs"]
pub struct AsyncExecution<'runner, 'model, 'ext> {
    /// Mutable borrow of the [`SignatureRunner`] whose asynchronous execution is in flight.
    pub(crate) runner: &'runner mut SignatureRunner<'model, 'ext>,
    /// Whether the hardware accelerator delegate attached synchronization events (`LiteRtEvent`)
    /// to the output buffers for asynchronous hardware execution.
    pub(crate) has_pending_output_events: bool,
}

#[cfg(async_support)]
impl<'runner, 'model, 'ext> AsyncExecution<'runner, 'model, 'ext> {
    /// Returns `true` if the hardware accelerator delegate attached synchronization events
    /// (`LiteRtEvent`, such as sync fences) to the output buffers for asynchronous hardware
    /// execution; `false` if execution completed completely synchronously during `run_async()`.
    ///
    /// # Returns
    ///
    /// `true` if output buffers have pending asynchronous hardware events; `false` otherwise.
    #[must_use]
    pub const fn is_parallel(&self) -> bool {
        self.has_pending_output_events
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

    /// Explicitly waits for and clears any remaining synchronization events across all output
    /// buffers, returning the first [`Error`] encountered (if any) after attempting every output
    /// buffer.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all output buffer events signaled and were cleared, or the first [`Error`]
    /// encountered while waiting on or clearing any output buffer's synchronization event.
    pub fn finish(self) -> Result<(), Error> {
        let res = self.runner.wait_all_output_events_blocking();
        std::mem::forget(self);
        res
    }
}

#[cfg(async_support)]
impl Drop for AsyncExecution<'_, '_, '_> {
    /// Blocks until any pending synchronization events on the signature's output buffers signal
    /// and clears them, recording the first error (if any) on [`SignatureRunner`] after attempting
    /// all output buffers so subsequent runner operations surface the error.
    fn drop(&mut self) {
        if let Err(err) = self.runner.wait_all_output_events_blocking() {
            self.runner.pending_event_error.set(Some(err));
        }
    }
}

impl Drop for SignatureRunner<'_, '_> {
    /// Ensures any unawaited asynchronous output buffer events (including on externally bound
    /// output buffers if an [`AsyncExecution`] guard was leaked via `std::mem::forget`) are
    /// awaited before input buffers and the backing `CompiledModel` / `Model` / `Environment` are
    /// dropped.
    fn drop(&mut self) {
        #[cfg(async_support)]
        if self.has_unawaited_output_events.get() || self.pending_event_error.get().is_some() {
            let _ = self.wait_all_output_events_blocking();
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
            use std::pin::pin;
            use std::sync::Arc;
            use std::task::{Context, Poll, Wake, Waker};

            struct NoopWake;
            impl Wake for NoopWake {
                fn wake(self: Arc<Self>) {}
            }

            /// Minimal single-threaded spin-executor for polling futures in unit tests.
            fn block_on<F: Future>(f: F) -> F::Output {
                let waker = Waker::from(Arc::new(NoopWake));
                let mut cx = Context::from_waker(&waker);
                let mut f = pin!(f);
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
            async_exec.finish().expect("Failed to finish async execution");

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

            // Verify that a deferred event synchronization error recorded in `pending_event_error`
            // is surfaced on the next operation and cleared once reported.
            owned_runner.pending_event_error.set(Some(Error::new(
                ErrorCause::WaitEvent,
                crate::bindings::LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            )));
            let deferred_err =
                owned_runner.run().err().expect("Expected deferred WaitEvent error on run()");
            assert_eq!(deferred_err.cause(), ErrorCause::WaitEvent);
            owned_runner.run().expect("Subsequent run() should succeed after error is taken");
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
            let mut m1 =
                Model::create_model_from_file(&env1, model_path).expect("Failed to load m1");
            let cm1 = CompiledModel::create(
                &env1,
                &mut m1,
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

        // Resize a second time from `[2, 2, 3]` to `[4, 2, 3]` (24 elements), where the
        // intermediate `[4, 2, 3]` vs `[2, 2, 3]` shapes are not broadcast-compatible until
        // both inputs have been resized.
        let in0_24: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let in1_24: Vec<f32> = (0..24).map(|x| (x * 5) as f32).collect();
        runner
            .resize_input(0, &[4, 2, 3])
            .expect("Failed to resize input 0 to [4, 2, 3]")
            .resize_input(1, &[4, 2, 3])
            .expect("Failed to resize input 1 to [4, 2, 3]")
            .set_input(0, &in0_24)
            .expect("Failed to set dynamic input 0 (24)")
            .set_input(1, &in1_24)
            .expect("Failed to set dynamic input 1 (24)")
            .run()
            .expect("Failed to run dynamic model (24)");
        let out_24 = runner.output_vec::<f32>(0).expect("Failed to read dynamic output (24)");
        assert_eq!(out_24.len(), 24);
        for i in 0..24 {
            assert!((out_24[i] - (in0_24[i] + in1_24[i])).abs() < 1e-4);
        }
    }
}
