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

//! Managed [`TensorBuffer`] allocation, host memory locking, and asynchronous event
//! synchronization.

#![allow(non_upper_case_globals)]

use std::ffi::c_void;
#[cfg(async_support)]
use std::future::Future;
use std::marker::PhantomData;
use std::mem;
#[cfg(async_support)]
use std::pin::Pin;
use std::sync::{Mutex, MutexGuard};
#[cfg(async_support)]
use std::task::{Context, Poll};

use crate::bindings::*;
use crate::call_check_status;
use crate::environment::Environment;
use crate::error::{Error, ErrorCause};
use crate::tensor_buffer_lock::{TensorBufferReadLock, TensorBufferWriteLock, TensorElement};

/// Requirements for a tensor buffer.
///
/// This struct represents the requirements for a tensor buffer. It is used to determine the
/// supported buffer types and the buffer size.
pub struct TensorBufferRequirements<'a> {
    /// Required buffer size in bytes, eagerly queried while the parent [`CompiledModel`] lock is
    /// held so subsequent `ResizeInputTensor` or `Run` calls cannot invalidate a borrowed C
    /// `LiteRtTensorBufferRequirements` pointer.
    buffer_size: usize,
    /// Supported [`TensorBufferType`] values for this tensor, eagerly queried while the parent
    /// [`CompiledModel`] lock is held.
    supported_types: Vec<TensorBufferType>,
    /// Binds the lifetime of these requirements to the parent compiled model.
    _phantom: PhantomData<&'a LiteRtTensorBufferRequirements>,
}

impl<'a> TensorBufferRequirements<'a> {
    /// Eagerly extracts the required buffer size and supported buffer types from a valid
    /// `raw_requirements` pointer while the caller holds the parent [`CompiledModel`] lock.
    ///
    /// # Safety
    ///
    /// `raw_requirements` must be a valid, non-null `LiteRtTensorBufferRequirements` handle that
    /// remains live and unmutated for the duration of this call (e.g., protected by the parent
    /// `CompiledModel`'s mutex).
    pub(crate) unsafe fn from_raw(
        raw_requirements: LiteRtTensorBufferRequirements,
    ) -> Result<Self, Error> {
        let mut buffer_size: usize = 0;
        call_check_status!(
            // SAFETY: Guaranteed by caller contract on `from_raw`.
            unsafe {
                LiteRtGetTensorBufferRequirementsBufferSize(raw_requirements, &mut buffer_size)
            },
            ErrorCause::GetTensorBufferRequirementsBufferSize
        );
        let mut num_supported_types: i32 = 0;
        call_check_status!(
            // SAFETY: Guaranteed by caller contract on `from_raw`.
            unsafe {
                LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                    raw_requirements,
                    &mut num_supported_types,
                )
            },
            ErrorCause::GetNumTensorBufferRequirementsSupportedBufferTypes
        );
        let cap = usize::try_from(num_supported_types).unwrap_or(0);
        let mut supported_types = Vec::with_capacity(cap);
        for i in 0..num_supported_types {
            let mut ttype = LiteRtTensorBufferType_kLiteRtTensorBufferTypeUnknown;
            call_check_status!(
                // SAFETY: Guaranteed by caller contract on `from_raw`.
                unsafe {
                    LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                        raw_requirements,
                        i,
                        &mut ttype,
                    )
                },
                ErrorCause::GetTensorBufferRequirementsSupportedTensorBufferType
            );
            supported_types.push(TensorBufferType::from_c_enum(ttype)?);
        }
        Ok(Self { buffer_size, supported_types, _phantom: PhantomData {} })
    }

    /// Returns the size of the buffer in bytes.
    ///
    /// # Returns
    ///
    /// The required buffer size in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::GetTensorBufferRequirementsBufferSize`) if querying the
    /// buffer size fails.
    pub fn buffer_size(&self) -> Result<usize, Error> {
        Ok(self.buffer_size)
    }

    /// Ensures the required buffer size is at least `min_size` bytes.
    pub(crate) fn set_min_buffer_size(&mut self, min_size: usize) {
        if self.buffer_size < min_size {
            self.buffer_size = min_size;
        }
    }

    /// Returns the supported tensor buffer types.
    ///
    /// # Returns
    ///
    /// A vector of [`TensorBufferType`]s supported by this tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if querying the count or values of supported buffer types fails, or if an
    /// unrecognized buffer type enum value is returned.
    pub fn supported_types(&self) -> Result<Vec<TensorBufferType>, Error> {
        Ok(self.supported_types.clone())
    }
}

/// The element type of a tensor buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// Unspecified element type.
    None,
    /// 8-bit boolean (1 byte per element).
    ///
    /// Note: Because Rust's `bool` requires every byte in `&[bool]` to be strictly `0x00` or
    /// `0x01` (whereas foreign C/C++ kernels or non-host buffers can contain arbitrary bytes),
    /// `bool` does not implement [`TensorElement`]. Read and write `ElementType::Bool` buffers as
    /// `u8` (or `i8`) and convert explicitly via `byte != 0` / `u8::from(flag)`.
    Bool,
    /// 4-bit signed integer.
    Int4,
    /// 8-bit signed integer (`i8`).
    Int8,
    /// 16-bit signed integer (`i16`).
    Int16,
    /// 32-bit signed integer (`i32`).
    Int32,
    /// 64-bit signed integer (`i64`).
    Int64,
    /// 8-bit unsigned integer (`u8`).
    UInt8,
    /// 16-bit unsigned integer (`u16`).
    UInt16,
    /// 32-bit unsigned integer (`u32`).
    UInt32,
    /// 64-bit unsigned integer (`u64`).
    UInt64,
    /// 16-bit IEEE 754 half-precision floating-point.
    Float16,
    /// 16-bit brain floating-point (`bfloat16`).
    BFloat16,
    /// 32-bit IEEE 754 single-precision floating-point (`f32`).
    Float32,
    /// 64-bit IEEE 754 double-precision floating-point (`f64`).
    Float64,
    /// 64-bit complex number (two 32-bit floats).
    Complex64,
    /// 128-bit complex number (two 64-bit floats).
    Complex128,
    /// TensorFlow resource handle.
    TfResource,
    /// TensorFlow variable-length string.
    TfString,
    /// TensorFlow dynamic variant type.
    TfVariant,
}

impl ElementType {
    /// Converts the [`ElementType`] into its corresponding `LiteRtElementType` C enum constant.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn to_c_enum(self) -> LiteRtElementType {
        match self {
            Self::None => LiteRtElementType_kLiteRtElementTypeNone,
            Self::Bool => LiteRtElementType_kLiteRtElementTypeBool,
            Self::Int4 => LiteRtElementType_kLiteRtElementTypeInt4,
            Self::Int8 => LiteRtElementType_kLiteRtElementTypeInt8,
            Self::Int16 => LiteRtElementType_kLiteRtElementTypeInt16,
            Self::Int32 => LiteRtElementType_kLiteRtElementTypeInt32,
            Self::Int64 => LiteRtElementType_kLiteRtElementTypeInt64,
            Self::UInt8 => LiteRtElementType_kLiteRtElementTypeUInt8,
            Self::UInt16 => LiteRtElementType_kLiteRtElementTypeUInt16,
            Self::UInt32 => LiteRtElementType_kLiteRtElementTypeUInt32,
            Self::UInt64 => LiteRtElementType_kLiteRtElementTypeUInt64,
            Self::Float16 => LiteRtElementType_kLiteRtElementTypeFloat16,
            Self::BFloat16 => LiteRtElementType_kLiteRtElementTypeBFloat16,
            Self::Float32 => LiteRtElementType_kLiteRtElementTypeFloat32,
            Self::Float64 => LiteRtElementType_kLiteRtElementTypeFloat64,
            Self::Complex64 => LiteRtElementType_kLiteRtElementTypeComplex64,
            Self::Complex128 => LiteRtElementType_kLiteRtElementTypeComplex128,
            Self::TfResource => LiteRtElementType_kLiteRtElementTypeTfResource,
            Self::TfString => LiteRtElementType_kLiteRtElementTypeTfString,
            Self::TfVariant => LiteRtElementType_kLiteRtElementTypeTfVariant,
        }
    }

    /// Converts a `LiteRtElementType` C enum constant into an [`ElementType`].
    pub(crate) fn from_c_enum(enum_value: LiteRtElementType) -> Result<ElementType, Error> {
        match enum_value {
            LiteRtElementType_kLiteRtElementTypeNone => Ok(Self::None),
            LiteRtElementType_kLiteRtElementTypeBool => Ok(Self::Bool),
            LiteRtElementType_kLiteRtElementTypeInt4 => Ok(Self::Int4),
            LiteRtElementType_kLiteRtElementTypeInt8 => Ok(Self::Int8),
            LiteRtElementType_kLiteRtElementTypeInt16 => Ok(Self::Int16),
            LiteRtElementType_kLiteRtElementTypeInt32 => Ok(Self::Int32),
            LiteRtElementType_kLiteRtElementTypeInt64 => Ok(Self::Int64),
            LiteRtElementType_kLiteRtElementTypeUInt8 => Ok(Self::UInt8),
            LiteRtElementType_kLiteRtElementTypeUInt16 => Ok(Self::UInt16),
            LiteRtElementType_kLiteRtElementTypeUInt32 => Ok(Self::UInt32),
            LiteRtElementType_kLiteRtElementTypeUInt64 => Ok(Self::UInt64),
            LiteRtElementType_kLiteRtElementTypeFloat16 => Ok(Self::Float16),
            LiteRtElementType_kLiteRtElementTypeBFloat16 => Ok(Self::BFloat16),
            LiteRtElementType_kLiteRtElementTypeFloat32 => Ok(Self::Float32),
            LiteRtElementType_kLiteRtElementTypeFloat64 => Ok(Self::Float64),
            LiteRtElementType_kLiteRtElementTypeComplex64 => Ok(Self::Complex64),
            LiteRtElementType_kLiteRtElementTypeComplex128 => Ok(Self::Complex128),
            LiteRtElementType_kLiteRtElementTypeTfResource => Ok(Self::TfResource),
            LiteRtElementType_kLiteRtElementTypeTfString => Ok(Self::TfString),
            LiteRtElementType_kLiteRtElementTypeTfVariant => Ok(Self::TfVariant),
            _ => Err(Error::new(
                ErrorCause::InvalidElementTypeEnumValue,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )),
        }
    }

    /// Returns `true` if `T` is layout-compatible with `self`.
    ///
    /// # Returns
    ///
    /// `true` if `T` can be safely read from or written to a [`TensorBuffer`] of element type
    /// `self`, `false` otherwise.
    #[must_use]
    pub fn is_compatible<T: TensorElement>(self) -> bool {
        T::is_compatible(self)
    }
}

/// Backing memory type of a [`TensorBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorBufferType {
    /// Unknown or unspecified buffer type.
    Unknown,
    /// Standard CPU host memory.
    HostMemory,
    /// Android Hardware Buffer (`AHardwareBuffer`).
    Ahwb,
    /// Android ION memory buffer.
    Ion,
    /// Linux DMA-BUF memory buffer.
    DmaBuf,
    /// Qualcomm `FastRPC` memory buffer.
    FastRpc,
    /// OpenGL buffer object.
    GlBuffer,
    /// OpenGL texture object.
    GlTexture,
    /// `OpenCL` buffer object.
    OpenClBuffer,
    /// `OpenCL` FP16 buffer object.
    OpenClBufferFp16,
    /// `OpenCL` texture object.
    OpenClTexture,
    /// `OpenCL` FP16 texture object.
    OpenClTextureFp16,
    /// Packed `OpenCL` buffer object.
    OpenClBufferPacked,
}

impl TensorBufferType {
    /// Converts the tensor buffer type variant into its corresponding `LiteRtTensorBufferType` C
    /// enum constant.
    ///
    /// # Returns
    ///
    /// The `LiteRtTensorBufferType` C enum value corresponding to `self`.
    #[allow(clippy::wrong_self_convention)]
    #[must_use]
    pub fn to_c_enum(&self) -> LiteRtTensorBufferType {
        match self {
            Self::Unknown => LiteRtTensorBufferType_kLiteRtTensorBufferTypeUnknown,
            Self::HostMemory => LiteRtTensorBufferType_kLiteRtTensorBufferTypeHostMemory,
            Self::Ahwb => LiteRtTensorBufferType_kLiteRtTensorBufferTypeAhwb,
            Self::Ion => LiteRtTensorBufferType_kLiteRtTensorBufferTypeIon,
            Self::DmaBuf => LiteRtTensorBufferType_kLiteRtTensorBufferTypeDmaBuf,
            Self::FastRpc => LiteRtTensorBufferType_kLiteRtTensorBufferTypeFastRpc,
            Self::GlBuffer => LiteRtTensorBufferType_kLiteRtTensorBufferTypeGlBuffer,
            Self::GlTexture => LiteRtTensorBufferType_kLiteRtTensorBufferTypeGlTexture,
            Self::OpenClBuffer => LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBuffer,
            Self::OpenClBufferFp16 => {
                LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBufferFp16
            }
            Self::OpenClTexture => LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClTexture,
            Self::OpenClTextureFp16 => {
                LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClTextureFp16
            }
            Self::OpenClBufferPacked => {
                LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBufferPacked
            }
        }
    }

    /// Converts a `LiteRtTensorBufferType` C enum value into a [`TensorBufferType`].
    ///
    /// # Arguments
    ///
    /// * `enum_value` - The raw `LiteRtTensorBufferType` C enum value.
    ///
    /// # Returns
    ///
    /// The matching [`TensorBufferType`] variant.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::InvalidTensorBufferTypeEnumValue`) if `enum_value` is not
    /// recognized.
    pub fn from_c_enum(enum_value: LiteRtTensorBufferType) -> Result<TensorBufferType, Error> {
        match enum_value {
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeUnknown => Ok(Self::Unknown),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeHostMemory => Ok(Self::HostMemory),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeAhwb => Ok(Self::Ahwb),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeIon => Ok(Self::Ion),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeDmaBuf => Ok(Self::DmaBuf),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeFastRpc => Ok(Self::FastRpc),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeGlBuffer => Ok(Self::GlBuffer),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeGlTexture => Ok(Self::GlTexture),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBuffer => Ok(Self::OpenClBuffer),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBufferFp16 => {
                Ok(Self::OpenClBufferFp16)
            }
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClTexture => Ok(Self::OpenClTexture),
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClTextureFp16 => {
                Ok(Self::OpenClTextureFp16)
            }
            LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBufferPacked => {
                Ok(Self::OpenClBufferPacked)
            }
            _ => Err(Error::new(
                ErrorCause::InvalidTensorBufferTypeEnumValue,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )),
        }
    }
}

/// Tracks whether a [`TensorBuffer`] is currently unlocked, locked by a Rust host-slice guard
/// ([`TensorBufferReadLock`] or [`TensorBufferWriteLock`]), or in use by [`CompiledModel`]
/// inference execution (`run` / `run_async`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferAccessState {
    /// No active Rust host-memory lock guard or inference execution is accessing this buffer.
    Idle,
    /// A [`TensorBufferReadLock`] or [`TensorBufferWriteLock`] guard currently holds a host slice
    /// (`&[T]` or `&mut [T]`) over this buffer's memory.
    HostLocked,
    /// An active `LiteRtRunCompiledModel` or `LiteRtRunCompiledModelAsync` call is exclusively
    /// using this buffer.
    InUseByRun,
}

/// RAII guard that keeps a [`TensorBuffer`] in [`BufferAccessState::InUseByRun`] for the duration
/// of a synchronous `LiteRtRunCompiledModel` or `LiteRtRunCompiledModelAsync` invocation.
pub(crate) struct TensorBufferRunGuard<'a> {
    /// Borrowed reference to the [`TensorBuffer`] participating in the run.
    buffer: &'a TensorBuffer<'a>,
}

impl Drop for TensorBufferRunGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.buffer.lock_state_mutex();
        *state = BufferAccessState::Idle;
    }
}

/// A managed tensor buffer holding input or output tensor data for model execution.
pub struct TensorBuffer<'a> {
    /// Underlying C `LiteRtTensorBuffer` handle owned by this struct.
    pub(crate) raw_tensor_buffer: LiteRtTensorBuffer,
    /// Scalar element type of this tensor buffer.
    element_type: ElementType,
    /// Shape dimensions of this tensor buffer.
    dimensions: Vec<i32>,
    /// Tracks active [`TensorBufferReadLock`] / [`TensorBufferWriteLock`] guards and active
    /// `CompiledModel::run` invocations, serializing all access to `raw_tensor_buffer`'s lock
    /// state (`LiteRtTensorBufferT::is_locked_`).
    lock_mutex: Mutex<BufferAccessState>,
    /// Optional lifetime marker for borrowed buffer contexts.
    _phantom: PhantomData<&'a LiteRtTensorBuffer>,
}

impl<'a> TensorBuffer<'a> {
    /// Acquires the internal mutex tracking active lock guards and run state on
    /// `self.raw_tensor_buffer`.
    pub(crate) fn lock_state_mutex(&self) -> MutexGuard<'_, BufferAccessState> {
        self.lock_mutex.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Marks this buffer as exclusively participating in an active `CompiledModel::run` or
    /// `run_async` call, returning [`ErrorCause::RunCompiledModel`]
    /// (`kLiteRtStatusErrorRuntimeFailure`) if a [`TensorBufferReadLock`] /
    /// [`TensorBufferWriteLock`] guard or another concurrent run is currently using `self`.
    pub(crate) fn acquire_for_run(&self) -> Result<TensorBufferRunGuard<'_>, Error> {
        let mut state = self.lock_state_mutex();
        match *state {
            BufferAccessState::Idle => {
                *state = BufferAccessState::InUseByRun;
                Ok(TensorBufferRunGuard { buffer: self })
            }
            BufferAccessState::HostLocked | BufferAccessState::InUseByRun => Err(Error::new(
                ErrorCause::RunCompiledModel,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            )),
        }
    }

    /// Allocates and zero-initializes a new managed [`TensorBuffer`] in the given [`Environment`].
    pub(crate) fn new(
        environment: &Environment,
        tensor_type: &LiteRtRankedTensorType,
        buffer_type: &TensorBufferType,
        buffer_size: usize,
        element_type: ElementType,
    ) -> Result<TensorBuffer<'a>, Error> {
        let mut buffer_ptr: *mut LiteRtTensorBufferT = std::ptr::null_mut();
        {
            let _env_guard = environment.lock_env();
            call_check_status!(
                // SAFETY: `environment.raw_environment` is a valid `LiteRtEnvironment` serialized
                // by `_env_guard`, `tensor_type` is a valid `LiteRtRankedTensorType` reference,
                // and `buffer_ptr` is a valid out-ptr.
                unsafe {
                    LiteRtCreateManagedTensorBuffer(
                        environment.raw_environment,
                        buffer_type.to_c_enum(),
                        tensor_type as *const LiteRtRankedTensorType,
                        buffer_size,
                        &mut buffer_ptr,
                    )
                },
                ErrorCause::CreateManagedTensorBuffer
            );
        }
        // Zero-initialize managed buffers on creation when supported by the backing buffer type.
        // SAFETY: `buffer_ptr` is a valid `LiteRtTensorBuffer` returned by
        // `LiteRtCreateManagedTensorBuffer`.
        let clear_status = unsafe { LiteRtClearTensorBuffer(buffer_ptr) };
        if clear_status != LiteRtStatus_kLiteRtStatusOk
            && clear_status != LiteRtStatus_kLiteRtStatusErrorUnsupported
        {
            // SAFETY: `buffer_ptr` was just created above and has not been wrapped in
            // `TensorBuffer` yet, so destroying it on error cleanup cannot double-free.
            unsafe {
                LiteRtDestroyTensorBuffer(buffer_ptr);
            }
            return Err(Error::new(ErrorCause::CreateManagedTensorBuffer, clear_status));
        }
        let rank = usize::try_from(tensor_type.layout.rank())
            .unwrap_or(0)
            .min(tensor_type.layout.dimensions.len());
        let dimensions = tensor_type.layout.dimensions[..rank].to_vec();
        Ok(TensorBuffer {
            raw_tensor_buffer: buffer_ptr,
            dimensions,
            element_type,
            lock_mutex: Mutex::new(BufferAccessState::Idle),
            _phantom: PhantomData {},
        })
    }

    /// Returns the element type of the tensor buffer.
    ///
    /// # Returns
    ///
    /// The [`ElementType`] of elements stored in this buffer.
    #[must_use]
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Returns dimensions of the tensor buffer.
    ///
    /// # Returns
    ///
    /// A slice of dimension sizes (`&[i32]`) for this buffer's layout.
    #[must_use]
    pub fn dimensions(&self) -> &[i32] {
        &self.dimensions
    }

    /// Locks the tensor buffer in read mode and returns a zero-copy RAII guard implementing
    /// `Deref<Target = [T]>`.
    ///
    /// Note that the underlying `LiteRT` runtime enforces exclusive lock semantics on a
    /// `LiteRtTensorBuffer`: attempting to call `lock_read` or `lock_write` while a lock guard on
    /// the same buffer is still alive, or while the buffer is in use by an active
    /// `CompiledModel::run` / `run_async` invocation, returns `Err`
    /// (`ErrorCause::LockTensorBufferRead`).
    ///
    /// # Returns
    ///
    /// A [`TensorBufferReadLock`] guard dereferencing to `&[T]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `T` is incompatible with `self.element_type()`
    /// (`ErrorCause::IncompatibleReadType`) or if locking the underlying buffer fails
    /// (`ErrorCause::LockTensorBufferRead`).
    pub fn lock_read<T: TensorElement>(&self) -> Result<TensorBufferReadLock<'_, T>, Error> {
        if !T::is_compatible(self.element_type) {
            return Err(Error::new(
                ErrorCause::IncompatibleReadType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        let byte_size = self.packed_size()?;
        let len = byte_size / mem::size_of::<T>();
        let mut data: *mut c_void = std::ptr::null_mut();
        {
            let mut state = self.lock_state_mutex();
            if *state != BufferAccessState::Idle {
                return Err(Error::new(
                    ErrorCause::LockTensorBufferRead,
                    LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
                ));
            }
            call_check_status!(
                // SAFETY:
                // - `self.raw_tensor_buffer` is a valid, non-null `LiteRtTensorBuffer` owned by
                //   `self`.
                // - `state` serializes `LiteRtLockTensorBuffer` / `LiteRtUnlockTensorBuffer` and
                //   `CompiledModel::run` calls on this buffer across threads, and `*state == Idle`
                //   guarantees no other lock guard or inference run is accessing the buffer.
                // - `&mut data` is a valid stack out-pointer populated with the mapped host memory
                //   address when `LiteRtLockTensorBuffer` returns `kLiteRtStatusOk`.
                unsafe {
                    LiteRtLockTensorBuffer(
                        self.raw_tensor_buffer,
                        &mut data,
                        LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeRead,
                    )
                },
                ErrorCause::LockTensorBufferRead
            );
            if len > 0 && (data.is_null() || !(data as usize).is_multiple_of(mem::align_of::<T>()))
            {
                // SAFETY: `self.raw_tensor_buffer` was just locked above while holding `state`.
                unsafe {
                    let _ = LiteRtUnlockTensorBuffer(self.raw_tensor_buffer);
                }
                return Err(Error::new(
                    ErrorCause::LockTensorBufferRead,
                    LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
                ));
            }
            // In C++ (`LiteRtTensorBufferT::Lock`), locking a buffer with an attached
            // synchronization event automatically waits on the event (`event_->Wait(-1)`), but
            // does not clear `event_`. Clear any attached event now while `state` is held so
            // subsequent `CompiledModel::run` / `run_async` calls do not reject this buffer for
            // still having a signaled event attached.
            #[cfg(async_support)]
            {
                let mut has_event = false;
                // SAFETY: `self.raw_tensor_buffer` is valid and `state` serializes `event_` access.
                if unsafe { LiteRtHasTensorBufferEvent(self.raw_tensor_buffer, &mut has_event) }
                    == LiteRtStatus_kLiteRtStatusOk
                    && has_event
                {
                    // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
                    unsafe {
                        let _ = LiteRtClearTensorBufferEvent(self.raw_tensor_buffer);
                    }
                }
            }
            *state = BufferAccessState::HostLocked;
        }
        let ptr = if len == 0 { std::ptr::NonNull::dangling().as_ptr() } else { data as *const T };
        Ok(TensorBufferReadLock { buffer: self, ptr, len })
    }

    /// Locks the tensor buffer in write mode and returns a zero-copy RAII guard implementing
    /// `DerefMut<Target = [T]>`.
    ///
    /// Note that the underlying `LiteRT` runtime enforces exclusive lock semantics on a
    /// `LiteRtTensorBuffer`: attempting to call `lock_read` or `lock_write` while a lock guard on
    /// the same buffer is still alive, or while the buffer is in use by an active
    /// `CompiledModel::run` / `run_async` invocation, returns `Err`
    /// (`ErrorCause::LockTensorBufferWrite`).
    ///
    /// # Returns
    ///
    /// A [`TensorBufferWriteLock`] guard dereferencing to `&mut [T]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `T` is incompatible with `self.element_type()`
    /// (`ErrorCause::IncompatibleWriteType`) or if locking the underlying buffer fails
    /// (`ErrorCause::LockTensorBufferWrite`).
    pub fn lock_write<T: TensorElement>(&self) -> Result<TensorBufferWriteLock<'_, T>, Error> {
        if !T::is_compatible(self.element_type) {
            return Err(Error::new(
                ErrorCause::IncompatibleWriteType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        let byte_size = self.packed_size()?;
        let len = byte_size / mem::size_of::<T>();
        let mut data: *mut c_void = std::ptr::null_mut();
        {
            let mut state = self.lock_state_mutex();
            if *state != BufferAccessState::Idle {
                return Err(Error::new(
                    ErrorCause::LockTensorBufferWrite,
                    LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
                ));
            }
            call_check_status!(
                // SAFETY:
                // - `self.raw_tensor_buffer` is a valid, non-null `LiteRtTensorBuffer` owned by
                //   `self`.
                // - `state` serializes `LiteRtLockTensorBuffer` / `LiteRtUnlockTensorBuffer` and
                //   `CompiledModel::run` calls on this buffer across threads, and `*state == Idle`
                //   guarantees no other lock guard or inference run is accessing the buffer.
                // - `&mut data` is a valid stack out-pointer populated with the mapped host memory
                //   address when `LiteRtLockTensorBuffer` returns `kLiteRtStatusOk`.
                unsafe {
                    LiteRtLockTensorBuffer(
                        self.raw_tensor_buffer,
                        &mut data,
                        LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeWrite,
                    )
                },
                ErrorCause::LockTensorBufferWrite
            );
            if len > 0 && (data.is_null() || !(data as usize).is_multiple_of(mem::align_of::<T>()))
            {
                // SAFETY: `self.raw_tensor_buffer` was just locked above while holding `state`.
                unsafe {
                    let _ = LiteRtUnlockTensorBuffer(self.raw_tensor_buffer);
                }
                return Err(Error::new(
                    ErrorCause::LockTensorBufferWrite,
                    LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
                ));
            }
            // In C++ (`LiteRtTensorBufferT::Lock`), locking a buffer with an attached
            // synchronization event automatically waits on the event (`event_->Wait(-1)`), but
            // does not clear `event_`. Clear any attached event now while `state` is held so
            // subsequent `CompiledModel::run` / `run_async` calls do not reject this buffer for
            // still having a signaled event attached.
            #[cfg(async_support)]
            {
                let mut has_event = false;
                // SAFETY: `self.raw_tensor_buffer` is valid and `state` serializes `event_` access.
                if unsafe { LiteRtHasTensorBufferEvent(self.raw_tensor_buffer, &mut has_event) }
                    == LiteRtStatus_kLiteRtStatusOk
                    && has_event
                {
                    // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
                    unsafe {
                        let _ = LiteRtClearTensorBufferEvent(self.raw_tensor_buffer);
                    }
                }
            }
            *state = BufferAccessState::HostLocked;
        }
        let ptr = if len == 0 { std::ptr::NonNull::dangling().as_ptr() } else { data as *mut T };
        Ok(TensorBufferWriteLock { buffer: self, ptr, len })
    }

    /// Returns the packed size of the tensor buffer in bytes.
    ///
    /// # Returns
    ///
    /// The packed byte size of the tensor buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::GetTensorBufferPackedSize`) if querying the packed size
    /// fails.
    pub fn packed_size(&self) -> Result<usize, Error> {
        let mut size: usize = 0;
        call_check_status!(
            // SAFETY: self.raw_tensor_buffer is always valid, it's guaranteed to be initialized by
            // a wrapper function.
            unsafe { LiteRtGetTensorBufferPackedSize(self.raw_tensor_buffer, &mut size) },
            ErrorCause::GetTensorBufferPackedSize
        );
        Ok(size)
    }

    /// Writes data from `data` into the tensor buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of elements to copy into the tensor buffer. `T` must be compatible with
    ///   [`Self::element_type`], and `data.len()` must not exceed the buffer's element capacity.
    ///
    /// # Returns
    ///
    /// The number of bytes written to the tensor buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `T` is incompatible (`ErrorCause::IncompatibleWriteType`), `data`
    /// exceeds the buffer capacity (`ErrorCause::TensorBufferTooSmall`), or locking the buffer
    /// fails.
    pub fn write<T: TensorElement>(&self, data: &[T]) -> Result<usize, Error> {
        let mut lock = self.lock_write::<T>()?;
        if lock.len() < data.len() {
            return Err(Error::new(
                ErrorCause::TensorBufferTooSmall,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        lock[..data.len()].copy_from_slice(data);
        Ok(mem::size_of_val(data))
    }

    /// Reads data from the tensor buffer into `data`.
    ///
    /// # Arguments
    ///
    /// * `data` - Destination slice to receive the tensor buffer elements. `T` must be compatible
    ///   with [`Self::element_type`], and `data.len()` must be at least the number of elements in
    ///   the tensor buffer.
    ///
    /// # Returns
    ///
    /// The number of elements read from the tensor buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `T` is incompatible (`ErrorCause::IncompatibleReadType`), `data` is
    /// smaller than the buffer's element count (`ErrorCause::ReadBufferTooSmall`), or locking the
    /// buffer fails.
    pub fn read<T: TensorElement>(&self, data: &mut [T]) -> Result<usize, Error> {
        let lock = self.lock_read::<T>()?;
        if data.len() < lock.len() {
            return Err(Error::new(
                ErrorCause::ReadBufferTooSmall,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        let to_copy = lock.len();
        data[..to_copy].copy_from_slice(&lock[..to_copy]);
        Ok(to_copy)
    }

    /// Polls whether any synchronization event attached to this buffer has signaled, and if so,
    /// clears the event while holding `self.lock_state_mutex()`.
    ///
    /// Returns `Ok(true)` if there is no attached event or if the attached event has signaled and
    /// been cleared; returns `Ok(false)` if an attached event is still pending.
    #[cfg(async_support)]
    pub(crate) fn poll_event_signaled_and_clear(&self) -> Result<bool, Error> {
        let state = self.lock_state_mutex();
        if *state != BufferAccessState::Idle {
            return Err(Error::new(
                ErrorCause::GetTensorBufferEvent,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        let mut has_event = false;
        call_check_status!(
            // SAFETY: `self.raw_tensor_buffer` is a valid handle owned by `self`, and `state`
            // serializes access to `LiteRtTensorBufferT::event_` across threads.
            unsafe { LiteRtHasTensorBufferEvent(self.raw_tensor_buffer, &mut has_event) },
            ErrorCause::HasTensorBufferEvent
        );
        if !has_event {
            return Ok(true);
        }
        let mut raw_event: LiteRtEvent = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
            unsafe { LiteRtGetTensorBufferEvent(self.raw_tensor_buffer, &mut raw_event) },
            ErrorCause::GetTensorBufferEvent
        );
        if raw_event.is_null() {
            return Ok(true);
        }
        let mut is_signaled = false;
        // SAFETY: `raw_event` is valid while `state` prevents `LiteRtClearTensorBufferEvent` or
        // `LiteRtSetTensorBufferEvent` from running concurrently.
        let status = unsafe { LiteRtIsEventSignaled(raw_event, &mut is_signaled) };
        if status == LiteRtStatus_kLiteRtStatusOk {
            if is_signaled {
                call_check_status!(
                    // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
                    unsafe { LiteRtClearTensorBufferEvent(self.raw_tensor_buffer) },
                    ErrorCause::SetTensorBufferEvent
                );
                Ok(true)
            } else {
                Ok(false)
            }
        } else if status == LiteRtStatus_kLiteRtStatusErrorInvalidArgument
            || status == LiteRtStatus_kLiteRtStatusErrorUnsupported
        {
            // Event types such as `LiteRtEventTypeOpenCl` and `LiteRtEventTypeEglSyncFence` do not
            // implement `IsSignaled()` in C++ (`event.cc`), but do implement `Wait(-1)`. Fall back
            // to waiting and clearing the event.
            call_check_status!(
                // SAFETY: `raw_event` is valid while `state` is held.
                unsafe { LiteRtWaitEvent(raw_event, -1) },
                ErrorCause::WaitEvent
            );
            call_check_status!(
                // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
                unsafe { LiteRtClearTensorBufferEvent(self.raw_tensor_buffer) },
                ErrorCause::SetTensorBufferEvent
            );
            Ok(true)
        } else {
            Err(Error::new(ErrorCause::IsEventSignaled, status))
        }
    }

    /// Returns a future that resolves once any synchronization event attached to this buffer is
    /// signaled.
    #[cfg(async_support)]
    #[allow(dead_code)]
    pub(crate) fn wait_event(&self) -> WaitEvent<'_> {
        WaitEvent { buffer: self }
    }

    /// Blocks the current thread until any synchronization event attached to this buffer is
    /// signaled, and clears the event so the buffer can be reused in subsequent runs.
    #[cfg(async_support)]
    #[allow(dead_code)]
    pub(crate) fn wait_event_blocking(&self) -> Result<(), Error> {
        let state = self.lock_state_mutex();
        if *state != BufferAccessState::Idle {
            return Err(Error::new(
                ErrorCause::WaitEvent,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        let mut has_event = false;
        call_check_status!(
            // SAFETY: `self.raw_tensor_buffer` is valid and `state` serializes access to `event_`.
            unsafe { LiteRtHasTensorBufferEvent(self.raw_tensor_buffer, &mut has_event) },
            ErrorCause::HasTensorBufferEvent
        );
        if !has_event {
            return Ok(());
        }
        let mut raw_event: LiteRtEvent = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
            unsafe { LiteRtGetTensorBufferEvent(self.raw_tensor_buffer, &mut raw_event) },
            ErrorCause::GetTensorBufferEvent
        );
        if !raw_event.is_null() {
            // Call `LiteRtWaitEvent(raw_event, -1)` directly (matching C++
            // `LiteRtTensorBufferT::Lock` and `LiteRtCompiledModelT::Run`), because
            // `LiteRtIsEventSignaled` is unsupported on OpenCL and EGL events.
            call_check_status!(
                // SAFETY:
                // - `raw_event` is a valid, non-null `LiteRtEvent` borrowed from `self` while
                //   `state` prevents concurrent clearing or mutation of `event_`.
                // - `timeout_in_ms = -1` instructs the LiteRT runtime to block indefinitely until
                //   the event signals.
                unsafe { LiteRtWaitEvent(raw_event, -1) },
                ErrorCause::WaitEvent
            );
            call_check_status!(
                // SAFETY: `self.raw_tensor_buffer` is valid and `state` is held.
                unsafe { LiteRtClearTensorBufferEvent(self.raw_tensor_buffer) },
                ErrorCause::SetTensorBufferEvent
            );
        }
        Ok(())
    }

    /// Attaches a synchronization [`Event`] to this tensor buffer (used in unit tests).
    #[cfg(all(test, async_support))]
    fn set_event(&self, mut event: Event<'_>) -> Result<(), Error> {
        let state = self.lock_state_mutex();
        if *state != BufferAccessState::Idle {
            return Err(Error::new(
                ErrorCause::SetTensorBufferEvent,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        call_check_status!(
            // SAFETY: `self.raw_tensor_buffer` and `event.raw_event` are valid, and `state`
            // serializes access to `LiteRtTensorBufferT::event_`.
            unsafe { LiteRtSetTensorBufferEvent(self.raw_tensor_buffer, event.raw_event) },
            ErrorCause::SetTensorBufferEvent
        );
        event.owned = false;
        Ok(())
    }

    /// Reads data from the tensor buffer asynchronously into `data`.
    ///
    /// If the buffer has a synchronization event attached, the returned future waits for the event
    /// to signal (and clears it) before reading the buffer into `data`.
    ///
    /// # Arguments
    ///
    /// * `data` - Destination slice to receive the tensor buffer elements.
    ///
    /// # Returns
    ///
    /// A [`ReadAsync`] future that resolves to `Result<usize, Error>` with the number of elements
    /// read.
    #[cfg(async_support)]
    pub fn read_async<'b, T: TensorElement>(&self, data: &'b mut [T]) -> ReadAsync<'_, 'b, T> {
        ReadAsync { buffer: self, data: Some(data) }
    }
}

/// A synchronization event attached to or managed by a [`TensorBuffer`].
#[cfg(all(test, async_support))]
pub(crate) struct Event<'a> {
    /// Underlying C `LiteRtEvent` handle.
    pub(crate) raw_event: LiteRtEvent,
    /// Whether this wrapper owns `raw_event` and should destroy it on drop.
    owned: bool,
    /// Binds this event's lifetime to the parent [`TensorBuffer`] when borrowed.
    _phantom: PhantomData<&'a LiteRtEvent>,
}

#[cfg(all(test, async_support))]
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: self.raw_event is valid and owned.
            unsafe {
                LiteRtDestroyEvent(self.raw_event);
            }
        }
    }
}

/// Future returned by [`TensorBuffer::wait_event`] that resolves when the buffer's event signals.
#[cfg(async_support)]
pub(crate) struct WaitEvent<'a> {
    /// Reference to the [`TensorBuffer`] whose attached event is being awaited.
    buffer: &'a TensorBuffer<'a>,
}

#[cfg(async_support)]
impl Future for WaitEvent<'_> {
    type Output = Result<(), Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.buffer.poll_event_signaled_and_clear() {
            Ok(true) => Poll::Ready(Ok(())),
            Ok(false) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

/// Future returned by [`TensorBuffer::read_async`] that awaits any attached synchronization event
/// before copying tensor data into the caller's slice.
#[cfg(async_support)]
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct ReadAsync<'a, 'b, T> {
    /// Reference to the [`TensorBuffer`] being read.
    buffer: &'a TensorBuffer<'a>,
    /// Destination slice consumed once the read completes.
    data: Option<&'b mut [T]>,
}

#[cfg(async_support)]
impl<'a, 'b, T: TensorElement> Future for ReadAsync<'a, 'b, T> {
    type Output = Result<usize, Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.buffer.poll_event_signaled_and_clear() {
            Ok(true) => {
                let data = self.data.take().expect("Future polled after completion");
                Poll::Ready(self.buffer.read(data))
            }
            Ok(false) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl Drop for TensorBuffer<'_> {
    fn drop(&mut self) {
        // If a `TensorBufferReadLock` or `TensorBufferWriteLock` guard was leaked via
        // `std::mem::forget`, `&mut self` in `Drop` guarantees the leaked guard's borrow (`'buf`)
        // has already ended. Unlock the underlying C++ buffer and reset the state to `Idle` so
        // `wait_event_blocking()` can run and `LiteRtTensorBufferT::~LiteRtTensorBufferT` does not
        // destroy a locked buffer.
        {
            let mut state = self.lock_state_mutex();
            if *state == BufferAccessState::HostLocked {
                // SAFETY: `self.raw_tensor_buffer` is valid and `&mut self` guarantees no active
                // host slice reference (`&[T]` or `&mut [T]`) exists.
                unsafe {
                    let _ = LiteRtUnlockTensorBuffer(self.raw_tensor_buffer);
                }
                *state = BufferAccessState::Idle;
            }
        }
        // If an asynchronous hardware execution attached a synchronization event to this buffer
        // and the caller dropped the buffer (or leaked an `AsyncExecution` guard via
        // `std::mem::forget`) before awaiting the event, wait for the event to signal before
        // freeing the underlying buffer memory (`LiteRtTensorBufferT::~LiteRtTensorBufferT` in C++
        // does not wait on `event_` in its destructor).
        #[cfg(async_support)]
        {
            let _ = self.wait_event_blocking();
        }
        // SAFETY: `self.raw_tensor_buffer` is a valid, non-null `LiteRtTensorBuffer` owned by
        // `self`, any leaked host lock has been unlocked above, and any pending hardware event has
        // been awaited above.
        unsafe {
            LiteRtDestroyTensorBuffer(self.raw_tensor_buffer);
        }
    }
}

// SAFETY: `TensorBuffer` uniquely owns its `LiteRtTensorBuffer` handle and has no thread-local
// affinity, so transferring ownership across threads is safe.
unsafe impl Send for TensorBuffer<'_> {}

// SAFETY: Shared references `&TensorBuffer` serialize all host-memory locking
// (`LiteRtLockTensorBuffer` / `LiteRtUnlockTensorBuffer`), inference execution access
// (`TensorBufferRunGuard` during `CompiledModel::run` / `run_async`), and synchronization event
// access (`LiteRtHasTensorBufferEvent` / `LiteRtGetTensorBufferEvent` /
// `LiteRtClearTensorBufferEvent`) through `self.lock_mutex: Mutex<BufferAccessState>` (`Idle`,
// `HostLocked`, `InUseByRun`), preventing concurrent lock/unlock races, aliased host slices, and
// concurrent mutations during inference.
unsafe impl Sync for TensorBuffer<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(async_support)]
    use std::future::Future;
    #[cfg(async_support)]
    use std::pin::Pin;
    #[cfg(async_support)]
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    #[cfg(async_support)]
    fn block_on<F: Future>(mut f: F) -> F::Output {
        let raw_waker = RawWaker::new(std::ptr::null(), &VTABLE);
        // SAFETY: raw_waker is a valid RawWaker.
        let waker = unsafe { Waker::from_raw(raw_waker) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: f is a valid Future.
        let mut f = unsafe { Pin::new_unchecked(&mut f) };
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(res) => return res,
                Poll::Pending => {
                    std::thread::yield_now();
                }
            }
        }
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(std::ptr::null(), &VTABLE), // clone
            |_| {},                                       // wake
            |_| {},                                       // wake_by_ref
            |_| {},                                       // drop
        );
    }

    #[test]
    fn test_element_type_compatibility() {
        assert!(ElementType::Bool.is_compatible::<u8>());
        assert!(ElementType::Bool.is_compatible::<i8>());
        assert!(ElementType::Int8.is_compatible::<i8>());
        assert!(ElementType::UInt8.is_compatible::<u8>());
        assert!(ElementType::Int16.is_compatible::<i16>());
        assert!(ElementType::UInt16.is_compatible::<u16>());
        assert!(ElementType::Int32.is_compatible::<i32>());
        assert!(ElementType::UInt32.is_compatible::<u32>());
        assert!(ElementType::Int64.is_compatible::<i64>());
        assert!(ElementType::UInt64.is_compatible::<u64>());
        assert!(ElementType::Float32.is_compatible::<f32>());
        assert!(ElementType::Float64.is_compatible::<f64>());
        assert!(!ElementType::Bool.is_compatible::<u32>());
        assert!(!ElementType::Float32.is_compatible::<u32>());
        assert!(ElementType::Int32.is_compatible::<u32>());
        assert!(ElementType::UInt32.is_compatible::<i32>());
        assert!(!ElementType::Int32.is_compatible::<i16>());
        assert!(!ElementType::Int64.is_compatible::<i32>());
    }

    #[test]
    fn test_tensor_buffer_lock_read_write() {
        let env = crate::EnvironmentBuilder::build_default().unwrap();
        // SAFETY: layout is a C struct with pod fields.
        let mut layout: LiteRtLayout = unsafe { std::mem::zeroed() };
        layout.set_rank(1);
        layout.dimensions[0] = 4;

        let tensor_type = LiteRtRankedTensorType {
            element_type: LiteRtElementType_kLiteRtElementTypeFloat32,
            layout,
        };

        let buffer = TensorBuffer::new(
            &env,
            &tensor_type,
            &TensorBufferType::HostMemory,
            16,
            ElementType::Float32,
        )
        .unwrap();

        {
            let mut write_lock = buffer.lock_write::<f32>().unwrap();
            assert_eq!(write_lock.len(), 4);
            for (i, v) in write_lock.iter_mut().enumerate() {
                *v = (i as f32) * 2.5;
            }
        }

        {
            let read_lock = buffer.lock_read::<f32>().unwrap();
            assert_eq!(&*read_lock, &[0.0f32, 2.5, 5.0, 7.5]);
        }

        assert!(buffer.lock_read::<i32>().is_err());
        assert!(buffer.lock_write::<i32>().is_err());

        // Double-lock attempts while a lock is held return an error instead of UB or deadlock.
        {
            let read_lock = buffer.lock_read::<f32>().unwrap();
            let err_read = buffer.lock_read::<f32>().err().expect("Expected double-lock error");
            assert_eq!(err_read.cause(), ErrorCause::LockTensorBufferRead);
            let err_write = buffer.lock_write::<f32>().err().expect("Expected double-lock error");
            assert_eq!(err_write.cause(), ErrorCause::LockTensorBufferWrite);
            assert_eq!(&*read_lock, &[0.0f32, 2.5, 5.0, 7.5]);
        }

        // Write buffer too small vs read buffer too small.
        let err_write_small = buffer.write(&[1.0f32; 5]).unwrap_err();
        assert_eq!(err_write_small.cause(), ErrorCause::TensorBufferTooSmall);

        let mut short_out = [0.0f32; 2];
        let err_read_small = buffer.read(&mut short_out).unwrap_err();
        assert_eq!(err_read_small.cause(), ErrorCause::ReadBufferTooSmall);
    }

    #[test]
    fn test_bool_tensor_buffer_zero_initialized_and_read_write() {
        let env = crate::EnvironmentBuilder::build_default().unwrap();
        // SAFETY: `LiteRtLayout` is a plain-old-data C struct.
        let mut layout: LiteRtLayout = unsafe { std::mem::zeroed() };
        layout.set_rank(1);
        layout.dimensions[0] = 4;

        let tensor_type = LiteRtRankedTensorType {
            element_type: LiteRtElementType_kLiteRtElementTypeBool,
            layout,
        };

        let buffer = TensorBuffer::new(
            &env,
            &tensor_type,
            &TensorBufferType::HostMemory,
            4,
            ElementType::Bool,
        )
        .unwrap();
        assert_eq!(buffer.element_type(), ElementType::Bool);
        assert_eq!(buffer.dimensions(), &[4]);

        // `ElementType::Bool` buffers are accessed via `u8` (or `i8`) so any byte value from C/C++
        // or non-host buffers is valid without violating Rust's `bool` bit-validity invariant.
        {
            let initial_lock = buffer.lock_read::<u8>().unwrap();
            assert_eq!(&*initial_lock, &[0u8, 0, 0, 0]);
        }

        let flags = [true, false, true, false].map(u8::from);
        buffer.write(&flags).unwrap();
        let mut out = [0u8; 4];
        let count = buffer.read(&mut out).unwrap();
        assert_eq!(count, 4);
        assert_eq!(out.map(|b| b != 0), [true, false, true, false]);
    }

    #[test]
    #[cfg(async_support)]
    fn test_read_async_no_event() {
        let env = crate::EnvironmentBuilder::build_default().unwrap();

        // SAFETY: layout is a C struct with pod fields.
        let mut layout: LiteRtLayout = unsafe { std::mem::zeroed() };
        // Try to set rank. If it fails to compile, we will see the error.
        layout.set_rank(1);
        layout.dimensions[0] = 4;

        let tensor_type = LiteRtRankedTensorType {
            element_type: LiteRtElementType_kLiteRtElementTypeFloat32,
            layout,
        };

        let buffer = TensorBuffer::new(
            &env,
            &tensor_type,
            &TensorBufferType::HostMemory,
            16, // 4 * 4 bytes
            ElementType::Float32,
        )
        .unwrap();

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        buffer.write(&input_data).unwrap();

        let mut output_data = vec![0.0f32; 4];

        block_on(buffer.read_async(&mut output_data)).unwrap();

        assert_eq!(input_data, output_data);
    }

    #[test]
    #[cfg(async_support)]
    fn test_read_async_with_event() {
        // This test requires Linux to run as it uses Linux-specific sync fence FDs.

        extern "C" {
            fn pipe(fds: *mut i32) -> i32;
            fn write(fd: i32, buf: *const std::ffi::c_void, count: usize) -> isize;
            fn close(fd: i32) -> i32;
        }

        let env = crate::EnvironmentBuilder::build_default().unwrap();

        let mut fds = [0i32; 2];
        // SAFETY: fds is a valid pointer to a 2-element array. The pipe function
        // writes the file descriptors to the array.
        unsafe {
            if pipe(fds.as_mut_ptr()) != 0 {
                panic!("Failed to create pipe");
            }
        }
        let (rx, tx) = (fds[0], fds[1]);

        let mut raw_event: LiteRtEvent = std::ptr::null_mut();
        // SAFETY: env.raw_environment is valid as it's owned by env.
        // rx is a valid file descriptor as it's owned by this test and was created by pipe.
        // owns_fd is true as we want the event to own the file descriptor.
        let status = unsafe {
            LiteRtCreateEventFromSyncFenceFd(
                env.raw_environment,
                rx,
                true, // owns_fd
                &mut raw_event,
            )
        };

        if status != LiteRtStatus_kLiteRtStatusOk {
            if status == LiteRtStatus_kLiteRtStatusErrorUnsupported {
                println!("Sync fence event is not supported on this platform, skipping test.");
                // SAFETY: rx and tx are valid file descriptors and are owned by this test.
                unsafe {
                    close(rx);
                    close(tx);
                }
                return;
            }
            panic!("Failed to create event from sync fence fd: {}", status);
        }

        let event = Event { raw_event, owned: true, _phantom: PhantomData };

        // SAFETY: layout is a C struct with pod fields.
        let mut layout: LiteRtLayout = unsafe { std::mem::zeroed() };
        layout.set_rank(1);
        layout.dimensions[0] = 4;

        let tensor_type = LiteRtRankedTensorType {
            element_type: LiteRtElementType_kLiteRtElementTypeFloat32,
            layout,
        };

        let buffer = TensorBuffer::new(
            &env,
            &tensor_type,
            &TensorBufferType::HostMemory,
            16,
            ElementType::Float32,
        )
        .unwrap();

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        buffer.write(&input_data).unwrap();

        buffer.set_event(event).unwrap();

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            let val = 1u64;
            // Writing to the pipe to trigger the event to complete.
            // SAFETY: tx is valid as it's owned by this thread and was created by pipe.
            // data in write is a pointer to a valid u64, and 8 is the correct size.
            // close is safe as tx is owned by this thread.
            unsafe {
                write(tx, &val as *const u64 as *const std::ffi::c_void, 8);
                close(tx);
            }
        });

        let mut output_data = vec![0.0f32; 4];

        block_on(buffer.read_async(&mut output_data)).unwrap();

        assert_eq!(input_data, output_data);
    }
}
