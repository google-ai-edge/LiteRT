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

use std::any::TypeId;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;

use crate::bindings::*;
use crate::call_check_status;
use crate::environment::Environment;
use crate::error::{Error, ErrorCause};

/// Requirements for a tensor buffer.
///
/// This struct represents the requirements for a tensor buffer. It is used to determine the
/// supported buffer types and the buffer size.
pub struct TensorBufferRequirements<'a> {
    raw_requirements: LiteRtTensorBufferRequirements,
    _phantom: PhantomData<&'a LiteRtTensorBufferRequirements>,
}

impl<'a> TensorBufferRequirements<'a> {
    pub(crate) fn new(raw_requirements: LiteRtTensorBufferRequirements) -> Self {
        Self { raw_requirements: raw_requirements, _phantom: PhantomData {} }
    }

    /// Returns the size of the buffer in bytes.
    pub fn buffer_size(&self) -> Result<usize, Error> {
        let mut buffer_size: usize = 0;
        call_check_status!(
            // SAFETY: self.raw_requirements is always valid, it's guaranteed to be initialized by
            // a wrapper function.
            unsafe {
                LiteRtGetTensorBufferRequirementsBufferSize(self.raw_requirements, &mut buffer_size)
            },
            ErrorCause::GetTensorBufferRequirementsBufferSize
        );
        Ok(buffer_size)
    }

    /// Returns the supported tensor buffer types.
    pub fn supported_types(&self) -> Result<Vec<TensorBufferType>, Error> {
        let mut num_supported_types: i32 = 0;
        call_check_status!(
            // SAFETY: self.raw_requirements is always valid, it's guaranteed to be initialized by
            // a wrapper function.
            unsafe {
                LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                    self.raw_requirements,
                    &mut num_supported_types,
                )
            },
            ErrorCause::GetNumTensorBufferRequirementsSupportedBufferTypes
        );
        let mut result = Vec::with_capacity(num_supported_types as usize);
        for i in 0..num_supported_types {
            let mut ttype = LiteRtTensorBufferType_kLiteRtTensorBufferTypeUnknown;
            call_check_status!(
                // SAFETY: self.raw_requirements is always valid, it's guaranteed to be initialized by
                // a wrapper function.
                unsafe {
                    LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                        self.raw_requirements,
                        i,
                        &mut ttype,
                    )
                },
                ErrorCause::GetTensorBufferRequirementsSupportedTensorBufferType
            );
            result.push(TensorBufferType::from_c_enum(ttype)?);
        }
        Ok(result)
    }
}

/// The element type of a tensor buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementType {
    None,
    Bool,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    TfResource,
    TfString,
    TfVariant,
}

impl ElementType {
    #[allow(dead_code)]
    pub(crate) fn to_c_enum(&self) -> LiteRtElementType {
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

    fn is_compatible<T: 'static>(self) -> bool {
        let type_id = TypeId::of::<T>();
        if TypeId::of::<bool>() == type_id {
            self == Self::Bool
        } else if TypeId::of::<i8>() == type_id || TypeId::of::<i8>() == type_id {
            self == Self::Int8 || self == Self::UInt8
        } else if TypeId::of::<i16>() == type_id || TypeId::of::<i16>() == type_id {
            self == Self::Int16 || self == Self::UInt16
        } else if TypeId::of::<i32>() == type_id || TypeId::of::<i32>() == type_id {
            self == Self::Int32 || self == Self::UInt32
        } else if TypeId::of::<f32>() == type_id {
            self == Self::Float32
        } else if TypeId::of::<f64>() == type_id {
            self == Self::Float64
        } else {
            // TODO: Add support for other types.
            false
        }
    }
}

pub enum TensorBufferType {
    Unknown,
    HostMemory,
    Ahwb,
    Ion,
    DmaBuf,
    FastRpc,
    GlBuffer,
    GlTexture,
    OpenClBuffer,
    OpenClBufferFp16,
    OpenClTexture,
    OpenClTextureFp16,
    OpenClBufferPacked,
}

impl TensorBufferType {
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

pub struct TensorBuffer<'a> {
    pub(crate) raw_tensor_buffer: LiteRtTensorBuffer,
    element_type: ElementType,
    _phantom: PhantomData<&'a LiteRtTensorBuffer>,
}

struct TensorBufferLock<'a, T> {
    buffer: &'a TensorBuffer<'a>,
    raw_data: *mut T,
}

impl<T> Drop for TensorBufferLock<'_, T> {
    fn drop(&mut self) {
        // SAFETY: self.buffer.raw_tensor_buffer is always valid, it's guaranteed to be initialized by
        // a wrapper function.
        unsafe {
            LiteRtUnlockTensorBuffer(self.buffer.raw_tensor_buffer);
        }
    }
}

impl<'a> TensorBuffer<'a> {
    pub(crate) fn new(
        environment: &Environment,
        tensor_type: *const LiteRtRankedTensorType,
        buffer_type: &TensorBufferType,
        buffer_size: usize,
        element_type: ElementType,
    ) -> Result<TensorBuffer<'a>, Error> {
        let mut buffer_ptr: *mut LiteRtTensorBufferT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: environment.raw_environment is always valid, it's guaranteed to be initialized by
            // a wrapper function.
            unsafe {
                LiteRtCreateManagedTensorBuffer(
                    environment.raw_environment,
                    buffer_type.to_c_enum(),
                    tensor_type,
                    buffer_size,
                    &mut buffer_ptr,
                )
            },
            ErrorCause::CreateManagedTensorBuffer
        );
        Ok(TensorBuffer { raw_tensor_buffer: buffer_ptr, element_type, _phantom: PhantomData {} })
    }

    /// Returns the element type of the tensor buffer.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    fn lock_read<T>(&'a self) -> Result<TensorBufferLock<'a, T>, Error> {
        let mut data: *mut c_void = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_tensor_buffer is always valid, it's guaranteed to be initialized by
            // a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe {
                LiteRtLockTensorBuffer(
                    self.raw_tensor_buffer,
                    &mut data,
                    LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeRead,
                )
            },
            ErrorCause::LockTensorBufferRead
        );
        Ok(TensorBufferLock { buffer: self, raw_data: data as *mut T })
    }

    fn lock_write<T>(&'a self) -> Result<TensorBufferLock<'a, T>, Error> {
        let mut data: *mut c_void = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: self.raw_tensor_buffer is always valid, it's guaranteed to be initialized by
            // a wrapper function.
            // We assume that the output is valid if the return status is OK or don't use the output pointer.
            unsafe {
                LiteRtLockTensorBuffer(
                    self.raw_tensor_buffer,
                    &mut data,
                    LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeWrite,
                )
            },
            ErrorCause::LockTensorBufferWrite
        );
        Ok(TensorBufferLock { buffer: self, raw_data: data as *mut T })
    }

    /// Returns the size of the tensor buffer in bytes.
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

    /// Writes data to the tensor buffer.
    ///
    /// The data must be compatible with the element type of the tensor buffer.
    /// The data must be big enough to fill the tensor buffer.
    ///
    /// Returns the number of bytes written to the tensor buffer.
    pub fn write<T: 'static>(&self, data: &[T]) -> Result<usize, Error> {
        if !self.element_type.is_compatible::<T>() {
            return Err(Error::new(
                ErrorCause::IncompatibleWriteType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        let lock = self.lock_write()?;
        let dst_size = self.packed_size()?;
        let src_size = mem::size_of_val(data);
        if dst_size < src_size {
            return Err(Error::new(
                ErrorCause::TensorBufferTooSmall,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        // TODO(mgubin): Do something when input data is smaller that the tensor buffer.
        // SAFETY: lock.raw_data is always valid, it's guaranteed to be initialized by
        // lock_write function.
        // data is a pointer to the start of the data buffer, it's valid as provided by safe
        // Rust code.
        // src_size / std::mem::size_of::<T>() is the number of elements to copy, it's
        // guaranteed that it won't overwrite the output buffer of read after the end of the input data.
        unsafe {
            std::ptr::copy(data.as_ptr(), lock.raw_data, src_size / std::mem::size_of::<T>());
        }

        Ok(src_size)
    }

    /// Reads data from the tensor buffer.
    ///
    /// The data must be compatible with the element type of the tensor buffer.
    /// The data must be big enough.
    ///
    /// Returns the number of bytes read from the tensor buffer.
    pub fn read<T: 'static>(&self, data: &mut [T]) -> Result<usize, Error> {
        if !self.element_type.is_compatible::<T>() {
            return Err(Error::new(
                ErrorCause::IncompatibleReadType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        let lock = self.lock_read()?;
        let src_size = self.packed_size()?;
        let dst_size = mem::size_of_val(data);
        if dst_size < src_size {
            return Err(Error::new(
                ErrorCause::TensorBufferTooSmall,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            ));
        }
        let to_copy = std::cmp::min(src_size, dst_size) / std::mem::size_of::<T>();
        // SAFETY: lock.raw_data is always valid, it's guaranteed to be initialized by
        // lock_read function.
        // data is a pointer to the start of the data buffer, it's valid as provided by safe
        // Rust code.
        // to_copy is the number of elements to copy, it's guaranteed that it won't overwrite
        // the output buffer of read after the end of the input data.
        unsafe {
            std::ptr::copy(lock.raw_data, data.as_mut_ptr(), to_copy);
        }
        Ok(to_copy)
    }
}

impl Drop for TensorBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: self.raw_tensor_buffer is always valid, it's guaranteed to be initialized by
        // a wrapper function.
        unsafe {
            LiteRtDestroyTensorBuffer(self.raw_tensor_buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_element_type_compatibility() {
        assert!(ElementType::Bool.is_compatible::<bool>());
        assert!(!ElementType::Bool.is_compatible::<u32>());
        assert!(!ElementType::Float32.is_compatible::<u32>());
    }
}
