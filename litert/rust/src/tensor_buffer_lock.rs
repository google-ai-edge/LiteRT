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

//! Compile-time element type safety (`TensorElement`) and zero-copy RAII lock guards
//! (`TensorBufferReadLock`, `TensorBufferWriteLock`) for `TensorBuffer`.

use crate::bindings::LiteRtUnlockTensorBuffer;
use crate::tensor_buffer::{ElementType, TensorBuffer};
use std::ops::{Deref, DerefMut};
use std::slice;

mod private {
    /// Sealed supertrait preventing external implementations of [`super::TensorElement`].
    pub trait Sealed {}
}

/// Marker and validation trait for scalar types that can be read from or written to a
/// [`TensorBuffer`].
///
/// # Safety
/// Implementors must be non-zero-sized plain-old-data (`Copy + 'static`) with no uninitialized
/// padding, a valid 1-to-1 memory representation with at least one `LiteRtElementType`, and valid
/// when zero-initialized (`0x00` bytes).
pub unsafe trait TensorElement: Copy + 'static + private::Sealed {
    /// Returns `true` if `Self` is layout-compatible with the runtime [`ElementType`].
    ///
    /// # Arguments
    ///
    /// * `element_type` - The runtime [`ElementType`] of the [`TensorBuffer`].
    ///
    /// # Returns
    ///
    /// `true` if `Self` has a valid, size- and alignment-compatible representation for
    /// `element_type`, `false` otherwise.
    #[must_use]
    fn is_compatible(element_type: ElementType) -> bool;
}

/// Implements `private::Sealed` and [`TensorElement`] for a primitive scalar type `$ty`.
macro_rules! impl_tensor_element {
    ($ty:ty, $($elem:ident)|+) => {
        impl private::Sealed for $ty {}
        // SAFETY: `$ty` is a non-zero-sized primitive scalar with no padding, matches `$elem`, and
        // is valid when zero-initialized (`0x00` bytes; for `bool`, managed buffers are
        // zero-initialized on creation and LiteRT boolean tensors store `0x00` or `0x01`).
        unsafe impl TensorElement for $ty {
            fn is_compatible(element_type: ElementType) -> bool {
                matches!(element_type, $(ElementType::$elem)|+)
            }
        }
    };
}

impl_tensor_element!(bool, Bool);
impl_tensor_element!(i8, Int8 | UInt8);
impl_tensor_element!(u8, Int8 | UInt8);
impl_tensor_element!(i16, Int16 | UInt16);
impl_tensor_element!(u16, Int16 | UInt16);
impl_tensor_element!(i32, Int32 | UInt32);
impl_tensor_element!(u32, Int32 | UInt32);
impl_tensor_element!(i64, Int64 | UInt64);
impl_tensor_element!(u64, Int64 | UInt64);
impl_tensor_element!(f32, Float32);
impl_tensor_element!(f64, Float64);

/// RAII guard providing zero-copy immutable slice access (`&[T]`) to a locked [`TensorBuffer`].
#[must_use = "if unused, the TensorBuffer will immediately be unlocked"]
pub struct TensorBufferReadLock<'buf, T> {
    /// Reference to the locked [`TensorBuffer`], unlocked on drop.
    pub(crate) buffer: &'buf TensorBuffer<'buf>,
    /// Non-null, aligned pointer to the mapped host memory (or dangling when `len == 0`).
    pub(crate) ptr: *const T,
    /// Number of initialized elements of type `T` in the mapped buffer.
    pub(crate) len: usize,
}

impl<T> Deref for TensorBufferReadLock<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY:
        // - `self.ptr` is guaranteed non-null and aligned to `align_of::<T>()` by
        //   `TensorBuffer::lock_read` (using `NonNull::dangling()` when `self.len == 0`).
        // - The underlying buffer is locked in read mode for `self.len` initialized elements of `T`
        //   and remains locked and valid for the lifetime of `self` (`'buf`).
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> Drop for TensorBufferReadLock<'_, T> {
    fn drop(&mut self) {
        let _guard = self.buffer.lock_state_mutex();
        // SAFETY:
        // - `self.buffer` outlives `self` via `'buf`, and `TensorBufferReadLock` is only
        //   constructed after a successful `LiteRtLockTensorBuffer` call.
        // - `_guard` serializes lock/unlock state transitions across threads, and `drop` unlocks
        //   the buffer exactly once.
        unsafe {
            LiteRtUnlockTensorBuffer(self.buffer.raw_tensor_buffer);
        }
    }
}

/// RAII guard providing zero-copy mutable slice access (`&mut [T]`) to a locked [`TensorBuffer`].
#[must_use = "if unused, the TensorBuffer will immediately be unlocked"]
pub struct TensorBufferWriteLock<'buf, T> {
    /// Reference to the locked [`TensorBuffer`], unlocked on drop.
    pub(crate) buffer: &'buf TensorBuffer<'buf>,
    /// Non-null, aligned pointer to the mapped host memory (or dangling when `len == 0`).
    pub(crate) ptr: *mut T,
    /// Number of initialized elements of type `T` in the mapped buffer.
    pub(crate) len: usize,
}

impl<T> Deref for TensorBufferWriteLock<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY:
        // - `self.ptr` is guaranteed non-null and aligned to `align_of::<T>()` by
        //   `TensorBuffer::lock_write` (using `NonNull::dangling()` when `self.len == 0`).
        // - The underlying buffer is locked in write mode for `self.len` initialized elements of
        //   `T` and cannot be locked again while `self` is alive.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for TensorBufferWriteLock<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY:
        // - `self.ptr` is guaranteed non-null and aligned to `align_of::<T>()` by
        //   `TensorBuffer::lock_write` (using `NonNull::dangling()` when `self.len == 0`).
        // - `&mut self` guarantees exclusive access to the write-locked buffer mapping of
        //   `self.len` initialized elements of `T` for the duration of the borrow.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for TensorBufferWriteLock<'_, T> {
    fn drop(&mut self) {
        let _guard = self.buffer.lock_state_mutex();
        // SAFETY:
        // - `self.buffer` outlives `self` via `'buf`, and `TensorBufferWriteLock` is only
        //   constructed after a successful `LiteRtLockTensorBuffer` call.
        // - `_guard` serializes lock/unlock state transitions across threads, and `drop` unlocks
        //   the buffer exactly once.
        unsafe {
            LiteRtUnlockTensorBuffer(self.buffer.raw_tensor_buffer);
        }
    }
}
