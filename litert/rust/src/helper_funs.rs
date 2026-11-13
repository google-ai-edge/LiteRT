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

//! Internal FFI string conversion helpers.

pub use crate::bindings::*;
use crate::error::{Error, ErrorCause};

use std::ffi::{c_char, CStr};

/// Converts a NUL-terminated C string pointer into a borrowed UTF-8 `&'a str`.
///
/// # Arguments
///
/// * `c_str` - Pointer to a NUL-terminated C string (or null).
///
/// # Returns
///
/// The validated UTF-8 string slice `&'a str`.
///
/// # Errors
///
/// Returns [`Error`] (`ErrorCause::InvalidStringEncoding`) if `c_str` is null or does not contain
/// valid UTF-8.
///
/// # Safety
///
/// If `c_str` is non-null, it must point to a valid NUL-terminated C string that remains valid and
/// immutable for the lifetime `'a`.
pub unsafe fn c_str_to_str<'a>(c_str: *const c_char) -> Result<&'a str, Error> {
    if c_str.is_null() {
        return Err(Error::new(
            ErrorCause::InvalidStringEncoding,
            LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
        ));
    }
    // SAFETY:
    // - `c_str` is checked to be non-null above.
    // - By the `# Safety` contract of `c_str_to_str`, `c_str` points to a valid NUL-terminated C
    //   string that remains valid and immutable for lifetime `'a`.
    unsafe {
        CStr::from_ptr(c_str).to_str().map_err(|_| {
            Error::new(
                ErrorCause::InvalidStringEncoding,
                LiteRtStatus_kLiteRtStatusErrorRuntimeFailure,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nullptr_to_str() {
        // SAFETY: This is a test, so it is safe to pass a nullptr.
        unsafe {
            let result = c_str_to_str(std::ptr::null());
            assert_eq!(
                result,
                Err(Error::new(
                    ErrorCause::InvalidStringEncoding,
                    LiteRtStatus_kLiteRtStatusErrorRuntimeFailure
                ))
            );
        }
    }
}
