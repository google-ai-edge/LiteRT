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

//! `LiteRT` runtime [`Environment`] and [`EnvironmentBuilder`] configuration.

use crate::bindings::*;
use crate::call_check_status;
use crate::compiled_model::LiteRtHwAccelerator;
use crate::error::{Error, ErrorCause};
use std::any::{Any, TypeId};
use std::ffi::CString;
use std::sync::{Mutex, MutexGuard};

/// Configuration option tags for a `LiteRT` [`Environment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionTag {
    /// Directory path containing compiler plugin shared libraries.
    CompilerPluginLibraryDir,
    /// Directory path containing dispatch shared libraries.
    DispatchLibraryDir,
    /// `OpenCL` device ID.
    ClDeviceId,
    /// `OpenCL` platform ID.
    ClPlatformId,
    /// `OpenCL` context handle.
    ClContext,
    /// `OpenCL` command queue handle.
    ClCommandQueue,
    /// EGL context handle.
    EglContext,
    /// EGL display handle.
    EglDisplay,
    /// `WebGPU` device handle.
    WebGpuDevice,
    /// `WebGPU` queue handle.
    WebGpuQueue,
    /// Metal device handle.
    MetalDevice,
    /// Metal command queue handle.
    MetalCommandQueue,
    /// System GPU accelerator handle.
    SystemGpuAcceleratorHandle,
    // TODO(mgubin): Add VulkanEnvironment.
}

impl OptionTag {
    /// Converts the option tag into its corresponding `LiteRtEnvOptionTag` C enum constant.
    #[allow(clippy::wrong_self_convention)]
    #[must_use]
    fn to_c_enum(&self) -> LiteRtEnvOptionTag {
        match self {
            Self::CompilerPluginLibraryDir => {
                LiteRtEnvOptionTag_kLiteRtEnvOptionTagCompilerPluginLibraryDir
            }
            Self::DispatchLibraryDir => LiteRtEnvOptionTag_kLiteRtEnvOptionTagDispatchLibraryDir,
            Self::ClDeviceId => LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClDeviceId,
            Self::ClPlatformId => LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClPlatformId,
            Self::ClContext => LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClContext,
            Self::ClCommandQueue => LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClCommandQueue,
            Self::EglContext => LiteRtEnvOptionTag_kLiteRtEnvOptionTagEglContext,
            Self::EglDisplay => LiteRtEnvOptionTag_kLiteRtEnvOptionTagEglDisplay,
            Self::WebGpuDevice => LiteRtEnvOptionTag_kLiteRtEnvOptionTagWebGpuDevice,
            Self::WebGpuQueue => LiteRtEnvOptionTag_kLiteRtEnvOptionTagWebGpuQueue,
            Self::MetalDevice => LiteRtEnvOptionTag_kLiteRtEnvOptionTagMetalDevice,
            Self::MetalCommandQueue => LiteRtEnvOptionTag_kLiteRtEnvOptionTagMetalCommandQueue,
            Self::SystemGpuAcceleratorHandle => {
                LiteRtEnvOptionTag_kLiteRtEnvOptionTagSystemGpuAcceleratorHandle
            }
        }
    }
}

/// Builder for [`Environment`].
///
/// This builder is used to create a `LiteRT` runtime [`Environment`] with specific options.
///
/// Example usage:
///
/// ```no_run
/// use litert::environment::{EnvironmentBuilder, OptionTag};
///
/// # fn main() -> Result<(), litert::Error> {
/// let env = EnvironmentBuilder::new()
///     .add_option(OptionTag::ClDeviceId, 123)?
///     .add_option(OptionTag::ClPlatformId, 456)?
///     .build()?;
/// # Ok(())
/// # }
/// ```
///
/// If custom options are not needed, use [`EnvironmentBuilder::build_default()`]:
///
/// ```no_run
/// use litert::EnvironmentBuilder;
///
/// let env = EnvironmentBuilder::build_default().expect("Failed to create environment");
/// ```
#[derive(Default)]
pub struct EnvironmentBuilder {
    /// C `LiteRtEnvOption` entries accumulated by [`Self::add_option`].
    options: Vec<LiteRtEnvOption>,

    /// Owned `CString` storage for string-valued `LiteRtEnvOption` entries, transferred to
    /// [`Environment`] on [`Self::build`] so string pointers remain valid until the environment is
    /// destroyed.
    cstring_storage: Vec<CString>,
}

/// The `LiteRT` runtime environment managing hardware accelerators and shared runtime state.
pub struct Environment {
    /// Underlying C `LiteRtEnvironment` handle owned by this struct.
    pub(crate) raw_environment: LiteRtEnvironment,
    /// Serializes C API operations on `self.raw_environment` across threads (`LiteRtEnvironmentT`
    /// in C++ lazily initializes `GpuEnvironmentSingleton` without internal synchronization).
    mutex: Mutex<()>,
    /// Owned `CString` backing storage for string-valued environment options, dropped after
    /// `LiteRtDestroyEnvironment` runs in `Drop::drop`.
    #[allow(dead_code)]
    cstring_storage: Vec<CString>,
}

impl EnvironmentBuilder {
    /// Builds a new [`Environment`] with the configured options.
    ///
    /// # Returns
    ///
    /// The newly created [`Environment`].
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::CreateEnvironment`) if the underlying `LiteRT` C API fails
    /// to create the environment.
    pub fn build(self) -> Result<Environment, Error> {
        Environment::new(self)
    }

    /// Builds a new [`Environment`] with default options.
    ///
    /// # Returns
    ///
    /// The newly created default [`Environment`].
    ///
    /// # Errors
    ///
    /// Returns [`Error`] (`ErrorCause::CreateEnvironment`) if the underlying `LiteRT` C API fails
    /// to create the environment.
    pub fn build_default() -> Result<Environment, Error> {
        let default_builder = EnvironmentBuilder::default();
        Environment::new(default_builder)
    }

    /// Creates a new, empty [`EnvironmentBuilder`].
    ///
    /// # Returns
    ///
    /// A new `EnvironmentBuilder` with no options set.
    #[must_use]
    pub fn new() -> Self {
        Self { options: Vec::new(), cstring_storage: Vec::new() }
    }

    /// Converts a supported Rust scalar, `&str`, or `String` value `T` into a C `LiteRtAny` union.
    fn build_lite_rt_any<T: 'static>(&mut self, value: T) -> Result<LiteRtAny, Error> {
        let type_id = TypeId::of::<T>();
        let any_value: &dyn Any = &value;
        if TypeId::of::<bool>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeBool,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // Checked via `TypeId::of::<bool>()` above.
                    bool_value: *any_value.downcast_ref::<bool>().unwrap(),
                },
            })
        } else if TypeId::of::<i32>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeInt,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // Checked via `TypeId::of::<i32>()` above.
                    int_value: *any_value.downcast_ref::<i32>().unwrap() as i64,
                },
            })
        } else if TypeId::of::<i64>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeInt,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // Checked via `TypeId::of::<i64>()` above.
                    int_value: *any_value.downcast_ref::<i64>().unwrap(),
                },
            })
        } else if TypeId::of::<f32>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeReal,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // Checked via `TypeId::of::<f32>()` above.
                    real_value: *any_value.downcast_ref::<f32>().unwrap() as f64,
                },
            })
        } else if TypeId::of::<f64>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeReal,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // Checked via `TypeId::of::<f64>()` above.
                    real_value: *any_value.downcast_ref::<f64>().unwrap(),
                },
            })
        } else if TypeId::of::<&str>() == type_id {
            // Checked via `TypeId::of::<&str>()` above.
            let str_value = *any_value.downcast_ref::<&str>().unwrap();
            let cstr = CString::new(str_value).map_err(|_| {
                Error::new(
                    ErrorCause::InvalidStringEncoding,
                    LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
                )
            })?;
            let cstr_ptr = cstr.as_ptr();
            self.cstring_storage.push(cstr);
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeString,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 { str_value: cstr_ptr },
            })
        } else if TypeId::of::<String>() == type_id {
            // Checked via `TypeId::of::<String>()` above.
            let str_value = any_value.downcast_ref::<String>().unwrap().as_str();
            let cstr = CString::new(str_value).map_err(|_| {
                Error::new(
                    ErrorCause::InvalidStringEncoding,
                    LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
                )
            })?;
            let cstr_ptr = cstr.as_ptr();
            self.cstring_storage.push(cstr);
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeString,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 { str_value: cstr_ptr },
            })
        } else {
            Err(Error::new(
                ErrorCause::NotSupportedLiteRtAnyType,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ))
        }
    }

    /// Appends a configuration option (`tag`, `value`) to the builder.
    ///
    /// Supported value types `T` are `bool`, `i32`, `i64`, `f32`, `f64`, `&str`, and `String`.
    ///
    /// # Arguments
    ///
    /// * `tag` - The [`OptionTag`] identifying the environment option.
    /// * `value` - The option value.
    ///
    /// # Returns
    ///
    /// The updated builder with the option added.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `T` is not a supported `LiteRtAny` type
    /// (`ErrorCause::NotSupportedLiteRtAnyType`) or if a `&str` or `String` value contains an
    /// interior NUL byte (`ErrorCause::InvalidStringEncoding`).
    pub fn add_option<T: 'static>(mut self, tag: OptionTag, value: T) -> Result<Self, Error> {
        let val = self.build_lite_rt_any(value)?;
        self.options.push(LiteRtEnvOption { tag: tag.to_c_enum(), value: val });
        Ok(self)
    }
}

impl Environment {
    /// Acquires the internal mutex serializing C API operations on `self.raw_environment`.
    pub(crate) fn lock_env(&self) -> MutexGuard<'_, ()> {
        self.mutex.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Creates an [`Environment`] from the options accumulated in `builder`.
    fn new(builder: EnvironmentBuilder) -> Result<Self, Error> {
        let num_options = i32::try_from(builder.options.len()).map_err(|_| {
            Error::new(
                ErrorCause::CreateEnvironment,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )
        })?;
        let options_ptr = builder.options.as_ptr();
        let mut raw_environment_ptr: *mut LiteRtEnvironmentT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY:
            // - `options_ptr` is a valid, aligned pointer to `builder.options.len()` initialized
            //   `LiteRtEnvOption` elements. Any string pointers inside `builder.options` point
            //   into heap-allocated `CString`s in `builder.cstring_storage`, which is moved into
            //   `Self` so those addresses remain valid for the lifetime of `Environment`.
            // - `&mut raw_environment_ptr` is a valid, aligned, non-null stack out-pointer.
            unsafe { LiteRtCreateEnvironment(num_options, options_ptr, &mut raw_environment_ptr) },
            ErrorCause::CreateEnvironment
        );
        Ok(Self {
            raw_environment: raw_environment_ptr,
            mutex: Mutex::new(()),
            cstring_storage: builder.cstring_storage,
        })
    }

    /// Returns the list of hardware accelerators available in this environment.
    ///
    /// # Returns
    ///
    /// A vector of [`LiteRtHwAccelerator`]s registered in `self`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if querying the environment's accelerators fails.
    pub fn get_available_accelerators(&self) -> Result<Vec<LiteRtHwAccelerator>, Error> {
        let _guard = self.lock_env();
        let mut num_accelerators: LiteRtParamIndex = 0;
        call_check_status!(
            // SAFETY:
            // - `self.raw_environment` is a valid, non-null pointer to an initialized
            //   `LiteRtEnvironment` managed by `Environment` and only freed when `self` is
            //   dropped.
            // - `&mut num_accelerators` is a valid, properly aligned, non-null pointer to local
            //   stack memory with exclusive write permission to receive the accelerator count.
            unsafe { LiteRtGetNumAccelerators(self.raw_environment, &mut num_accelerators) },
            ErrorCause::GetNumAccelerators
        );

        let mut accelerators = Vec::with_capacity(num_accelerators);
        for i in 0..num_accelerators {
            let mut accelerator: LiteRtAccelerator = std::ptr::null_mut();
            call_check_status!(
                // SAFETY:
                // - `self.raw_environment` is valid and live for the lifetime of `self`.
                // - `i` is within `[0, num_accelerators)`, where `num_accelerators` was obtained
                //   from `LiteRtGetNumAccelerators` above and the environment's accelerator
                //   registry is immutable during `&self`.
                // - `&mut accelerator` is a valid, properly aligned, non-null pointer to local
                //   stack memory with exclusive write permission to receive the unowned
                //   accelerator handle borrowing from `self.raw_environment` (which does not
                //   require manual destruction).
                unsafe { LiteRtGetAccelerator(self.raw_environment, i, &mut accelerator) },
                ErrorCause::GetAccelerator
            );

            let mut hardware_support: LiteRtHwAcceleratorSet = 0;
            call_check_status!(
                // SAFETY:
                // - `accelerator` is a valid handle returned by `LiteRtGetAccelerator` (guaranteed
                //   successful by the preceding `call_check_status!`). It is an unowned reference
                //   whose lifetime is bound to `self.raw_environment`.
                // - `&mut hardware_support` is a valid, properly aligned, non-null pointer to
                //   local stack memory with exclusive write permission to receive the bitfield.
                unsafe { LiteRtGetAcceleratorHardwareSupport(accelerator, &mut hardware_support) },
                ErrorCause::GetAcceleratorHardwareSupport
            );

            if (hardware_support
                & (LiteRtHwAccelerators_kLiteRtHwAcceleratorCpu as LiteRtHwAcceleratorSet))
                != 0
            {
                accelerators.push(LiteRtHwAccelerator::Cpu);
            }
            if (hardware_support
                & (LiteRtHwAccelerators_kLiteRtHwAcceleratorGpu as LiteRtHwAcceleratorSet))
                != 0
            {
                accelerators.push(LiteRtHwAccelerator::Gpu);
            }
            if (hardware_support
                & (LiteRtHwAccelerators_kLiteRtHwAcceleratorNpu as LiteRtHwAcceleratorSet))
                != 0
            {
                accelerators.push(LiteRtHwAccelerator::Npu);
            }
        }
        Ok(accelerators)
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // SAFETY:
        // - `self.raw_environment` is a valid, non-null handle initialized by
        //   `LiteRtCreateEnvironment`.
        // - `LiteRtDestroyEnvironment` runs in `Drop::drop` before field `self.cstring_storage` is
        //   dropped, ensuring any option string pointers remain valid until after the C environment
        //   is destroyed.
        unsafe {
            LiteRtDestroyEnvironment(self.raw_environment);
        }
    }
}

// SAFETY: `LiteRtEnvironment` is an opaque handle to a runtime environment and `cstring_storage`
// (`Vec<CString>`) is owned and never mutated after construction, so `Environment` can be safely
// moved across threads and destroyed on another thread.
unsafe impl Send for Environment {}

// SAFETY: C API operations on `self.raw_environment` across threads are serialized by
// `self.mutex`, and `cstring_storage` is never mutated after construction, so sharing
// `&Environment` across threads is safe.
unsafe impl Sync for Environment {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_options_builder() {
        let environment = EnvironmentBuilder::new()
            .add_option(OptionTag::ClDeviceId, true)
            .expect("Valid add_option")
            .add_option(OptionTag::ClDeviceId, 123)
            .expect("Valid add_option")
            .add_option(OptionTag::ClDeviceId, 1.23)
            .expect("Valid add_option")
            .add_option(OptionTag::CompilerPluginLibraryDir, "/tmp/plugin")
            .expect("Valid add_option")
            .add_option(OptionTag::SystemGpuAcceleratorHandle, true)
            .expect("Valid add_option")
            .build();
        assert!(environment.is_ok());
        let nul_err =
            EnvironmentBuilder::new().add_option(OptionTag::CompilerPluginLibraryDir, "bad\0path");
        assert!(nul_err.is_err());
        assert_eq!(nul_err.err().unwrap().cause(), ErrorCause::InvalidStringEncoding);
    }

    #[test]
    fn test_get_available_accelerators() {
        let environment =
            EnvironmentBuilder::build_default().expect("Failed to create environment");
        let accelerators =
            environment.get_available_accelerators().expect("Failed to get available accelerators");
        // CPU accelerator is always available by default.
        assert!(!accelerators.is_empty());
        assert!(accelerators.contains(&LiteRtHwAccelerator::Cpu));
    }
}
