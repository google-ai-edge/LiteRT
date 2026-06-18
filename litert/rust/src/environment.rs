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

use crate::bindings::*;
use crate::call_check_status;
use crate::error::{Error, ErrorCause};
use std::any::{Any, TypeId};
use std::ffi::CString;

/// Options for environment Tags.
pub enum OptionTag {
    CompilerPluginLibraryDir,
    DispatchLibraryDir,
    ClDeviceId,
    ClPlatformId,
    ClContext,
    ClCommandQueue,
    EglContext,
    EglDisplay,
    WebGpuDevice,
    WebGpuQueue,
    MetalDevice,
    MetalCommandQueue,
    // TODO(mgubin): Add VulkanEnvironment.
}

impl OptionTag {
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
        }
    }
}

/// Builder for Environment.
///
/// This builder is used to create an environment with specific options.
///
/// Example usage:
///
/// ```
/// let builder = EnvironmentBuilder::new();
/// builder.add_option(OptionTag::ClDeviceId, 123);
/// builder.add_option(OptionTag::ClPlatformId, 456);
/// builder.add_option(OptionTag::ClContext, context);
/// builder.add_option(OptionTag::ClCommandQueue, command_queue);
/// let env = builder.build();
/// ```
/// If options are not needed, you can use Environment::create_default() to create an environment
/// with default options.
///
/// ```
/// let env = Environment::create_default();
/// ```
#[derive(Default)]
pub struct EnvironmentBuilder {
    options: Vec<LiteRtEnvOption>,

    // If a LiteRtEnvOption has a string value, we need to store the CString in a separate vector
    // and keep it alive until the environment is destroyed.
    cstring_storage: Vec<CString>,
}

pub struct Environment {
    pub(crate) raw_environment: LiteRtEnvironment,
    // See the comment in EnvironmentBuilder, internally Environment owns environment options,
    // some of them may be strings, so we need to store the CStrings somewhere.
    #[allow(dead_code)]
    cstring_storage: Vec<CString>,
}

impl EnvironmentBuilder {
    pub fn build(self) -> Result<Environment, Error> {
        Environment::new(self)
    }

    pub fn build_default() -> Result<Environment, Error> {
        let default_builder = EnvironmentBuilder::default();
        Environment::new(default_builder)
    }

    pub fn new() -> Self {
        Self { options: Vec::new(), cstring_storage: Vec::new() }
    }

    fn build_lite_rt_any<T: 'static>(&mut self, value: T) -> Result<LiteRtAny, Error> {
        let type_id = TypeId::of::<T>();
        let any_value: &dyn Any = &value;
        if TypeId::of::<bool>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeBool,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // SAFETY: unwrap is safe here, because we checked that type_id is of<bool>.
                    bool_value: *any_value.downcast_ref::<bool>().unwrap(),
                },
            })
        } else if TypeId::of::<i32>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeInt,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // SAFETY: unwrap is safe here, because we checked that type_id is of<i32>.
                    int_value: *any_value.downcast_ref::<i32>().unwrap() as i64,
                },
            })
        } else if TypeId::of::<i64>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeInt,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // SAFETY: unwrap is safe here, because we checked that type_id is of<i64>.
                    int_value: *any_value.downcast_ref::<i64>().unwrap(),
                },
            })
        } else if TypeId::of::<f32>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeReal,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // SAFETY: unwrap is safe here, because we checked that type_id is of<f32>.
                    real_value: *any_value.downcast_ref::<f32>().unwrap() as f64,
                },
            })
        } else if TypeId::of::<f64>() == type_id {
            Ok(LiteRtAny {
                type_: LiteRtAnyType_kLiteRtAnyTypeReal,
                __bindgen_anon_1: LiteRtAny__bindgen_ty_1 {
                    // SAFETY: unwrap is safe here, because we checked that type_id is of<f64>.
                    real_value: *any_value.downcast_ref::<f64>().unwrap(),
                },
            })
        } else if TypeId::of::<&str>() == type_id {
            // SAFETY: unwrap is safe here, because we checked that type_id is of<&str>.
            let str_value = *any_value.downcast_ref::<&str>().unwrap();
            self.cstring_storage.push(CString::new(str_value).expect("Failed to create CString"));
            let cstr_ptr = self.cstring_storage.last().unwrap().as_ptr();
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

    pub fn add_option<T: 'static>(mut self, tag: OptionTag, value: T) -> Result<Self, Error> {
        let val = self.build_lite_rt_any(value)?;
        self.options.push(LiteRtEnvOption { tag: tag.to_c_enum(), value: val });
        Ok(self)
    }
}

impl Environment {
    fn new(builder: EnvironmentBuilder) -> Result<Self, Error> {
        let options_ptr = builder.options.as_ptr();
        let mut raw_environment_ptr: *mut LiteRtEnvironmentT = std::ptr::null_mut();
        call_check_status!(
            // SAFETY: Calling unsafe functions LiteRtCreateEnvironment().
            // options_ptr is a pointer into a vector is valid.
            unsafe {
                LiteRtCreateEnvironment(
                    builder.options.len() as i32,
                    options_ptr,
                    &mut raw_environment_ptr,
                )
            },
            ErrorCause::CreateEnvironment
        );
        Ok(Self { raw_environment: raw_environment_ptr, cstring_storage: builder.cstring_storage })
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // SAFETY: self.raw_environment is always valid, it's guaranteed to be initialized by
        // create* function.
        unsafe {
            LiteRtDestroyEnvironment(self.raw_environment);
        }
    }
}

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
            .build();
        assert_eq!(environment.is_ok(), true);
    }
}
