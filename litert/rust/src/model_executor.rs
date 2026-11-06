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

//! High-level thread-safe model executor (`LiteRTModelExecutor`) and signature selector
//! (`SignatureId`).

use crate::bindings::{LiteRtParamIndex, LiteRtStatus_kLiteRtStatusErrorInvalidArgument};
use crate::compiled_model::{CompiledModel, Options};
use crate::environment::{Environment, EnvironmentBuilder};
use crate::error::{Error, ErrorCause};
use crate::model::{Model, SignatureIndex};
use crate::signature_runner::{BufferSlot, NamedBufferSlot, SignatureRunner};
use std::borrow::{Borrow, Cow};
use std::path::Path;
use std::sync::Arc;

/// Identifies a model signature either by its 0-based index or by its signature key string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureId<'a> {
    /// Selects a signature by its 0-based [`SignatureIndex`].
    Index(SignatureIndex),
    /// Selects a signature by its string key (for example, `"serving_default"`).
    Key(&'a str),
}

impl Default for SignatureId<'static> {
    /// Returns `SignatureId::Index(SignatureIndex::DEFAULT)` (signature index `0`).
    fn default() -> Self {
        Self::Index(SignatureIndex::DEFAULT)
    }
}

impl From<SignatureIndex> for SignatureId<'_> {
    /// Converts a [`SignatureIndex`] into [`SignatureId::Index`].
    fn from(index: SignatureIndex) -> Self {
        Self::Index(index)
    }
}

impl From<LiteRtParamIndex> for SignatureId<'_> {
    /// Converts a 0-based `LiteRtParamIndex` (`usize`) into [`SignatureId::Index`].
    fn from(index: LiteRtParamIndex) -> Self {
        Self::Index(SignatureIndex::from(index))
    }
}

impl<'a> From<&'a str> for SignatureId<'a> {
    /// Converts a signature key string slice into [`SignatureId::Key`].
    fn from(key: &'a str) -> Self {
        Self::Key(key)
    }
}

impl<'a> From<&'a String> for SignatureId<'a> {
    /// Converts a borrowed `String` signature key into [`SignatureId::Key`].
    fn from(key: &'a String) -> Self {
        Self::Key(key.as_str())
    }
}

/// Internal owner of `CompiledModel`, `Model`, and `Environment` with strict drop-order guarantees.
///
/// Field declaration order is a critical safety invariant: Rust drops struct fields in declaration
/// order, so `compiled_model` is destroyed first, followed by `model`, and finally `environment`.
pub(crate) struct ExecutorInner<'model> {
    /// Compiled model handle; declared first so `LiteRtDestroyCompiledModel` runs before `model`
    /// and `environment` are dropped.
    pub(crate) compiled_model: CompiledModel,
    /// Backing model handle (and any owned `Cow<[u8]>` model buffer); dropped second via
    /// `LiteRtDestroyModel`.
    pub(crate) model: Model<'model>,
    /// Backing runtime environment; dropped last via `LiteRtDestroyEnvironment`.
    pub(crate) environment: Environment,
}

/// Thread-safe owner of a compiled `LiteRT` model and its backing [`Model`] and [`Environment`].
///
/// Cloning a `LiteRTModelExecutor` is an `O(1)` atomic reference count increment (`Arc::clone`).
#[derive(Clone)]
pub struct LiteRTModelExecutor<'model> {
    /// Shared reference-counted state co-owning the compiled model, model, and environment.
    pub(crate) inner: Arc<ExecutorInner<'model>>,
}

impl LiteRTModelExecutor<'static> {
    /// Loads a model from `path` using a default [`Environment`] and compiles it with `options`.
    ///
    /// # Arguments
    ///
    /// * `path` - File system path to the `.tflite` model file.
    /// * `options` - Compilation [`Options`] (or `&Options`) configuring hardware acceleration.
    ///
    /// # Returns
    ///
    /// A new [`LiteRTModelExecutor`], or an [`Error`] if environment creation, model loading, or
    /// compilation fails.
    pub fn from_file(path: impl AsRef<Path>, options: impl Borrow<Options>) -> Result<Self, Error> {
        let environment = EnvironmentBuilder::build_default()?;
        Self::from_file_with_env(environment, path, options)
    }

    /// Loads a model from `path` using the provided [`Environment`] and compiles it with `options`.
    ///
    /// # Arguments
    ///
    /// * `environment` - The `LiteRT` [`Environment`] that owns accelerator registrations.
    /// * `path` - File system path to the `.tflite` model file.
    /// * `options` - Compilation [`Options`] (or `&Options`) configuring hardware acceleration.
    ///
    /// # Returns
    ///
    /// A new [`LiteRTModelExecutor`], or an [`Error`] if the path is not valid UTF-8, model
    /// loading fails, or compilation fails.
    pub fn from_file_with_env(
        environment: Environment,
        path: impl AsRef<Path>,
        options: impl Borrow<Options>,
    ) -> Result<Self, Error> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            Error::new(
                ErrorCause::InvalidStringEncoding,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            )
        })?;
        let model = Model::create_model_from_file(&environment, path_str)?;
        let compiled_model = CompiledModel::create(&environment, &model, options.borrow())?;
        Self::from_parts(environment, model, compiled_model)
    }
}

impl<'model> LiteRTModelExecutor<'model> {
    /// Creates an executor from an in-memory `FlatBuffer` slice or owned `Vec<u8>` using a default
    /// [`Environment`].
    ///
    /// # Arguments
    ///
    /// * `buffer` - Borrowed `&'model [u8]` slice or owned `Vec<u8>` containing the `FlatBuffer`
    ///   model bytes.
    /// * `options` - Compilation [`Options`] (or `&Options`) configuring hardware acceleration.
    ///
    /// # Returns
    ///
    /// A new [`LiteRTModelExecutor`], or an [`Error`] if environment creation, model parsing, or
    /// compilation fails.
    pub fn from_buffer(
        buffer: impl Into<Cow<'model, [u8]>>,
        options: impl Borrow<Options>,
    ) -> Result<Self, Error> {
        let environment = EnvironmentBuilder::build_default()?;
        Self::from_buffer_with_env(environment, buffer, options)
    }

    /// Creates an executor from an in-memory `FlatBuffer` slice or owned `Vec<u8>` using the
    /// provided [`Environment`].
    ///
    /// # Arguments
    ///
    /// * `environment` - The `LiteRT` [`Environment`] that owns accelerator registrations.
    /// * `buffer` - Borrowed `&'model [u8]` slice or owned `Vec<u8>` containing the `FlatBuffer`
    ///   model bytes.
    /// * `options` - Compilation [`Options`] (or `&Options`) configuring hardware acceleration.
    ///
    /// # Returns
    ///
    /// A new [`LiteRTModelExecutor`], or an [`Error`] if model parsing or compilation fails.
    pub fn from_buffer_with_env(
        environment: Environment,
        buffer: impl Into<Cow<'model, [u8]>>,
        options: impl Borrow<Options>,
    ) -> Result<Self, Error> {
        let model = Model::create_model_from_buffer(&environment, buffer)?;
        let compiled_model = CompiledModel::create(&environment, &model, options.borrow())?;
        Self::from_parts(environment, model, compiled_model)
    }

    /// Assembles an executor from pre-created [`Environment`], [`Model`], and [`CompiledModel`]
    /// instances.
    ///
    /// # Arguments
    ///
    /// * `environment` - The [`Environment`] used to create `model` and `compiled_model`.
    /// * `model` - The [`Model`] from which `compiled_model` was compiled.
    /// * `compiled_model` - The [`CompiledModel`] compiled from `model` and `environment`.
    ///
    /// # Returns
    ///
    /// A new [`LiteRTModelExecutor`] co-owning all three handles with guaranteed drop order, or an
    /// [`Error`] if querying `compiled_model`'s environment fails or `compiled_model` was not
    /// created with `environment`.
    pub fn from_parts(
        environment: Environment,
        model: Model<'model>,
        compiled_model: CompiledModel,
    ) -> Result<Self, Error> {
        if !compiled_model.has_same_environment(&environment)? {
            return Err(Error::new(
                ErrorCause::GetCompiledModelEnvironment,
                LiteRtStatus_kLiteRtStatusErrorInvalidArgument,
            ));
        }
        Ok(Self { inner: Arc::new(ExecutorInner { compiled_model, model, environment }) })
    }

    /// Creates a stateful [`SignatureRunner`] for the specified signature (by index or key),
    /// pre-allocating managed input and output [`crate::tensor_buffer::TensorBuffer`]s.
    ///
    /// # Arguments
    ///
    /// * `signature` - Any selector convertible into [`SignatureId`] (e.g. `0`,
    ///   [`SignatureIndex::DEFAULT`], `"serving_default"`, or [`SignatureId::default()`]).
    ///
    /// # Returns
    ///
    /// A [`SignatureRunner`] with pre-allocated input/output buffers, or an [`Error`] if the
    /// signature is not found or buffer creation fails.
    pub fn signature_runner<'s, 'ext>(
        &self,
        signature: impl Into<SignatureId<'s>>,
    ) -> Result<SignatureRunner<'model, 'ext>, Error> {
        let signature_index = match signature.into() {
            SignatureId::Index(idx) => idx,
            SignatureId::Key(key) => self.inner.model.signature_index_by_key(key)?,
        };

        let sig = self.inner.model.signature(signature_index)?;
        let num_inputs = sig.num_inputs()?;
        let num_outputs = sig.num_outputs()?;

        let mut inputs = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let name = sig.input_name_at(i)?.to_owned();
            let tensor = sig.input_tensor_by_index(i)?;
            let buffer = self.inner.compiled_model.create_input_buffer_for_tensor(
                &self.inner.environment,
                signature_index,
                i,
                &tensor,
            )?;
            inputs.push(NamedBufferSlot { name, slot: BufferSlot::Managed(buffer) });
        }

        let mut outputs = Vec::with_capacity(num_outputs);
        let mut runtime_output_layouts = None;
        for i in 0..num_outputs {
            let name = sig.output_name_at(i)?.to_owned();
            let tensor = sig.output_tensor_by_index(i)?;
            let buffer = self.inner.compiled_model.create_output_buffer_for_tensor(
                &self.inner.environment,
                signature_index,
                i,
                num_outputs,
                &tensor,
                &mut runtime_output_layouts,
            )?;
            outputs.push(NamedBufferSlot { name, slot: BufferSlot::Managed(buffer) });
        }

        Ok(SignatureRunner {
            signature_index,
            inputs,
            outputs,
            executor: Arc::clone(&self.inner),
            _not_sync: std::marker::PhantomData,
        })
    }

    /// Borrows the underlying [`CompiledModel`].
    ///
    /// # Returns
    ///
    /// A shared reference to the [`CompiledModel`].
    #[must_use]
    pub fn compiled_model(&self) -> &CompiledModel {
        &self.inner.compiled_model
    }

    /// Borrows the underlying [`Model`].
    ///
    /// # Returns
    ///
    /// A shared reference to the [`Model`].
    #[must_use]
    pub fn model(&self) -> &Model<'model> {
        &self.inner.model
    }

    /// Borrows the underlying [`Environment`].
    ///
    /// # Returns
    ///
    /// A shared reference to the [`Environment`].
    #[must_use]
    pub fn environment(&self) -> &Environment {
        &self.inner.environment
    }
}
