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

//! # `LiteRT` Rust Bindings
//!
//! This crate provides the public API for `LiteRT` Rust bindings, allowing for the execution
//! of TensorFlow Lite (`.tflite`) models at two abstraction levels:
//!
//! 1. **High-Level Usability Wrappers** ([`LiteRTModelExecutor`], [`SignatureRunner`],
//!    [`AsyncExecution`], [`TensorBufferReadLock`], [`TensorBufferWriteLock`]) for ergonomic,
//!    type-safe, and reusable-buffer inference.
//! 2. **Low-Level C API Wrappers** ([`Environment`], [`Model`], [`CompiledModel`],
//!    [`TensorBuffer`]) for direct control over individual `LiteRT` runtime handles, close to the
//!    original C API.
//!
//! ## Advantages of the High-Level Usability Wrappers
//!
//! Compared to using the low-level handles directly, the high-level usability layer solves several
//! ergonomics, performance, and memory-safety pitfalls:
//!
//! - **Single-Step Access by Name or Index ([`SignatureId`], [`TensorId`])**: In the low-level API,
//!   accessing a tensor by name requires querying the [`Model`]'s [`model::Signature`] to translate
//!   a tensor name into an [`InputOutputIndex`] before indexing into the [`TensorBuffer`] vector.
//!   [`SignatureRunner`] caches the signature's input and output tensor names when created and
//!   accepts either a string name (`"input_0"`) or a 0-based index (`0`,
//!   `InputOutputIndex::new(0)`) on every method with zero heap allocations and zero FFI calls
//!   during lookup.
//! - **Object-Graph & Drop-Order Safety ([`LiteRTModelExecutor`])**: The low-level API requires
//!   passing separate `&Environment` and `&Model` / `&mut Model` references into [`CompiledModel`]
//!   methods, which cannot statically prevent passing an [`Environment`] or [`Model`] from a
//!   different instance or dropping the [`Environment`] while a [`CompiledModel`] or
//!   [`TensorBuffer`] is still alive. [`LiteRTModelExecutor`] co-owns the [`CompiledModel`],
//!   [`Model`], and [`Environment`] inside an `Arc` and enforces the C API's strict destruction
//!   order (`CompiledModel` -> `Model` -> `Environment`).
//! - **Pre-Allocated Buffers & In-Place Loop Reuse ([`SignatureRunner`])**: Creating new input and
//!   output [`TensorBuffer`]s on every inference call is expensive. [`SignatureRunner`]
//!   pre-allocates managed [`TensorBuffer`]s and caches tensor names once when created, allowing
//!   tight inference loops to reuse buffers in-place, bind custom/hardware [`TensorBuffer`]s
//!   ([`SignatureRunner::bind_input_buffer`] / [`SignatureRunner::bind_output_buffer`]), or write
//!   and read host memory zero-copy via RAII lock guards ([`SignatureRunner::set_input_with`],
//!   [`SignatureRunner::lock_input`], [`SignatureRunner::lock_output`]).
//! - **Compile-Time Element Type Safety ([`TensorElement`])**: Replaces unconstrained `T: 'static`
//!   generics with a sealed [`TensorElement`] trait implemented only for valid FFI numeric scalar
//!   types (`i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`) where every byte
//!   bit-pattern is valid, catching invalid buffer element types at compile time in addition to
//!   runtime [`ElementType`] checks. (`bool` is intentionally excluded because Rust requires
//!   `&[bool]` bytes to be strictly `0x00` or `0x01`, which cannot be guaranteed for untrusted
//!   foreign C/C++ delegate outputs or non-host buffers; [`ElementType::Bool`] buffers are read and
//!   written as `u8` or `i8` and converted via `byte != 0` / `u8::from(flag)`.)
//! - **Multithreading & Async Borrow Safety (`Send` / `Sync` & [`AsyncExecution`])**:
//!   - [`LiteRTModelExecutor`] is `Send + Sync` and cheaply cloneable (`O(1)` `Arc::clone`), so a
//!     single compiled model can be shared across worker threads to spawn independent per-thread
//!     [`SignatureRunner`]s (`Send + !Sync`) with their own input/output [`TensorBuffer`]s (while
//!     the underlying [`CompiledModel`] serializes C API `run()` / `run_async()` calls via an
//!     internal mutex).
//!   - [`SignatureRunner::run_async`] returns an [`AsyncExecution`] guard that mutably borrows
//!     `&mut SignatureRunner` while accelerator output events are pending—preventing the caller
//!     from mutating input buffers, unbinding buffers, or starting a concurrent run through the
//!     runner until output events have been awaited, [`AsyncExecution::finish`] is called, or the
//!     guard is dropped (note that externally bound `&TensorBuffer`s remain accessible via their
//!     shared references, and C++ `LiteRtTensorBufferT::Lock` automatically waits on any attached
//!     `LiteRtEvent` before mapping host memory).
//!
//! ---
//!
//! ## High-Level Wrapper Examples
//!
//! ### 1. Simple One-Shot Execution (Method Chaining)
//!
//! Because [`SignatureRunner`] holds a reference-counted clone of the [`LiteRTModelExecutor`],
//! you can chain model loading, signature runner creation, input population, execution, and output
//! reading in a single expression without temporary lifetime errors (`E0716`):
//!
//! ```rust,no_run
//! use litert::{
//!     InputOutputIndex, LiteRTModelExecutor, LiteRtHwAccelerator, Options, SignatureIndex,
//! };
//!
//! # fn run_one_shot() -> Result<(), litert::Error> {
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)?;
//! let input_array0 = [1.0f32, 2.0, 3.0];
//! let input_array1 = [4.0f32, 5.0, 6.0];
//! let mut out_buff = [0.0f32; 3];
//!
//! // Creates a thread-safe model executor owning Environment, Model, and CompiledModel (via Arc).
//! LiteRTModelExecutor::from_file("/path/to/your/model.tflite", &options)?
//!     // Creates a signature runner that pre-allocates and owns input/output TensorBuffers.
//!     .signature_runner(SignatureIndex::DEFAULT)?
//!     // Set inputs by name or by index using the same method:
//!     .set_input("abc", &input_array0)?
//!     .set_input(InputOutputIndex::from(1), &input_array1)?
//!     // Or initialize a pre-allocated TensorBuffer in-place via a locked &mut [f32] slice:
//!     .set_input_with("a1", |slice: &mut [f32]| {
//!         slice.fill(1.0);
//!     })?
//!     // Execute inference synchronously:
//!     .run()?
//!     // Copy output(s) into caller buffer(s) (chainable across multiple outputs):
//!     .get_output("b2", &mut out_buff)?;
//!
//! println!("Inference output: {:?}", out_buff);
//! # Ok(())
//! # }
//! ```
//!
//! ### 2. Reusing [`SignatureRunner`] Across a Loop
//!
//! Pre-allocated input/output [`TensorBuffer`]s are retained inside [`SignatureRunner`], so
//! invariant inputs can be set once before a loop and only per-step inputs need to be updated:
//!
//! ```rust,no_run
//! use litert::{LiteRTModelExecutor, LiteRtHwAccelerator, Options, SignatureIndex};
//!
//! # fn run_loop(values: &[Vec<f32>]) -> Result<(), litert::Error> {
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)?;
//! let mut signature_runner = LiteRTModelExecutor::from_file("/path/to/model.tflite", &options)?
//!     .signature_runner(SignatureIndex::DEFAULT)?;
//!
//! // Set invariant input once before the loop:
//! let invariant_input = [1.0f32, 2.0, 3.0];
//! signature_runner.set_input("abc", &invariant_input)?;
//!
//! let mut output = vec![0.0f32; 3];
//! for v in values {
//!     signature_runner
//!         .set_input("xyz", v.as_slice())?
//!         .run()?
//!         .get_output("o1", &mut output)?;
//!     // Or read zero-copy without copying into `output`:
//!     let output_slice = signature_runner.lock_output::<f32>("o1")?;
//!     println!("First element: {}", output_slice[0]);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### 3. High-Level Asynchronous Execution (`run_async`)
//!
//! When the `async_support` feature is enabled, [`SignatureRunner::run_async`] returns an
//! [`AsyncExecution`] guard that holds `&mut SignatureRunner` until the in-flight execution
//! finishes:
//!
//! ```rust,no_run
//! # #[cfg(async_support)]
//! # async fn run_async_high_level() -> Result<(), litert::Error> {
//! use litert::{LiteRTModelExecutor, LiteRtHwAccelerator, Options, SignatureId};
//!
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Gpu)?;
//! let mut signature_runner = LiteRTModelExecutor::from_file("/path/to/model.tflite", &options)?
//!     .signature_runner(SignatureId::default())?;
//!
//! let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
//! signature_runner.set_input("abc", &input_data)?;
//!
//! // `async_exec` mutably borrows `signature_runner` while hardware accelerator execution is in
//! // flight, preventing concurrent mutation of input buffers until output events are awaited.
//! let mut async_exec = signature_runner.run_async()?;
//! if async_exec.is_parallel() {
//!     // Perform parallel CPU work while the hardware accelerator finishes filling outputs...
//! }
//!
//! // Waits for the output buffer's sync fence event (if any) and copies data into `output_data`:
//! let mut output_data: Vec<f32> = vec![0.0; 10];
//! async_exec.get_output("b2", &mut output_data).await?;
//!
//! println!("Async inference successful. Output data: {:?}", output_data);
//! # Ok(())
//! # }
//! ```
//!
//! ---
//!
//! ## Low-Level API Examples
//!
//! If you need direct access to individual [`Environment`], [`Model`], [`CompiledModel`], and
//! [`TensorBuffer`] handles, you can use the low-level API directly.
//!
//! ### 1. Low-Level Synchronous Execution
//!
//! ```rust,no_run
//! use litert::{
//!     CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options, SignatureIndex,
//! };
//!
//! // 1. Create an environment.
//! let environment = EnvironmentBuilder::build_default().expect("Failed to create environment");
//!
//! // 2. Load the model from a file.
//! let model_path = "/path/to/your/model.tflite";
//! let mut model = Model::create_model_from_file(&environment, model_path)
//!     .expect("Failed to load model");
//!
//! // 3. Set compilation options, for example, to use a hardware accelerator.
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
//!     .expect("Failed to create options");
//!
//! // 4. Compile the model.
//! let compiled_model = CompiledModel::create(&environment, &mut model, &options)
//!     .expect("Failed to compile model");
//!
//! // 5. Create input and output tensor buffers for the default signature (index 0).
//! let signature_index = SignatureIndex::DEFAULT;
//! let input_buffers = compiled_model
//!     .create_input_tensor_buffers(&environment, &model, signature_index)
//!     .expect("Failed to create input buffers");
//!
//! let output_buffers = compiled_model
//!     .create_output_tensor_buffers(&environment, &model, signature_index)
//!     .expect("Failed to create output buffers");
//!
//! // 6. Fill input buffers with data.
//! let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
//! input_buffers[0].write(&input_data).expect("Failed to write to input buffer");
//!
//! // 7. Run inference.
//! compiled_model.run(signature_index, &input_buffers, &output_buffers)
//!     .expect("Failed to run model");
//!
//! // 8. Read results from output buffers.
//! let mut output_data: Vec<f32> = vec![0.0; 10];
//! output_buffers[0].read(&mut output_data).expect("Failed to read from output buffer");
//!
//! println!("Inference successful. Output data: {:?}", output_data);
//! ```
//!
//! ### 2. Low-Level Asynchronous Execution
//!
//! ```rust,no_run
//! # #[cfg(async_support)]
//! # async fn run_async_low_level() -> Result<(), litert::Error> {
//! use litert::{
//!     CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options, SignatureIndex,
//! };
//!
//! let environment = EnvironmentBuilder::build_default()?;
//! let mut model = Model::create_model_from_file(&environment, "/path/to/model.tflite")?;
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Gpu)?;
//! let compiled_model = CompiledModel::create(&environment, &mut model, &options)?;
//!
//! let signature_index = SignatureIndex::DEFAULT;
//! let input_buffers =
//!     compiled_model.create_input_tensor_buffers(&environment, &model, signature_index)?;
//! let output_buffers =
//!     compiled_model.create_output_tensor_buffers(&environment, &model, signature_index)?;
//!
//! // 1. Run inference asynchronously.
//! // Returns `true` if the accelerator delegate attached synchronization events to the output
//! // buffers for asynchronous hardware execution.
//! let has_pending_events =
//!     compiled_model.run_async(signature_index, &input_buffers, &output_buffers)?;
//! if has_pending_events {
//!     // ... do other work while hardware accelerator finishes filling output buffers ...
//! }
//!
//! // 2. Read results asynchronously once the output buffer event signals.
//! let mut output_data: Vec<f32> = vec![0.0; 10];
//! output_buffers[0].read_async(&mut output_data).await?;
//!
//! println!("Async inference successful. Output data: {:?}", output_data);
//! # Ok(())
//! # }
//! ```

mod bindings;
pub mod compiled_model;
pub mod environment;
pub mod error;
mod helper_funs;
pub mod model;
pub mod model_executor;
pub mod signature_runner;
pub mod tensor_buffer;
pub mod tensor_buffer_lock;
#[macro_use]
mod macros;

// Make some types available to the user.
pub use bindings::LiteRtStatus;
pub use compiled_model::CompiledModel;
pub use compiled_model::LiteRtHwAccelerator;
pub use compiled_model::Options;
pub use environment::Environment;
pub use environment::EnvironmentBuilder;
pub use error::Error;
pub use error::ErrorCause;
pub use model::InputOutputIndex;
pub use model::Model;
pub use model::SignatureIndex;
pub use model::SubgraphIndex;
pub use model_executor::LiteRTModelExecutor;
pub use model_executor::SignatureId;
#[cfg(async_support)]
pub use signature_runner::AsyncExecution;
pub use signature_runner::SignatureRunner;
pub use signature_runner::TensorId;
pub use tensor_buffer::ElementType;
#[cfg(async_support)]
pub use tensor_buffer::ReadAsync;
pub use tensor_buffer::TensorBuffer;
pub use tensor_buffer::TensorBufferRequirements;
pub use tensor_buffer::TensorBufferType;
pub use tensor_buffer_lock::TensorBufferReadLock;
pub use tensor_buffer_lock::TensorBufferWriteLock;
pub use tensor_buffer_lock::TensorElement;
