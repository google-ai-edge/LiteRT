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

//! # LiteRT Rust Bindings
//!
//! This crate provides the public API for LiteRT Rust bindings, allowing for the execution
//! of TensorFlow Lite models.
//!
//! ## Minimal Usage Example
//!
//! The following example demonstrates how to create an environment, load a model from a file,
//! compile it, create input and output buffers, and run the model for inference.
//!
//! ```rust,no_run
//! use litert::{CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options, SignatureIndex};
//!
//! // 1. Create an environment.
//! let environment = EnvironmentBuilder::build_default().expect("Failed to create environment");
//!
//! // 2. Load the model from a file.
//! let model_path = "/path/to/your/model.tflite";
//! let model = Model::create_model_from_file(model_path)
//!     .expect("Failed to load model");
//!
//! // 3. Set compilation options, for example, to use a hardware accelerator.
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
//!     .expect("Failed to create options");
//!
//! // 4. Compile the model.
//! let compiled_model = CompiledModel::create(&environment, &model, &options)
//!     .expect("Failed to compile model");
//!
//! // 5. Create input and output tensor buffers.
//! // We are using the first signature (index 0).
//! let signature_index = SignatureIndex::from(0);
//! let input_buffers = compiled_model
//!     .create_input_tensor_buffers(signature_index)
//!     .expect("Failed to create input buffers");
//!
//! let output_buffers = compiled_model
//!     .create_output_tensor_buffers(signature_index)
//!     .expect("Failed to create output buffers");
//!
//! // 6. (Optional) Fill input buffers with data.
//! // This is just an example, actual data should match the model's requirements.
//! let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
//! input_buffers[0].write(&input_data).expect("Failed to write to input buffer");
//!
//! // 7. Run inference.
//! compiled_model.run(signature_index, &input_buffers, &output_buffers)
//!     .expect("Failed to run model");
//!
//! // 8. (Optional) Read results from output buffers.
//! let mut output_data: Vec<f32> = vec![0.0; 10]; // Pre-allocate space for results
//! output_buffers[0].read(&mut output_data).expect("Failed to read from output buffer");
//!
//! println!("Inference successful. Output data: {:?}", output_data);
//! ```
//!
//! ## Async Usage Example
//!
//! When the `async_support` feature is enabled, you can run models asynchronously
//! and await the results.
//!
//! ```rust,no_run
//! # #[cfg(async_support)]
//! # async fn run_async_example() -> Result<(), litert::Error> {
//! # use litert::{CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options};
//! # let environment = EnvironmentBuilder::build_default().unwrap();
//! # let model = Model::create_model_from_file("/path/to/model.tflite").unwrap();
//! # // Note: async is not available on all platforms.
//! # let options = Options::create_with_accelerator(LiteRtHwAccelerator::Gpu).unwrap();
//! # let compiled_model = CompiledModel::create(&environment, &model, &options).unwrap();
//! let signature_index = SignatureIndex::from(0);
//! # let input_buffers = compiled_model.create_input_tensor_buffers(signature_index).unwrap();
//! # let output_buffers = compiled_model.create_output_tensor_buffers(signature_index).unwrap();
//! // 1. Run inference asynchronously.
//! // This starts execution in a parallel thread if supported.
//! let running_parallel = compiled_model.run_async(signature_index, &input_buffers, &output_buffers)?;
//!
//! if running_parallel {
//!   // ... do other stuff ...
//! }
//! // 2. Read results asynchronously.
//! // `read_async` returns a Future that resolves when the output buffer is ready.
//! let mut output_data: Vec<f32> = vec![0.0; 10];
//! output_buffers[0].read_async(&mut output_data).await?;
//!
//! println!("Async inference successful. Output data: {:?}", output_data);
//! # Ok(())
//! # }
//! ```
//!
//! ## Using InputTensorsBuilder
//!
//! You can use [`InputTensorsBuilder`] to populate input tensors from user-provided buffers
//! by input name or index, execute inference with [`CompiledModel::run_with_inputs`], and read
//! results from the returned [`OutputTensors`] by name or index:
//!
//! ```rust,no_run
//! use litert::{
//!     CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options, SignatureIndex,
//! };
//!
//! // 1. Create an environment and load the model.
//! let environment = EnvironmentBuilder::build_default().expect("Failed to create environment");
//! let model = Model::create_model_from_file(&environment, "/path/to/your/model.tflite")
//!     .expect("Failed to load model");
//!
//! // 2. Compile the model.
//! let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)
//!     .expect("Failed to create options");
//! let compiled_model = CompiledModel::create(&environment, &model, &options)
//!     .expect("Failed to compile model");
//!
//! // 3. Create an InputTensorsBuilder for the model's default signature.
//! let input_builder = compiled_model
//!     .create_input_tensors_builder(SignatureIndex::DEFAULT)
//!     .expect("Failed to create input tensors builder");
//!
//! // 4. Populate input tensors with data from user slices by name or index.
//! let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
//! input_builder
//!     .from_user_buffer_by_name("input_tensor_name", &input_data)
//!     .expect("Failed to populate input buffer by name")
//!     .from_user_buffer_by_index(0, &input_data)
//!     .expect("Failed to populate input buffer by index");
//!
//! // 5. Run inference and obtain OutputTensors.
//! let outputs = compiled_model
//!     .run_with_inputs(&input_builder)
//!     .expect("Failed to run model");
//!
//! // 6. Read results from OutputTensors by name or index.
//! let mut output_data: Vec<f32> = vec![0.0; 10];
//! outputs
//!     .get_by_name("output_tensor_name")
//!     .expect("Failed to find output tensor")
//!     .read(&mut output_data)
//!     .expect("Failed to read output buffer");
//! ```
//!
//! ### Async Inference with InputTensorsBuilder
//!
//! When the `async_support` feature is enabled, use [`CompiledModel::run_async_with_inputs`]
//! with [`InputTensorsBuilder`]:
//!
//! ```rust,no_run
//! # #[cfg(async_support)]
//! # async fn run_async_with_builder() -> Result<(), litert::Error> {
//! # use litert::{CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options, SignatureIndex};
//! # let environment = EnvironmentBuilder::build_default()?;
//! # let model = Model::create_model_from_file(&environment, "/path/to/model.tflite")?;
//! # let options = Options::create_with_accelerator(LiteRtHwAccelerator::Gpu)?;
//! # let compiled_model = CompiledModel::create(&environment, &model, &options)?;
//! let input_builder = compiled_model.create_input_tensors_builder(SignatureIndex::DEFAULT)?;
//! let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
//! input_builder.from_user_buffer_by_name("input_tensor_name", &input_data)?;
//!
//! // 1. Run inference asynchronously using the builder.
//! let (running_parallel, outputs) = compiled_model.run_async_with_inputs(&input_builder)?;
//! if running_parallel {
//!     // ... do other work while inference runs ...
//! }
//!
//! // 2. Await and read results from the output tensor buffer.
//! let mut output_data: Vec<f32> = vec![0.0; 10];
//! outputs.get_by_name("output_tensor_name")?.read_async(&mut output_data).await?;
//! # Ok(())
//! # }
//! ```

mod bindings;
pub mod compiled_model;
pub mod environment;
pub mod error;
mod helper_funs;
pub mod model;
pub mod tensor_buffer;
#[macro_use]
mod macros;

// Make some types available to the user.
pub use bindings::LiteRtStatus;
pub use compiled_model::CompiledModel;
pub use compiled_model::InputTensorBuilder;
pub use compiled_model::InputTensorsBuilder;
pub use compiled_model::LiteRtHwAccelerator;
pub use compiled_model::Options;
pub use compiled_model::OutputTensors;
pub use compiled_model::TensorMap;
pub use environment::Environment;
pub use environment::EnvironmentBuilder;
pub use error::Error;
pub use error::ErrorCause;
pub use model::InputOutputIndex;
pub use model::Model;
pub use model::SignatureIndex;
pub use model::SubgraphIndex;
pub use tensor_buffer::ElementType;
#[cfg(async_support)]
pub use tensor_buffer::ReadAsync;
pub use tensor_buffer::TensorBuffer;
pub use tensor_buffer::TensorBufferRequirements;
pub use tensor_buffer::TensorBufferType;
