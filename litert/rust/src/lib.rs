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
//! use litert::{CompiledModel, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options};
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
//! let signature_index = 0;
//! let input_buffers = compiled_model
//!     .create_input_tensor_buffers(&environment, &model, signature_index)
//!     .expect("Failed to create input buffers");
//!
//! let output_buffers = compiled_model
//!     .create_output_tensor_buffers(&environment, &model, signature_index)
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
pub use compiled_model::LiteRtHwAccelerator;
pub use compiled_model::Options;
pub use environment::Environment;
pub use environment::EnvironmentBuilder;
pub use error::Error;
pub use error::ErrorCause;
pub use model::Model;
pub use tensor_buffer::ElementType;
pub use tensor_buffer::TensorBuffer;
pub use tensor_buffer::TensorBufferRequirements;
pub use tensor_buffer::TensorBufferType;
