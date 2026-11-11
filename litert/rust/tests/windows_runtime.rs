// Copyright 2026 Alexander Popov.
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

#![cfg(windows)]

use litert::{EnvironmentBuilder, LiteRtHwAccelerator};

#[test]
#[ignore = "requires a working WebGPU runtime and GPU"]
fn webgpu_model_runs_with_managed_buffers() {
    use litert::{CompiledModel, Model, Options, SignatureIndex};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../test/testdata/simple_model.tflite");
    let environment = EnvironmentBuilder::build_default().unwrap();
    let model = Model::create_model_from_file(&environment, path.to_str().unwrap()).unwrap();
    let options = Options::create_with_accelerator(LiteRtHwAccelerator::Gpu).unwrap();
    let compiled = CompiledModel::create(&environment, &model, &options).unwrap();
    let signature = SignatureIndex::from(0);
    let inputs = compiled.create_input_tensor_buffers(&environment, &model, signature).unwrap();
    let outputs = compiled.create_output_tensor_buffers(&environment, &model, signature).unwrap();
    inputs[0].write(&[1.0_f32, 2.0]).unwrap();
    inputs[1].write(&[10.0_f32, 20.0]).unwrap();
    compiled.run(signature, &inputs, &outputs).unwrap();
    let mut output = [0_f32; 2];
    outputs[0].read(&mut output).unwrap();
    assert_eq!(output, [11.0, 22.0]);
    assert!(compiled.is_non_cpu_fully_accelerated().unwrap());
}

#[test]
fn webgpu_buffer_types_round_trip_without_losing_runtime_requirements() {
    for value in 20..=26 {
        let buffer_type = litert::TensorBufferType::from_c_enum(value).unwrap();
        assert_eq!(buffer_type.to_c_enum(), value);
    }
}

#[test]
fn accelerator_discovery_handles_the_published_runtime_without_optional_exports() {
    let environment = EnvironmentBuilder::build_default().unwrap();
    let runtime = libloading::os::windows::Library::open_already_loaded("libLiteRt.dll").unwrap();
    let symbols_available = [
        b"LiteRtGetNumAccelerators\0".as_slice(),
        b"LiteRtGetAccelerator\0".as_slice(),
        b"LiteRtGetAcceleratorHardwareSupport\0".as_slice(),
    ]
    .iter()
    .all(|name| unsafe { runtime.get::<unsafe extern "C" fn()>(name) }.is_ok());
    if symbols_available {
        assert!(environment
            .get_available_accelerators()
            .unwrap()
            .contains(&LiteRtHwAccelerator::Cpu));
    } else {
        assert_eq!(environment.get_available_accelerators().unwrap_err().litert_status(), 5);
    }
}
