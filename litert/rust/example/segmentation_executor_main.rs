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

//! Simplified port of `litert/samples/async_segmentation/segmentation_model.cc`
//!
//! This example shows how to use the high-level `LiteRT` Rust usability wrappers
//! ([`litert::LiteRTModelExecutor`] and [`litert::SignatureRunner`]) to run a segmentation model.
//!
//! Example usage:
//! ```shell
//! cargo run --bin segmentation_executor \
//!   --model-path=models/selfie_multiclass_256x256.tflite \
//!   --input-image-path=image.jpeg \
//!   --output-image-path=/tmp/out.png
//! ```
use clap::Parser;
use image::{imageops, ImageBuffer, ImageReader};
use litert::{
    EnvironmentBuilder, Error, LiteRTModelExecutor, LiteRtHwAccelerator, Options, SignatureId,
    SignatureRunner,
};

/// Expected input and output image width in pixels for the segmentation model.
const IMAGE_WIDTH: u32 = 256;
/// Expected input and output image height in pixels for the segmentation model.
const IMAGE_HEIGHT: u32 = 256;

/// Command-line arguments for the segmentation executor example.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the `.tflite` segmentation model file.
    #[arg(short, long)]
    model_path: String,

    /// Path to the input RGB/RGBA image file.
    #[arg(short, long)]
    input_image_path: String,

    /// Destination path for the output PNG segmentation mask.
    #[arg(short, long)]
    output_image_path: String,
}

/// High-level segmentation model wrapper holding the shared executor and default signature runner.
struct SegmentationModel {
    /// Thread-safe model executor owning the environment, model, and compiled model.
    executor: LiteRTModelExecutor<'static>,
    /// Stateful runner for the default signature with pre-allocated input/output buffers.
    runner: SignatureRunner<'static, 'static>,
}

impl SegmentationModel {
    /// Initializes the `LiteRT` environment, compiles the model at `model_path` for CPU execution,
    /// and creates a [`SignatureRunner`] for the default signature.
    fn initialize(model_path: &str) -> Result<Self, Error> {
        let env = EnvironmentBuilder::build_default()?;
        let accelerators = env.get_available_accelerators()?;
        println!("Available accelerators: {:?}", accelerators);
        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)?;
        let executor = LiteRTModelExecutor::from_file_with_env(env, model_path, &options)?;
        let runner = executor.signature_runner(SignatureId::default())?;
        Ok(SegmentationModel { executor, runner })
    }

    /// Runs synchronous segmentation inference on `input` and copies the resulting class logits
    /// into `output`.
    fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        let input_buf = self.runner.input_buffer(0)?;
        println!("Input type: {:?}", input_buf.element_type());
        println!("Input dimensions: {:?}", input_buf.dimensions());
        let output_buf = self.runner.output_buffer(0)?;
        println!("Output type: {:?}", output_buf.element_type());
        println!("Output dimensions: {:?}", output_buf.dimensions());
        self.runner.set_input(0, input)?.run()?.get_output(0, output)?;
        Ok(())
    }
}

/// Loads an image from `path` and resizes it to `IMAGE_WIDTH x IMAGE_HEIGHT` RGBA pixels.
fn load_image(path: &str) -> ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let img = ImageReader::open(path)
        .expect("Failed to read an input image")
        .decode()
        .expect("Failed to decode an input image");
    imageops::resize(&img, IMAGE_WIDTH, IMAGE_HEIGHT, imageops::FilterType::Lanczos3)
}

/// Converts an RGBA image buffer into a normalized `[-1.0, 1.0]` RGB float tensor.
fn image_to_tensor(img: &ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> Vec<f32> {
    let mut tensor = vec![0.0f32; (3 * IMAGE_WIDTH * IMAGE_HEIGHT) as usize];
    for (index, pxl) in img.pixels().enumerate() {
        let rgba = pxl.0;
        let offset = 3 * index;
        tensor[offset] = ((rgba[0] as f32) - 127.0f32) / 127.0f32;
        tensor[offset + 1] = ((rgba[1] as f32) - 127.0f32) / 127.0f32;
        tensor[offset + 2] = ((rgba[2] as f32) - 127.0f32) / 127.0f32;
    }
    tensor
}

/// Maps per-pixel class scores in `tensor` to an RGBA color palette by argmax and saves the
/// resulting segmentation mask as a PNG at `path`.
fn save_image(tensor: &[f32], path: &str) {
    const OUTPUT_COLORS: [image::Rgba<u8>; 21] = [
        image::Rgba([0, 0, 0, 255]),
        image::Rgba([255, 0, 0, 255]),
        image::Rgba([0, 255, 0, 255]),
        image::Rgba([0, 0, 255, 255]),
        image::Rgba([64, 64, 64, 255]),
        image::Rgba([64, 0, 0, 255]),
        image::Rgba([0, 64, 0, 255]),
        image::Rgba([0, 0, 64, 255]),
        image::Rgba([128, 128, 128, 255]),
        image::Rgba([128, 0, 0, 255]),
        image::Rgba([0, 128, 0, 255]),
        image::Rgba([0, 0, 128, 255]),
        image::Rgba([192, 192, 192, 255]),
        image::Rgba([192, 0, 0, 255]),
        image::Rgba([0, 192, 0, 255]),
        image::Rgba([0, 0, 192, 255]),
        image::Rgba([255, 0, 255, 255]),
        image::Rgba([255, 255, 0, 255]),
        image::Rgba([0, 255, 255, 255]),
        image::Rgba([192, 255, 255, 255]),
        image::Rgba([255, 192, 255, 255]),
    ];

    let num_pixels = (IMAGE_WIDTH * IMAGE_HEIGHT) as usize;
    let num_classes = tensor.len() / num_pixels;
    assert_eq!(
        num_classes * num_pixels,
        tensor.len(),
        "Output tensor length is not a multiple of IMAGE_WIDTH * IMAGE_HEIGHT"
    );
    assert!(
        (1..=OUTPUT_COLORS.len()).contains(&num_classes),
        "Expected 1..={} classes in output tensor, got {num_classes}",
        OUTPUT_COLORS.len()
    );

    let image = ImageBuffer::from_fn(IMAGE_WIDTH, IMAGE_HEIGHT, |x, y| {
        let start_index = num_classes * (y * IMAGE_WIDTH + x) as usize;
        let max_index = tensor[start_index..(start_index + num_classes)]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("Empty class slice")
            .0;

        OUTPUT_COLORS[max_index]
    });
    image.save_with_format(path, image::ImageFormat::Png).expect("Failed to save an output image");
}

/// Parses CLI flags, initializes the segmentation model, runs inference over the input image,
/// and writes the colorized segmentation mask to disk.
fn main() {
    let args = Args::parse();
    let mut model = SegmentationModel::initialize(args.model_path.as_str())
        .expect("Failed to initialize model");
    println!("Model initialized successfully!");
    let input_image = load_image(args.input_image_path.as_str());
    println!("Input image loaded successfully!");
    let input_tensor = image_to_tensor(&input_image);
    let executor = model.executor.clone();
    for s in executor.model().signatures().expect("Failed to get signature iterator") {
        let signature = s.expect("Failed to get signature");
        println!("Signature key: {:?}", signature.key());
        for input_name in signature.input_names().expect("Failed to get input names") {
            println!("Input name: {input_name:?}");
        }
        for output_name in signature.output_names().expect("Failed to get output names") {
            println!("Output name: {output_name:?}");
        }
    }
    let mut output = vec![0.0f32; (6 * IMAGE_WIDTH * IMAGE_HEIGHT) as usize];
    model.run(&input_tensor, &mut output).expect("Failed to run model");
    save_image(&output, args.output_image_path.as_str());
}
