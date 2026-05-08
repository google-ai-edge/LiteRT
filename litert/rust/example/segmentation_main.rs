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


/// Simplified port of litert/samples/async_segmentation/segmentation_model.cc
///
/// This example shows how to use LiteRT Rust bindings to run a segmentation model.
///
/// Example usage:
/// cargo run \
///  --model-path=third_party/odml/litert/litert/samples/async_segmentation/models/selfie_multiclass_256x256.tflite \
///  --input-image-path=third_party/odml/litert/litert/samples/async_segmentation/test_images/image.jpeg \
///  --output-image-path=/tmp/out.png
use clap::Parser;
use image::{imageops, ImageBuffer, ImageReader};

const IMAGE_WIDTH: u32 = 256;
const IMAGE_HEIGHT: u32 = 256;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model_path: String,

    #[arg(short, long)]
    input_image_path: String,

    #[arg(short, long)]
    output_image_path: String,
}

struct SegmentationModel {
    model: litert::Model,
    env: litert::Environment,
    compiled_model: litert::CompiledModel,
}

impl SegmentationModel {
    fn initialize(model_path: &str) -> Result<SegmentationModel, litert::Error> {
        let m = litert::Model::create_model_from_file(model_path)?;
        let env = litert::EnvironmentBuilder::build_default()?;
        let options = litert::Options::create_with_accelerator(litert::LiteRtHwAccelerator::Cpu)?;
        let cm = litert::CompiledModel::create(&env, &m, &options)?;
        Ok(SegmentationModel { model: m, env: env, compiled_model: cm })
    }
    fn run(&self, input: &[f32], output: &mut [f32]) -> Result<(), litert::Error> {
        let inputs = self.compiled_model.create_input_tensor_buffers(&self.env, &self.model, 0)?;
        println!("Input type: {:?}", inputs[0].element_type());
        inputs[0].write(input)?;
        let outputs =
            self.compiled_model.create_output_tensor_buffers(&self.env, &self.model, 0)?;
        println!("Output type: {:?}", outputs[0].element_type());
        self.compiled_model.run(0, &inputs, &outputs)?;
        outputs[0].read(output)?;
        Ok(())
    }
}

fn load_image(path: &str) -> ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let img = ImageReader::open(path)
        .expect("Failed to read an input image")
        .decode()
        .expect("Failed to decode an input image");
    let resized_image =
        imageops::resize(&img, IMAGE_WIDTH, IMAGE_HEIGHT, imageops::FilterType::Lanczos3);
    resized_image
}

fn image_to_tensor(img: &ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> Vec<f32> {
    let mut tensor = vec![0.0f32; (3 * IMAGE_WIDTH * IMAGE_HEIGHT) as usize];
    for (index, pxl) in img.pixels().enumerate() {
        let rgba = pxl.0;
        let offset = 3 * index;
        tensor[offset + 0] = ((rgba[0] as f32) - 127.0f32) / 127.0f32;
        tensor[offset + 1] = ((rgba[1] as f32) - 127.0f32) / 127.0f32;
        tensor[offset + 2] = ((rgba[2] as f32) - 127.0f32) / 127.0f32;
    }
    tensor
}

fn save_image(tensor: &Vec<f32>, path: &str) {
    let output_colors = vec![
        image::Rgba::<u8>([0, 0, 0, 255]),
        image::Rgba::<u8>([255, 0, 0, 255]),
        image::Rgba::<u8>([0, 255, 0, 255]),
        image::Rgba::<u8>([0, 0, 255, 255]),
        image::Rgba::<u8>([64, 64, 64, 255]),
        image::Rgba::<u8>([64, 0, 0, 255]),
        image::Rgba::<u8>([0, 64, 0, 255]),
        image::Rgba::<u8>([0, 0, 64, 255]),
        image::Rgba::<u8>([128, 128, 128, 255]),
        image::Rgba::<u8>([128, 0, 0, 255]),
        image::Rgba::<u8>([0, 128, 0, 255]),
        image::Rgba::<u8>([0, 0, 128, 255]),
        image::Rgba::<u8>([192, 192, 192, 255]),
        image::Rgba::<u8>([192, 0, 0, 255]),
        image::Rgba::<u8>([0, 192, 0, 255]),
        image::Rgba::<u8>([0, 0, 192, 255]),
        image::Rgba::<u8>([255, 0, 255, 255]),
        image::Rgba::<u8>([255, 255, 0, 255]),
        image::Rgba::<u8>([0, 255, 255, 255]),
        image::Rgba::<u8>([192, 255, 255, 255]),
        image::Rgba::<u8>([255, 192, 255, 255]),
    ];

    let image = ImageBuffer::from_fn(IMAGE_WIDTH, IMAGE_HEIGHT, |x, y| {
        let num_classes: usize = tensor.len() / IMAGE_WIDTH as usize / IMAGE_HEIGHT as usize;
        assert!(num_classes < output_colors.len(), "Too many classes in the output tensor");
        let start_index = num_classes * (y * IMAGE_WIDTH + x) as usize;
        let max_index = tensor[start_index..(start_index + num_classes)]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        output_colors[max_index]
    });
    image.save_with_format(path, image::ImageFormat::Png).expect("Failed to save an output image");
}

fn main() {
    let args = Args::parse();
    let model = SegmentationModel::initialize(args.model_path.as_str())
        .expect("Failed to initialize model");
    println!("Model initialized successfully!");
    let input_image = load_image(args.input_image_path.as_str());
    println!("Input image loaded successfully!");
    let input_tensor = image_to_tensor(&input_image);
    for s in model.model.signatures().expect("Failed to get signature iterator") {
        let signature = s.expect("Failed to get signature");
        println!("Signature key: {:?}", signature.key());
        for input_name in signature.input_names().expect("Failed to get input names") {
            println!("Input name: {input_name:?}");
        }
        for output_name in signature.output_names().expect("Failed to get output names") {
            println!("Output name: {output_name:?}");
        }
        let mut output = vec![0.0f32; 2 * 3 * 256 * 256];
        model.run(&input_tensor, &mut output).expect("Failed to get input");
        save_image(&output, args.output_image_path.as_str());
    }
}
