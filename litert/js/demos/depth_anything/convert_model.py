# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Converts and quantizes a Depth Anything V2 Small model to TFLite.

This script uses ai-edge-torch to convert a Hugging Face Depth Anything V2 Small
model to a float TFLite model, and then uses ai-edge-quantizer to quantize
the float model to a weight-only quantized TFLite model.
"""

from pathlib import Path
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe
import ai_edge_torch
import PIL
import requests
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForDepthEstimation

# Load the pretrained model and processor from Hugging Face
model_id = "depth-anything/Depth-Anything-V2-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)

# Dummy input for tracing
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = PIL.Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")


# Wrapper class for the model
class ModelWrapper(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.model = AutoModelForDepthEstimation.from_pretrained(model_id)

  def forward(self, pixel_values):
    results = self.model(pixel_values=pixel_values)
    return (results["predicted_depth"],)

model_wrapped = ModelWrapper().eval()
args_litert = (inputs["pixel_values"],)

# Convert the model to TFLite using ai-edge-torch
print("Converting model to TFLite...")
model_litert = ai_edge_torch.convert(model_wrapped, args_litert)

# Export the float TFLite model
script_dir = Path(__file__).parent.resolve()
float_model_path = str(script_dir / "static" / "depth_anything_v2_small.tflite")
model_litert.export(float_model_path)
print(f"Float model saved to {float_model_path}")

# Quantize the model using ai-edge-quantizer
print("Quantizing model...")
qt = quantizer.Quantizer(float_model_path)
qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())

quantized_model_path = str(
    script_dir / "static" / "depth_anything_v2_small_wi8_afp32.tflite"
)
qt.quantize().export_model(quantized_model_path)
print(f"Quantized model saved to {quantized_model_path}")

print("Model conversion and quantization complete.")
