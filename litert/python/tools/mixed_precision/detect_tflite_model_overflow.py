# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to detect TFLite model overflow."""

from absl import app
from absl import flags
import tensorflow as tf

from litert.python.tools.mixed_precision import detect_tflite_model_overflow_lib


_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to the TFLite model file.',
    required=True)
_SIGNATURE_KEY = flags.DEFINE_string(
    'signature_key', None, 'Signature key to use (optional).'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with tf.io.gfile.GFile(_MODEL_PATH.value, 'rb') as f:
    encoder_model_tflite = f.read()

  interpreter = tf.lite.Interpreter(
      model_content=encoder_model_tflite,
      experimental_preserve_all_tensors=True)

  if _SIGNATURE_KEY.value:
    runner = interpreter.get_signature_runner(_SIGNATURE_KEY.value)
    inputs = {}
    for key, detail in runner.get_input_details().items():
      inputs[key] = detect_tflite_model_overflow_lib.generate_tensor_data(
          detail['shape'], detail['dtype'])

    runner(**inputs)
  else:
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    for detail in input_details:
      data = detect_tflite_model_overflow_lib.generate_tensor_data(
          detail['shape'], detail['dtype'])
      interpreter.set_tensor(detail['index'], data)

    interpreter.invoke()

  overflow_found = False
  tensor_details = interpreter.get_tensor_details()
  for detail in tensor_details:
    try:
      tensor = interpreter.get_tensor(detail['index'])
    except ValueError:
      continue
    if detect_tflite_model_overflow_lib.check_tensor_overflow(
        tensor, detail['name']):
      overflow_found = True

  if not overflow_found:
    print('No overflow detected in the model.')

if __name__ == '__main__':
  app.run(main)
