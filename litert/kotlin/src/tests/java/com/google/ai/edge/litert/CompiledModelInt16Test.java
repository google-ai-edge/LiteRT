/*
 * Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litert;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.Collections;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for INT16 tensors in the CompiledModel API. */
@RunWith(JUnit4.class)
public final class CompiledModelInt16Test {

  private static final String MODEL_PATH =
      "litert/test/testdata/single_add_default_a16w8_recipe_quantized.tflite";
  private static final String INPUT_1 = "input_1";
  private static final String INPUT_2 = "input_2";
  private static final String OUTPUT = "add";
  private static final int ELEMENT_COUNT = 1 * 32 * 32;

  @Test
  public void int16TensorBuffersSupportShortArrays() throws LiteRtException {
    short[] signedValues = new short[ELEMENT_COUNT];
    signedValues[0] = Short.MIN_VALUE;
    signedValues[1] = -123;
    signedValues[2] = 0;
    signedValues[3] = Short.MAX_VALUE;
    for (int i = 4; i < ELEMENT_COUNT; i++) {
      signedValues[i] = (short) (i - 512);
    }
    short[] zeroValues = new short[ELEMENT_COUNT];

    try (CompiledModel model = CompiledModel.create(MODEL_PATH);
        TensorBuffer input1 = model.createInputBuffer(INPUT_1, "");
        TensorBuffer input2 = model.createInputBuffer(INPUT_2, "");
        TensorBuffer output = model.createOutputBuffer(OUTPUT, "")) {
      assertThat(model.getInputTensorType(INPUT_1, "").getElementType())
          .isEqualTo(TensorType.ElementType.INT16);
      assertThat(model.getInputTensorType(INPUT_2, "").getElementType())
          .isEqualTo(TensorType.ElementType.INT16);
      assertThat(model.getOutputTensorType(OUTPUT, "").getElementType())
          .isEqualTo(TensorType.ElementType.INT16);

      try {
        input1.writeInt16(new short[ELEMENT_COUNT + 1]);
        fail("Expected an oversize INT16 write to fail.");
      } catch (LiteRtException expected) {
        // Expected: the input buffer only holds ELEMENT_COUNT INT16 elements.
      }

      input1.writeInt16(signedValues);
      assertThat(input1.readInt16()).isEqualTo(signedValues);

      input2.writeInt16(zeroValues);
      model.run(Arrays.asList(input1, input2), Collections.singletonList(output), 0);

      short[] outputValues = output.readInt16();
      assertThat(outputValues).hasLength(ELEMENT_COUNT);
      assertThat((int) outputValues[0]).isLessThan(0);
      assertThat((int) outputValues[1]).isLessThan(0);
      assertThat((int) outputValues[2]).isEqualTo(0);
      assertThat((int) outputValues[3]).isGreaterThan(0);
    }
  }
}
