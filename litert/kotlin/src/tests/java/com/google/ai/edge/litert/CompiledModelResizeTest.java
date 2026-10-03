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

/** Integration tests for CompiledModel input resizing. */
@RunWith(JUnit4.class)
public final class CompiledModelResizeTest {

  private static final String DYNAMIC_MODEL_PATH = "litert/test/testdata/dynamic_add_model.tflite";
  private static final String STATIC_MODEL_PATH = "litert/test/testdata/simple_model.tflite";
  private static final String SIGNATURE_MODEL_PATH =
      "litert/test/testdata/reverse_signature_model.tflite";
  private static final String INPUT_0 = "arg0";
  private static final String INPUT_1 = "arg1";
  private static final String OUTPUT = "tfl.add";
  private static final String SIGNATURE_INPUT = "x";
  private static final String SIGNATURE = "serving_default";

  @Test
  public void strictResizeSupportsDynamicDimensions() throws LiteRtException {
    int[] dimensions = {2, 2, 3};
    float[] input0Values = new float[12];
    float[] input1Values = new float[12];
    Arrays.fill(input0Values, 1.0f);
    Arrays.fill(input1Values, 2.0f);

    try (CompiledModel model = CompiledModel.create(DYNAMIC_MODEL_PATH)) {
      model.resizeInputTensor(INPUT_0, dimensions);
      model.resizeInputTensor(INPUT_1, dimensions);

      assertThat(model.getInputBufferRequirements(INPUT_0, "").getBufferSize()).isEqualTo(48);
      assertThat(model.getInputBufferRequirements(INPUT_1, "").getBufferSize()).isEqualTo(48);

      try (TensorBuffer input0 = model.createInputBuffer(INPUT_0, "");
          TensorBuffer input1 = model.createInputBuffer(INPUT_1, "");
          TensorBuffer output = model.createOutputBuffer(OUTPUT, "")) {
        input0.writeFloat(input0Values);
        input1.writeFloat(input1Values);
        model.run(Arrays.asList(input0, input1), Collections.singletonList(output), 0);

        assertThat(output.readFloat()).isEqualTo(new float[] {
            3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        });
      }
    }
  }

  @Test
  public void strictResizeRejectsStaticDimensions() throws LiteRtException {
    try (CompiledModel model = CompiledModel.create(DYNAMIC_MODEL_PATH)) {
      expectLiteRtException(() -> model.resizeInputTensor(INPUT_0, new int[] {1, 3, 3}));
    }
  }

  @Test
  public void nonStrictResizeSupportsStaticDimensions() throws LiteRtException {
    int[] dimensions = {3};

    try (CompiledModel model = CompiledModel.create(STATIC_MODEL_PATH)) {
      expectLiteRtException(() -> model.resizeInputTensor(INPUT_0, dimensions));

      model.resizeInputTensorNonStrict(INPUT_0, dimensions);
      model.resizeInputTensorNonStrict(INPUT_1, dimensions);

      assertThat(model.getInputBufferRequirements(INPUT_0, "").getBufferSize()).isEqualTo(12);
      assertThat(model.getInputBufferRequirements(INPUT_1, "").getBufferSize()).isEqualTo(12);

      try (TensorBuffer input0 = model.createInputBuffer(INPUT_0, "");
          TensorBuffer input1 = model.createInputBuffer(INPUT_1, "");
          TensorBuffer output = model.createOutputBuffer(OUTPUT, "")) {
        input0.writeFloat(new float[] {1.0f, 2.0f, 3.0f});
        input1.writeFloat(new float[] {10.0f, 20.0f, 30.0f});
        model.run(Arrays.asList(input0, input1), Collections.singletonList(output), 0);

        assertThat(output.readFloat()).isEqualTo(new float[] {11.0f, 22.0f, 33.0f});
      }
    }
  }

  @Test
  public void resizeSupportsNonEmptySignature() throws LiteRtException {
    int[] originalDimensions = {1, 8};
    int[] dimensions = {3};

    try (CompiledModel model = CompiledModel.create(SIGNATURE_MODEL_PATH)) {
      model.resizeInputTensor(SIGNATURE_INPUT, originalDimensions, SIGNATURE);
      model.resizeInputTensorNonStrict(SIGNATURE_INPUT, dimensions, SIGNATURE);

      assertThat(model.getInputBufferRequirements(SIGNATURE_INPUT, SIGNATURE).getBufferSize())
          .isEqualTo(12);
    }
  }

  private static void expectLiteRtException(ThrowingRunnable runnable) {
    try {
      runnable.run();
      fail("Expected LiteRtException.");
    } catch (LiteRtException expected) {
      // Expected.
    }
  }

  private interface ThrowingRunnable {
    void run() throws LiteRtException;
  }
}
