/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.google.ai.edge.litert.support.text.tokenizers;

import static com.google.common.truth.Truth.assertThat;

import android.content.Context;
import androidx.test.core.app.ApplicationProvider;
import com.google.ai.edge.litert.support.common.FileUtil;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.List;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Test class to test the functionalities of SentencePieceTokenizer. */
@RunWith(RobolectricTestRunner.class)
public final class SentencePieceTokenizerTest {
  private static final String SP_MODEL_DIR = "30k-clean.model";
  private SentencePieceTokenizer tokenizer;

  @Before
  public void setUp() throws Exception {
    Context context = ApplicationProvider.getApplicationContext();
    MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, SP_MODEL_DIR);
    tokenizer = new SentencePieceTokenizer(modelBuffer, true);
  }

  @Test
  public void tokenizeTest() throws Exception {
    String testExample = "Good morning, I'm your teacher.\n";
    assertThat(tokenizer.tokenize(testExample))
        .containsExactly("▁good", "▁morning", ",", "▁i", "'", "m", "▁your", "▁teacher", ".")
        .inOrder();
    assertThat(tokenizer.tokenize("")).isEmpty();

    String nullString = null;
    Assert.assertThrows(NullPointerException.class, () -> tokenizer.tokenize(nullString));
  }

  @Test
  public void convertTokensToIdsTest() throws Exception {
    List<String> testExample =
        Arrays.asList("▁good", "▁morning", ",", "▁i", "'", "m", "▁your", "▁teacher", ".");
    assertThat(tokenizer.convertTokensToIds(testExample))
        .containsExactly(254, 959, 15, 31, 22, 79, 154, 2197, 9)
        .inOrder();
  }
}
