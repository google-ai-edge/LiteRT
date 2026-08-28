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

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertArrayEquals;

import android.content.Context;
import androidx.test.core.app.ApplicationProvider;
import com.google.ai.edge.litert.support.common.FileUtil;
import com.google.common.collect.ImmutableList;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Test class to test the functionalities of BertTokenizer. */
@RunWith(RobolectricTestRunner.class)
public final class BertTokenizerTest {
  private BertTokenizer tokenizer;
  private List<String> vocab;
  private Context context;

  @Before
  public void setUp() throws Exception {

    this.vocab = new ArrayList<>(ImmutableList.of("hell", "##o", "wor", "##ld", "there"));
    this.context = ApplicationProvider.getApplicationContext();
    this.tokenizer = new BertTokenizer(vocab, new BertTokenizer.Options());
  }

  @Test
  public void testTokenize() {
    List<String> results = this.tokenizer.tokenize("hello there hello world");
    assertArrayEquals(new String[] {"hell", "##o", "there", "hell", "##o", "wor", "##ld"},
        results.toArray(new String[0]));
  }

  @Test
  public void testTokenizeWithVocabFile() throws IOException {
    // Write a temp file that contains the vocabulary
    File tempFile = File.createTempFile("test-file", ".txt", context.getFilesDir());
    try (PrintWriter writer = new PrintWriter(tempFile, UTF_8.name())) {
      for (String word : vocab) {
        writer.println(word);
      }
    }

    // Create a BertTokenizer reading the vocabulary file.
    InputStream is = context.openFileInput(tempFile.getName());
    List<String> vocab = FileUtil.loadSingleColumnTextFile(is, UTF_8);
    BertTokenizer tokenizer = new BertTokenizer(vocab, new BertTokenizer.Options());
    List<String> results = tokenizer.tokenize("hello there hello world");
    assertArrayEquals(new String[] {"hell", "##o", "there", "hell", "##o", "wor", "##ld"},
        results.toArray(new String[0]));
  }

  @Test
  public void testTokenizeZh() throws IOException {
    List<String> vocab =
        new ArrayList<>(ImmutableList.of("hell", "##o", "wor", "##ld", "there", "一", "二"));
    BertTokenizer tokenizer = new BertTokenizer(vocab, new BertTokenizer.Options());
    List<String> results = tokenizer.tokenize("一二");
    assertArrayEquals(new String[] {"一", "二"}, results.toArray(new String[0]));

    List<String> mixedResults = tokenizer.tokenize("hello 一二");
    assertArrayEquals(new String[] {"hell", "##o", "一", "二"},
        mixedResults.toArray(new String[0]));
  }

  @Test
  public void testTokenizeWithVocabFileUTF8() throws IOException {
    List<String> multilingualVocab =
        new ArrayList<>(ImmutableList.of("hell", "##o", "wor", "##ld", "there", "一", "二"));

    File tempFile = File.createTempFile("test-file", ".txt", context.getFilesDir());
    try (PrintWriter writer = new PrintWriter(tempFile, UTF_8.name())) {
      for (String word : multilingualVocab) {
        writer.println(word);
      }
    }

    // Create a BertTokenizer reading the vocabulary file.
    InputStream is = context.openFileInput(tempFile.getName());
    List<String> vocab = FileUtil.loadSingleColumnTextFile(is, UTF_8);
    BertTokenizer.Options options = new BertTokenizer.Options();
    options.useUnknownToken = false;
    BertTokenizer tokenizer = new BertTokenizer(vocab, options);
    List<String> results = tokenizer.tokenize("hello there hello world hello 一二");
    assertArrayEquals(
        new String[] {
          "hell", "##o", "there", "hell", "##o", "wor", "##ld", "hell", "##o", "一", "二"
        },
        results.toArray(new String[0]));
  }

  @Test
  public void testconvertTokensToIds() {
    List<String> tokens = this.tokenizer.tokenize("hello there hello world");
    List<Integer> results = this.tokenizer.convertTokensToIds(tokens);
    assertArrayEquals(new Integer[] {0, 1, 4, 0, 1, 2, 3},
        results.toArray(new Integer[0]));
  }
}
