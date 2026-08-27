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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * An object to tokenize inputs for BERT.
 *
 * <p>{@code String} inputs can be tokenized into inputs for BERT using an instance of {@code
 * BertTokenizer}. Create an instance of {@code BertTokenizer} using a vocabulary, which can be
 * loaded from static functions in {@code FileUtil}. Example usage:
 *
 * <pre>{@code
 * List<String> vocab = FileUtil.loadVocabularyFile(vocabFilePath);
 * BertTokenizer tokenizer = new BertTokenizer(vocab, new BertTokenizer.Options());
 * tokenizer.tokenize(new String[] { "hello there", "tokenize me" });
 *
 * }</pre>
 */
public class BertTokenizer implements Tokenizer {
  private static final String JNI_LIB = "bert_tokenizer_jni";

  /** Options to customize the outputs of BertTokenizer. */
  public static class Options {
    int maxBytesPerToken = 100;
    int maxCharsPerSubToken = 100;
    String suffixIndicator = "##";
    boolean useUnknownToken = true;
    String unknownToken = "[UNK]";
    boolean splitUnknownChars = false;
  }

  private final Options options;
  private long vocabHandle;
  private static volatile boolean isLibraryLoaded;

  /**
   * Creates an instance of {@code BertTokenizer}.
   *
   * @param vocab A {@code List} of {@code String} objects where each element is a word in the
   *     vocabulary. The position of the word in the list is used as the id of the vocabulary word.
   * @param options A list set of parameters to customize the tokenizer.
   */
  public BertTokenizer(List<String> vocab, Options options) {
    loadJNILib();
    this.options = options;
    nativeLoadResource(vocab);
  }

  // TODO(b/148419618): Define the following in followup CLs:
  //  - tokenizeWithOffsets that returns character offsets.
  //  - a return type that supports batches of ragged shapes and different dtypes.
  @Override
  public List<String> tokenize(String text) {
    List<String> tokens = Arrays.asList(nativeTokenize(vocabHandle, text));
    return tokens;
  }

  @Override
  public List<Integer> convertTokensToIds(List<String> tokens) {
    List<Integer> outputIds = new ArrayList<>();
    String[] tokensArray = new String[tokens.size()];
    tokensArray = tokens.toArray(tokensArray);
    for (int idInt : nativeConvertTokensToIds(vocabHandle, tokensArray)) {
      Integer id = idInt;
      outputIds.add(id);
    }
    return outputIds;
  }

  @Override
  public void close() {
    if (vocabHandle != 0) {
      vocabHandle = nativeUnloadResource(vocabHandle);
    }
  }

  private void nativeLoadResource(List<String> vocab) {
    if (vocabHandle == 0) {
      vocabHandle =
          nativeLoadResource(
              vocab,
              options.maxBytesPerToken,
              options.maxCharsPerSubToken,
              options.suffixIndicator,
              options.useUnknownToken,
              options.unknownToken,
              options.splitUnknownChars);
    }
  }

  private static void loadJNILib() {
    if (!isLibraryLoaded) {
      System.loadLibrary(JNI_LIB);
      isLibraryLoaded = true;
    }
  }

  private native long nativeLoadResource(
      List<String> vocab,
      int maxBytesPerToken,
      int maxCharsPerSubToken,
      String suffixIndicator,
      boolean useUnknownToken,
      String unknownToken,
      boolean splitUnknownChars);

  private native long nativeUnloadResource(long handle);

  private native String[] nativeTokenize(long processor, String text);

  private native int[] nativeConvertTokensToIds(long processor, String[] tokens);
}
