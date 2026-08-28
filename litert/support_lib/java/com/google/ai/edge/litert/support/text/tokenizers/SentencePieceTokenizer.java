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

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * A java implementation of Sentence Piece tokenizer.
 *
 * <p>{@code String} inputs can be tokenized into inputs for BERT using an instance of {@code
 * SentencePieceTokenizer}. Create an instance of {@code SentencePieceTokenizer} using a byte buffer
 * of the model, which can be loaded from asset using loadMappedFile in {@code FileUtil}.
 * Example usage:
 *
 * <pre>{@code
 * Context context = ApplicationProvider.getApplicationContext();
 * MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, SP_MODEL_DIR);
 * tokenizer = new SentencePieceTokenizer(modelBuffer, true);
 * String testExample = "Good morning, I'm your teacher.\n";
 * List<String> tokens = tokenizer.tokenize(testExample);
 * // output == ["▁good", "▁morning", ",", "▁i", "'", "m", "▁your", "▁teacher", "."];
 * List<Integer> token_ids = tokenizer.convertTokensToIds(tokens)
 * // token_ids == [254, 959, 15, 31, 22, 79, 154, 2197, 9];
 * }</pre>
 */
public final class SentencePieceTokenizer implements Tokenizer {
  private static final String JNI_LIB = "sentencepiece_jni";

  private long modelHandle;
  private boolean enableLowerCase = true;

  private static volatile boolean isLibraryLoaded;

  /**
   * Creates an instance of {@code SentencePieceTokenizer}.
   *
   * @param modelBuffer A {@code MappedByteBuffer} of the sentencepiece model.
   */
  public SentencePieceTokenizer(MappedByteBuffer modelBuffer) {
    this(modelBuffer, true);
  }

  /**
   * Creates an instance of {@code SentencePieceTokenizer}.
   *
   * @param modelBuffer A {@code MappedByteBuffer} of the sentencepiece model.
   * @param enableLowerCase A {@code boolean}, indicating if the input should be lowercased.
   */
  public SentencePieceTokenizer(MappedByteBuffer modelBuffer, boolean enableLowerCase) {
    loadJNILib();
    modelHandle = nativeLoadResource(modelBuffer);
    this.enableLowerCase = enableLowerCase;
  }

  @Override
  public List<String> tokenize(String text) {
    if (enableLowerCase) {
      text = text.toLowerCase(Locale.ROOT);
    }

    List<String> tokens = Arrays.asList(nativeTokenize(modelHandle, text));
    return tokens;
  }

  @Override
  public List<Integer> convertTokensToIds(List<String> tokens) {
    List<Integer> outputIds = new ArrayList<>();
    String[] tokensArray = new String[tokens.size()];
    tokensArray = tokens.toArray(tokensArray);
    for (int idInt : nativeConvertTokensToIds(modelHandle, tokensArray)) {
      Integer id = idInt;
      outputIds.add(id);
    }
    return outputIds;
  }

  @Override
  public void close() {
    if (modelHandle != 0) {
      modelHandle = nativeUnloadResource(modelHandle);
    }
  }

  private static void loadJNILib() {
    if (!isLibraryLoaded) {
      System.loadLibrary(JNI_LIB);
      isLibraryLoaded = true;
    }
  }

  private native long nativeLoadResource(MappedByteBuffer modelBuffer);

  private native long nativeUnloadResource(long processor);

  private native String[] nativeTokenize(long processor, String text);

  private native int[] nativeConvertTokensToIds(long processor, String[] tokens);
}
