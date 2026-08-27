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

import java.util.List;

/**
 * The interface for tokenizer.
 *
 * A tokenizer class that implements this interface should have a method tokenize to tokenize
 * <p>{@code String} into an array of {@code String} tokens, and a method convertTokensToIds to
 * convert the {@code String} tokens to{@code Integer} ids that can be fed into the models.
 *
 * }</pre>
 */
public interface Tokenizer extends AutoCloseable {

  /**
   * Performs tokenization on a {@code String} input.
   *
   * @param text A {@code String} elements that will be tokenized.
   * @return An array {@code String} containing the token results.
   */
  public abstract List<String> tokenize(String text);

  /**
   * Converts the tokens to the ids.
   *
   * @param tokens A {@code List} of {@code String} that will be converted to ids.
   * @return A {@code List} of {@code Integer} containing the ids of the tokens.
   */
  public abstract List<Integer> convertTokensToIds(List<String> tokens);
}

