# Semantic Similarity with Gemma in C++

This project provides a C++ application for running Gemma models to perform
semantic similarity tasks on-device using the LiteRT runtime. It calculates the
cosine similarity between two input sentences.

The model can be obtained here:
[link](https://huggingface.co/litert-community/embeddinggemma-300m)

### Linux CPU

1. **Build the project:**
   ```bash
   bazel build -c opt //third_party/odml/litert/litert/samples/semantic_similarity:semantic_similarity
   ```
1. **Run the application:**
   ```bash
   ./bazel-bin/third_party/odml/litert/litert/samples/semantic_similarity/semantic_similarity \
     --tokenizer=path/to/tokenizer.model \
     --embedder=path/to/embedder.tflite \
     --sentence1="The quick brown fox jumps over the lazy dog." \
     --sentence2="A fast, dark-colored fox leaps over a sleepy canine."
   ```

### Android (Phone)

The provided script automates building for Android, pushing the necessary files
to a connected device, and running the application.

- Ensure your Android device is connected and `adb` is available in your
  environment.
- The script requires `--sentence1` and `--sentence2` arguments.

**Minimal Example (Using Defaults)**

This command relies on the script's default values for the tokenizer, embedder,
and accelerator.

```bash
./third_party/odml/litert/litert/samples/semantic_similarity/deploy_and_run_android.sh \
  --sentence1 "The quick brown fox jumps over the lazy dog." \
  --sentence2 "A fast, dark-colored fox leaps over a sleepy canine."
```

**Full Example (All Flags Specified)**

This is a verbose example showing how you would provide all arguments
explicitly. This is useful if you want to use different models or specify the
accelerator.

```bash
./third_party/odml/litert/litert/samples/semantic_similarity/deploy_and_run_android.sh \
  --tokenizer "path/to/tokenizer.model" \
  --embedder "path/to/embedder.tflite" \
  --accelerator "cpu" \
  --sentence1 "The quick brown fox jumps over the lazy dog." \
  --sentence2 "A fast, dark-colored fox leaps over a sleepy canine."
```

#### Default Flag Values

When you run the script, it uses the following default values if they are not
provided explicitly:

- `--tokenizer`: Defaults to
  `third_party/odml/litert/litert/samples/semantic_similarity/models/262144.model`.
- `--embedder`: Defaults to
  `third_party/odml/litert/litert/samples/semantic_similarity/models/embedding_gemma_256_input_seq.tflite`.
- `--accelerator`: Defaults to `cpu`.

#### Important Note on Command Format

The `deploy_and_run_android.sh` script's command-line parsing is basic and
**requires a space** between a flag and its value. It does not support the
`--flag=value` format.

- **Correct:** `--accelerator "cpu"`
- **Incorrect:** `--accelerator="cpu"`
