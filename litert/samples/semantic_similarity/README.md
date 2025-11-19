# Semantic Similarity with Gemma in C++

This project provides a C++ application for running Gemma models to perform
semantic similarity tasks on-device using the LiteRT runtime. It calculates the
cosine similarity between two input sentences.

The model can be obtained here:
[link](https://huggingface.co/litert-community/embeddinggemma-300m)

### Linux CPU

1. **Build the project:**
   ```bash
   bazel build -c opt //litert/litert/samples/semantic_similarity:semantic_similarity
   ```
1. **Run the application:**
   ```bash
   ./bazel-bin/litert/samples/semantic_similarity/semantic_similarity \
     --tokenizer=path/to/tokenizer.model \
     --embedder=path/to/embedder.tflite \
     --sentence1="The quick brown fox jumps over the lazy dog." \
     --sentence2="A fast, dark-colored fox leaps over a sleepy canine." \
     --sequence_length 256
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
./litert/samples/semantic_similarity/deploy_and_run_android.sh \
  --sentence1 "The quick brown fox jumps over the lazy dog." \
  --sentence2 "A fast, dark-colored fox leaps over a sleepy canine." \
  --sequence_length 256
```

**Full Example (All Flags Specified)**

This is a verbose example showing how you would provide all arguments
explicitly. This is useful if you want to use different models or specify the
accelerator.

```bash
./litert/samples/semantic_similarity/deploy_and_run_android.sh \
  --tokenizer "path/to/tokenizer.model" \
  --embedder "path/to/embedder.tflite" \
  --accelerator "cpu" \
  --sentence1 "The quick brown fox jumps over the lazy dog." \
  --sentence2 "A fast, dark-colored fox leaps over a sleepy canine." \
  --sequence_length 256
```

#### Default Flag Values

When you run the script, it uses the following default values if they are not
provided explicitly:

- `--tokenizer`: Defaults to
  `litert/samples/semantic_similarity/models/262144.model`.
- `--embedder`: Defaults to
  `litert/samples/semantic_similarity/models/embedding_gemma_256_input_seq.tflite`.
- `--accelerator`: Defaults to `cpu`. To use `gpu` acceleration, please download
   libLiteRtOpenClAccelerator.so from the [prebuilt](https://github.com/google-ai-edge/LiteRT-LM/tree/main/prebuilt),
   and put it under `./libs`.

#### Important Note on Command Format

The `deploy_and_run_android.sh` script's command-line parsing is basic and
**requires a space** between a flag and its value. It does not support the
`--flag=value` format.

- **Correct:** `--accelerator "cpu"`
- **Incorrect:** `--accelerator=cpu`

#### Additional instructions for NPU acceleration
The provided script also resolves NPU libraries dependencies. Add `--soc_man` to 
setup correct configurations.

*Note: Pre-compiled NPU models available soon.*
##### Qualcomm HTP
```
export QNN_SDK_ROOT=<Qairt SDK path>
./litert/samples/semantic_similarity/deploy_and_run_android.sh \
  --tokenizer "path/to/tokenizer.model" \
  --embedder "path/to/embedder.tflite" \
  --accelerator "npu" \
  --sentence1 "The quick brown fox jumps over the lazy dog." \
  --sentence2 "A fast, dark-colored fox leaps over a sleepy canine." \
  --sequence_length 256 \
  --soc_man "Qualcomm"
```
##### MediaTek APU
```
./litert/samples/semantic_similarity/deploy_and_run_android.sh \
  --tokenizer "path/to/tokenizer.model" \
  --embedder "path/to/embedder.tflite" \
  --accelerator "npu" \
  --sentence1 "The quick brown fox jumps over the lazy dog." \
  --sentence2 "A fast, dark-colored fox leaps over a sleepy canine." \
  --sequence_length 256 \
  --soc_man "MediaTek"
```

