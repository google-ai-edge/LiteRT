# LiteRT Kotlin API Benchmark Tool

## Build and Install

```sh
blaze build -c opt --android_platforms=//buildenv/platforms/android:arm64-v8a \
  litert/kotlin/internal/java/com/google/ai/edge/litert/benchmark \
  && adb install blaze-bin/litert/kotlin/internal/java/com/google/ai/edge/litert/benchmark/benchmark.apk
```

## Example

### Prepare Models

This tool should work with any tflite models.

In this guide, we use the models used in go/odml-models-dashboard,

- `/cns/md-d/home/mediapipe/odml_models/jax/`
- `/cns/md-d/home/mediapipe/odml_models/pytorch/`

```sh
MODEL_DIR="$HOME/models/benchmark"
mkdir -p ${MODEL_DIR:?}

# Copying the models
fileutil cp -R -parallelism 1280 /cns/md-d/home/mediapipe/odml_models/jax ${MODEL_DIR:?}
fileutil cp -R -parallelism 1280 /cns/md-d/home/mediapipe/odml_models/pytorch ${MODEL_DIR:?}
```

### Launch the Benchmark Activity

```sh
# Push the model if it is not on the device.
maybe_push() {
  local local="$1"
  local remote="$2"

  if adb shell ls "$remote"; then
    echo "skip pushing '$remote'"
  else
    adb shell mkdir -p $(dirname $remote)
    adb push "$local" "$remote"
  fi
}

benchmark() {
    local MODEL_NAME="$1"
    local NUM_ITERATIONS="$2"
    local NUM_INFERENCES_PER_ITERATION="$3"

    # Prepare the model
    maybe_push ${MODEL_LOCAL_PATH:?} ${MODEL_REMOTE_PATH:?}

    # Launch the Benchmark activity
    adb shell am start \
        -n com.google.ai.edge.litert.benchmark/.BenchmarkActivity \
        --es "model_path" "${MODEL_REMOTE_PATH:?}" \
        --ei "num_iterations" "${NUM_ITERATIONS:?}" \
        --ei "num_inferences_per_iteration" "${NUM_INFERENCES_PER_ITERATION:?}"
}

# Parameters
# MODEL_NAME="jax/mobilenet_v2.tflite"
# NUM_ITERATIONS=100
# NUM_INFERENCES_PER_ITERATION=1
benchmark "jax/mobilenet_v2.tflite" "100" "1"
```
