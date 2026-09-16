# Running Examples on Android

This document describes how to deploy and run the examples on an Android device
using the `deploy_and_run_on_android.sh` script.

## Prerequisites

-   An Android device with ADB enabled and connected to the host machine.
-   The project has been built for both the host and Android.

## Usage

The `deploy_and_run_on_android.sh` script can be used to deploy and run both the
`run_mha_cc` and `run_gemma_gqa` examples.

```bash
./tensor/examples/deploy_and_run_on_android.sh [--use_gpu] <binary_name> <binary_build_path>
```

### Arguments

-   `--use_gpu`: (Optional) Whether to use GPU for execution. Defaults to `false`.
-   `<binary_name>`: The name of the binary to run (e.g., `run_mha_cc`, `run_gemma_gqa`).
-   `<binary_build_path>`: The path to the binary build directory (e.g., `blaze-bin`).

### Examples

#### Running `run_mha_cc` on CPU

1.  **Build the model and the binary:**
    ```bash
    blaze build //tensor/examples:multi_head_attention && \
    ./blaze-bin/tensor/examples/multi_head_attention && \
    blaze build --config=android_arm64 //tensor/examples:run_mha_cc
    ```

2.  **Deploy and run on the device:**
    ```bash
    ./tensor/examples/deploy_and_run_on_android.sh run_mha_cc blaze-bin
    ```

#### Running `run_gemma_gqa` on GPU

1.  **Build the model and the binary:**
    ```bash
    blaze build //tensor/examples:gemma_attention && \
    ./blaze-bin/tensor/examples/gemma_attention && \
    blaze build --config=android_arm64 //tensor/examples:run_gemma_gqa
    ```

2.  **Deploy and run on the device:**
    ```bash
    ./tensor/examples/deploy_and_run_on_android.sh --use_gpu run_gemma_gqa blaze-bin
    ```
