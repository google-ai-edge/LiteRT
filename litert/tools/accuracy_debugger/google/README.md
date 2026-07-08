# LiteRT Accuracy Debugger: Google Internal Usage Guide

This directory contains the internal automation and documentation for the
LiteRT Accuracy Debugger. The tool is designed to isolate numerical regressions
on hardware accelerators (NPU/GPU) using surgical operation partitioning and
cumulative drift tracking.

## 1. Quick Start

To run the accuracy debugger on a Qualcomm-based Android device, use the
provided driver script from the root of your workspace. This script automates
the cross-compilation, deployment of vendor libraries, and on-device execution.

### Example Usage:
```bash
sh litert/tools/accuracy_debugger/google/debug_accuracy_qc.sh \
  --accelerator=npu \
  --model_path=~/path/to/your_model.tflite \
  --signature_index=0 \
  --max_ops=100 \
  --use_accel_output_as_input=true
```

**Required Flags for `debug_accuracy_qc.sh`:**

*   `--accelerator`: The target backend (`npu` or `gpu`).
*   `--model_path`: Local path to the `.tflite` model.

**Optional Flags:**

*   `[extra_flags...]`: Any additional CLI flags supported by the debugger
    (see table below).

### Targeting a Specific Device
By default, the script uses the first available ADB device. If you have
multiple devices connected, specify the serial number using the `ADB_SERIAL`
environment variable:

```bash
ADB_SERIAL=<adb_serial> sh \
  litert/tools/accuracy_debugger/google/debug_accuracy_qc.sh \
  --accelerator=npu --model_path=model.tflite
```

---

## 2. Core Capabilities

*   **Surgical Partitioning:** Dynamically extracts a single operation into a
    standalone `.tflite` model for isolated verification on the target hardware.
*   **Multi-Op Chunking (Islands):** Supports partitioning the model into
    multi-op "islands" based on a provided list of boundary tensor names.
*   **StableHLO Support:** Correctly handles StableHLO Composite operations by
    automatically cloning and re-linking their decomposition subgraphs.
*   **Dual-Path Execution:** Runs every op/chunk on both a reference path
    (CPU or GPU FP32) and the target accelerator path simultaneously.
*   **Cumulative Drift Tracking:** If `--use_accel_output_as_input=true` is set,
    the accelerator path consumes the noisy output from the *previous* op,
    allowing you to see how errors compound across layers.
*   **Detailed Diagnostics:** Reports metrics including Max Absolute Difference,
    Cosine Similarity, SNR, PSNR, and NaN detection.
*   **Value Range Monitoring:** Displays the value range (min/max) for both
    reference and accelerator outputs to help identify clipping or overflows.
*   **Automated Artifact Export:** If a regression is detected and
    `--save_failing_models=true` is set, the tool saves the failing `.tflite`
    models and their corresponding `.bin` inputs for local reproduction.
*   **Dump Only Mode:** Allows extracting and saving subgraphs without running
    inference, useful for inspecting model partitions.

---

## 3. CLI Flags Reference

You can pass these flags as additional arguments to the `debug_accuracy_qc.sh`
script.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--signature_index` | `0` | The model signature to debug. |
| `--max_ops` | `-1` | Limits run to first N ops. |
| `--boundary_tensors`| `""` | Comma-separated list for multi-op chunking. |
| `--use_gpu_ref` | `false`| Use GPU FP32 as reference instead of CPU. |
| `--dump_only` | `false`| Only export models, do not run accuracy check. |
| `--output_dir` | `/data/local/...`| Device path for CSV and artifacts. |
| `--use_accel_output_as_input`| `true` | Propagates drifted outputs as inputs. |
| `--save_failing_models` | `false`| Saves models/inputs for failing ops. |
| `--skip_unsupported_npu_ops`| `true` | Skips unstable ops (e.g. CUSTOM). |
| `--summary_max_rows`| `50` | Max rows in console summary table. |
| `--sort_by` | `cos_sim`| Sort by: index, max_diff, mse, cos_sim, etc. |

### Accuracy Thresholds
An operation is marked as **REGRESSION** if it fails any of these thresholds:

*   `--max_diff` (Default: `5e-3`)
*   `--mse` (Default: `1e-5`)
*   `--cosine_similarity` (Default: `0.99`)
*   `--snr` (Default: `30.0` dB)
*   `--psnr` (Default: `35.0` dB)

---

## 4. Viewing Results

After the run completes, the tool generates a high-level summary in the
console. Results are automatically pulled from the device to your local
machine:

**Local Path:** `/tmp/accuracy_debug_result/accuracy_summary.csv`

The summary table and CSV provide detailed metrics for every operation or
partition, including the `Tensor Name` and `NaN` status.

### Exported Model Naming
When models are dumped (via `--dump_only` or `--save_failing_models`), they use
the following naming convention for clarity:
`op_<index>_<opcode_or_partition>_i<num_inputs>_o<num_outputs>.tflite`
