# LiteRT Qualcomm (QNN) Integration 🚀

This directory contains the LiteRT integration for Qualcomm Neural Network (QNN) accelerators. It enables LiteRT to offload machine learning models to Qualcomm NPUs, GPUs, and DSPs for hardware-accelerated inference.

## Overview ℹ️

The LiteRT Qualcomm integration consists of two main components:
1.  **Compiler Plugin**: Legalizes and compiles LiteRT graphs into QNN graphs, supporting both online and offline compilation.
2.  **Dispatch API**: Manages the execution of compiled QNN graphs on the target device.

By using this integration, developers can leverage the high performance and energy efficiency of Qualcomm NPUs in their LiteRT applications.

## Change Note 📝

*   **Initial Release**: Basic support for compilation and dispatch on Qualcomm QNN backends.
*   *Add your recent changes here to keep users informed.*

## Supported Devices 📱

LiteRT supports a wide range of Qualcomm SoCs through the QNN SDK.

### Top-Tier Supported SoCs:
*   **Snapdragon 8 Gen 5** (SM8850)
*   **Snapdragon 8 Elite** (SM8750)
*   **Snapdragon 8 Gen 3** (SM8650)

For a complete and up-to-date list of supported devices, please refer to [supported_soc.csv](./supported_soc.csv).

## Build Instructions 🛠️

Detailed build instructions can be found in [Qualcomm_Build.md](./Qualcomm_Build.md).

## Configuration Options ⚙️

You can configure the Qualcomm backend using `LrtQualcommOptions`. Below are the available options defined in [litert_qualcomm_options.h](../../c/options/litert_qualcomm_options.h).

### General Settings
*   **Log Level**: Set the logging level for Qualcomm SDK libraries.

### Compilation Options
*   **Use INT64 Bias as INT32**: Convert bias tensors from `int64` to `int32` for `FullyConnected` and `Conv2D` ops. Defaults to `true`.
*   **Enable Weight Sharing**: Allow different subgraphs to share weight tensors (supported on x86 AOT).
*   **Use Conv HMX**: Enable short convolution HMX for better performance (may affect accuracy).
*   **Use Fold ReLU**: Enable folding ReLU operations into convolutions.
*   **VTCM Size**: Configure Vector Tightly Coupled Memory size.
*   **Num HVX Threads**: Set the number of HVX threads.
*   **Optimization Level**: Set optimization level for inference or prepare.

### Dispatch Options
*   **Backend**: Select the target backend (`GPU` (not yet supported), `HTP`, `DSP`, `IR`).
*   **HTP/DSP Performance Mode**: Configure for performance or power efficiency.
*   **Profiling**: Set profiling level (Off, Basic, Detailed, Linting, Optrace).
*   **Graph Priority**: Set execution priority for the graph.

## Tooling 🛠️

*   **Optrace Profiling**: See [optrace_profiling](./debugger/optrace_profiling) for details on debugging and profiling.
*   **LiteRT Tools**: Refer to `litert/tools` for general LiteRT tools.
*   **QAIRT Native Tools**: You can also use native tools provided by the Qualcomm AI Rule Toolkit (QAIRT).


## References & Links 🔗

*   **Detailed Compiler Info**: See [Qualcomm_QNN_Compiler.md](./compiler/Qualcomm_QNN_Compiler.md) for supported ops and data types.
*   **Google Dev Site**: [LiteRT Qualcomm Documentation](https://ai.google.dev/edge/litert/next/qualcomm)
*   **Blog Posts**:
    *   [Unlocking Peak Performance on Qualcomm NPU with LiteRT](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)
    *   [Building Real-World On-Device AI with LiteRT and NPU](https://developers.googleblog.com/building-real-world-on-device-ai-with-litert-and-npu/)
*   **Partner Library Documentation**: [LiteRT LM NPU Qualcomm](https://ai.google.dev/edge/litert/next/litert_lm_npu#qualcomm)
