# Comparison of NPU Implementations in LiteRT: Hailo vs. Qualcomm vs. Samsung

This document highlights the differences between the proposed Hailo NPU integration and the existing Qualcomm QNN and Samsung AI LiteCore integrations in LiteRT.

---

## High-Level Comparison Table

| Dimension | Qualcomm Integration (QNN) | Samsung Integration (LiteCore) | Proposed Hailo Integration |
| :--- | :--- | :--- | :--- |
| **Target Platform** | Snapdragon SoCs | Exynos SoCs | Raspberry Pi (Hailo HAT), Hailo-8/10/15 PCIe boards |
| **SDK Type** | Proprietary QAIRT/QNN SDK | Proprietary Exynos AI LiteCore | **Open-source HailoRT** (MIT License) |
| **Driver License** | Proprietary | Proprietary | GPL v2 (PCIe/NPU kernel driver) |
| **Compilation API** | In-process C++ QNN APIs (`libQnnSystem.so`, HTP backend) | In-process Exynos compiler APIs (`libexynos_ai_litecore.so`) | **Out-of-process Python CLI/API** (Hailo Dataflow Compiler) |
| **Output format** | Context Binary (`.bin` / `.dlc`) | Compiled Graph Binary | **Hailo Executable Format (`.hef`)** |
| **Runtime SDK** | QNN Runtime API | Exynos LiteCore Graph Executor | **HailoRT C++ API (`hailo/hailort.hpp`)** |
| **Execution Flow** | In-process execution via QNN HTP Graph | In-process execution via Samsung LiteCore Graph | Streaming read/write via **HailoRT Virtual Streams (`VStream`)** |
| **Buffer Types** | AHardwareBuffer, Ion, FastRPC, VTCM | Custom Samsung memory wrappers | Standard host Cpu/AHWB buffers streamed into NPU |

---

## Detailed Differences

### 1. Model Compilation Flow (Compiler Plugin)
- **Qualcomm / Samsung**: Run compilation entirely in-process using C++ shared libraries (`libQnnSystem.so` and `libexynos_ai_litecore.so` respectively) loaded at build-time / AOT-time.
- **Hailo**: The Hailo compiler (Dataflow Compiler - DFC) is Python-based. The Hailo compiler plugin will output a temporary `.tflite` file from the selected subgraph and run a subprocess invocation calling either the Python API or the `hailo` command line interface to generate the `.hef` output.

### 2. Runtime Loading & Execution (Dispatch API)
- **Qualcomm**: Loads the context binary (.bin) via `QnnContext_loadFromBinary`. Runs inference via graph inputs/outputs bound directly to Qualcomm's tensor structures.
- **Samsung**: Loads and executes Exynos compiled binary using Samsung's in-process engine.
- **Hailo**: Loads the `.hef` file via `Hef::create_from_buffer`. Executes inference by opening virtual input and output streams (`VStreams`), feeding inputs to `input_vstream->write()`, and collecting outputs via `output_vstream->read()`. HailoRT handles internal driver queues, scheduling, and device execution.

### 3. Open-Source Status & Redistribution
- **Qualcomm / Samsung**: Headers are downloaded at build time, but binaries are entirely proprietary closed-source and preloaded on target devices by OEMs. 
- **Hailo**: The user-space library `hailort` is completely open source under the MIT License. It can be compiled directly from source and bundled with LiteRT if desired, or dynamically linked as an open-source dependency (e.g. from the apt repository of Raspberry Pi OS).

### 4. Memory Buffering & Handshaking
- **Qualcomm**: Uses complex hardware buffer mappings (such as `AHardwareBuffer` or custom `Ion` and `FastRPC` allocations) and negotiates VTCM memory sizes for optimal latency.
- **Samsung**: Uses custom Exynos system allocations.
- **Hailo**: The Dispatch implementation uses standard host memory (`kLiteRtTensorBufferTypeCpu`). Data is copied to HailoRT's `VStream` write buffers, and HailoRT transfers the data internally to the NPU's SRAM/DRAM.
