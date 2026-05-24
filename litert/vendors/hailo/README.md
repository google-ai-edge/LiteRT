# LiteRT Hailo Integration 🚀

This directory contains the LiteRT integration for Hailo NPU accelerators (specifically targeting the Raspberry Pi AI HAT board, as well as Hailo-8, Hailo-8L, Hailo-10, and Hailo-15 devices). It enables LiteRT to execute machine learning models accelerated by the Hailo NPU.

## Overview ℹ️

The LiteRT Hailo integration consists of two main components:
1. **Compiler Plugin**: A wrapper plugin that reads user-supplied precompiled Hailo Executable Format (`.hef`) files and packages them directly into the output `.tflite` model (removing compilation subprocess complexity from target devices).
2. **Dispatch API**: A native C++ runtime library (`libLiteRtDispatch_Hailo.so`) that loads the embedded `.hef` model and executes it using the open-source **HailoRT** SDK via input/output virtual streams (`VStreams`).

## Documentation 📖

Detailed integration plans, architectural notes, and walkthroughs are available here:
*   [Implementation Plan](./doc/implementation_plan.md) — Architectural design, API mapping, and integration stages.
*   [Walkthrough & Build Guide](./doc/walkthrough.md) — Guide on building the libraries, packaging models, and running on target hardware.
*   [Comparative Analysis](./doc/comparison_notes.md) — Technical differences compared against Qualcomm QNN and Samsung AI LiteCore.
