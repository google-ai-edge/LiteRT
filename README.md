# LiteRT

<p align="center">
  <img src="./g3doc/sources/litert_logo.png" alt="LiteRT Logo"/>
</p>

Google's on-device framework for high-performance ML & GenAI deployment on edge
platforms, via efficient conversion, runtime, and optimization

📖 [Get Started](#-installation) | 🤝 [Contributing](#-contributing) | 📜
[License](#-license) | 🛡 [Security Policy](SECURITY.md) | 📄
[Documentation](https://ai.google.dev/edge/litert)

## Build Status

| Nightly Builds                                                                                                                                                                                                                                                         | Continuous Builds                                                                                                                                                                                                                                                                                                                                                                                                                      | Other Builds                                                                                                                                                                                                                          |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [![Linux Nightly Wheel](https://github.com/google-ai-edge/LiteRT/actions/workflows/linux_nightly_wheel.yml/badge.svg)](https://github.com/google-ai-edge/LiteRT/actions/workflows/linux_nightly_wheel.yml) <br> [![macOS Nightly Wheel](https://github.com/google-ai-edge/LiteRT/actions/workflows/macos_nightly_wheel.yml/badge.svg)](https://github.com/google-ai-edge/LiteRT/actions/workflows/macos_nightly_wheel.yml) | [![macOS arm64](https://github.com/google-ai-edge/LiteRT/actions/workflows/macos-arm64.yml/badge.svg)](https://github.com/google-ai-edge/LiteRT/actions/workflows/macos-arm64.yml) <br> [![Linux x86_64](https://github.com/google-ai-edge/LiteRT/actions/workflows/linux_x86_64.yml/badge.svg)](https://github.com/google-ai-edge/LiteRT/actions/workflows/linux_x86_64.yml) <br> [![Windows x86_64](https://github.com/google-ai-edge/LiteRT/actions/workflows/windows_x86_64.yml/badge.svg)](https://github.com/google-ai-edge/LiteRT/actions/workflows/windows_x86_64.yml) | [![CMake Android Linux x86_64](https://github.com/google-ai-edge/LiteRT/actions/workflows/cmake_android_linux_x86_64.yml/badge.svg)](https://github.com/google-ai-edge/LiteRT/actions/workflows/cmake_android_linux_x86_64.yml) |


## Description

LiteRT continues the legacy of TensorFlow Lite as the trusted, high-performance
runtime for on-device AI.

LiteRT features advanced GPU/NPU acceleration, delivers superior ML & GenAI
performance, making on-device ML inference easier than ever.

### 🌟 What's New

- **🆕 New LiteRT Compiled Model API**: Streamline development with automated accelerator
  selection, true async execution, and efficient I/O buffer handling.

  - Automated accelerator selection vs explicit delegate creation
  - Async execution for faster overall execution time
  - Easy NPU runtime and model distribution
  - Efficient I/O buffer handling

- **🤖 Unified NPU Acceleration**: Offer seamless access to NPUs from major
  chipset providers with a consistent developer experience. LiteRT NPU,
  previously under Early access program is available to all
  users: https://ai.google.dev/edge/litert/next/npu

- **⚡ Best-in-class GPU Performance**: Use state-of-the-art GPU acceleration for
  on-device ML. The new buffer interoperability enables zero-copy and minimizes
  latency across various GPU buffer types.

- **🧠 Superior Generative AI inference**: Enable the simplest integration with
  the best performance for GenAI models.

## 💻 Platforms Supported

LiteRT is designed for cross-platform deployment on a wide range of hardware.

| Platform   | CPU Support | GPU Support           | NPU Support                                                       |
| ---------- | ----------- | --------------------- | ----------------------------------------------------------------- |
| 🤖 Android | ✅          | ✅OpenCL<br>✅OpenGL | Google Tensor\*<br>✅ Qualcomm<br>✅ MediaTek<br>S.LSI\*<br>Intel\* |
| 🍎 iOS     | ✅          | ✅ Metal                 | ANE\*                                                             |
| 🐧 Linux   | ✅          | ✅ WebGPU                | N/A                                                               |
| 🍎 macOS   | ✅          | ✅ WebGPU<br>✅ Metal                 | ANE\*                                                             |
| 💻 Windows | ✅          | ✅ WebGPU                | Intel\*                                                           |
| 🌐 Web     | ✅          | ✅ WebGPU                | Coming soon                                                       |
| 🧩 IoT     |  ✅           | ✅ WebGPU                      | Broadcom\*<br>Raspberry Pi\*                                      |

*\*Coming soon*

## Model Coverage and Performance

Coming soon...

## 🏁 Installation

For a comprehensive guide to setting up your application with LiteRT, see
the [Get Started guide](https://ai.google.dev/edge/litert).

You can build LiteRT from source:

1. Start a docker daemon.
1. Run `build_with_docker.sh` under `docker_build/`

The script automatically creates a Linux Docker image, which allows you to build
artifacts for Linux and Android (through cross compilation). See build
instructions in
[CMake build instructions](./g3doc/instructions/CMAKE_BUILD_INSTRUCTIONS.md) and
[Bazel build instructions](./g3doc/instructions/BUILD_INSTRUCTIONS.md) for more
information on how to build runtime libraries with the docker container.

For more information about using docker interactive shell or building different
targets, please refer to `docker_build/README.md`.

## 🗺 Choose Your Adventure

Every developer's path is different. Here are a few common journeys to help you
get started based on your goals:

### 1. 🔄 I have a PyTorch model...

- **Goal**: Convert a model from PyTorch to run on LiteRT.
- **Path1 (classic models)**: Use the
  [LiteRT Torch Converter](https://github.com/google-ai-edge/litert-torch) to
  transform your PyTorch model into the `.tflite` format, and use AI Edge
  Quantizer to optimize the model for optimal performance under resource
  constraints. From there, you can deploy it using the standard LiteRT runtime.
- **Path2 (LLMs)**: Use
  [LiteRT Generative Torch API](https://github.com/google-ai-edge/litert-torch/tree/main/litert_torch/generative) to
  reauthor and convert your PyTorch LLMs into Apache format, and deploy it using
  [LiteRT LM](https://github.com/google-ai-edge/litert-lm).

### 2. 🌱 I'm new to on-device ML...

- **Goal**: Run a pre-trained model (like image segmentation) in a mobile app
  for the first time.
- **Path1 (Beginner dev)**: Follow step-by-step instructions via Android Studio
  to create a
  [Real-time segmentation App](https://developers.google.com/codelabs/litert-image-segmentation-android#0)
  for CPU/GPU/NPU inference. Source code
  [link](https://github.com/google-ai-edge/litert-samples/tree/main/compiled_model_api/image_segmentation).
- **Path2 (Experienced dev)**: Start with the
  [Get Started guide](https://ai.google.dev/edge/litert/next/get_started), find
  a pre-trained .tflite model on [Kaggle Models](https://www.kaggle.com/models?framework=tfLite&size=small),
  and use the standard LiteRT runtime to integrate it into your Android or iOS
  app.

### 3. ⚡ I need to maximize performance...

- **Goal**: Accelerate an existing model to run faster and more efficiently
  on-device.
- **Path**:
  - Explore the [LiteRT API](https://ai.google.dev/edge/api/litert/c) to
    easily leverage hardware acceleration.
  - **For working with Generative AI**: Dive into
    [LiteRT LM](https://github.com/google-ai-edge/LiteRT-LM), our specialized
    solution for running GenAI models.

### 4. 🧠 I'm working with Generative AI...

- **Goal**: Deploy a large language model (LLM) or diffusion model on a mobile
  device.
- **Path**: Dive into [LiteRT LM](https://github.com/google-ai-edge/LiteRT-LM),
  our specialized solution for running GenAI models. You'll focus on model
  quantization and optimizations specific to large model architectures.

## 🗺 Roadmap

Our commitment is to make LiteRT the best runtime for any on-device ML
deployment. Our product strategies are:

- **Expanding Hardware Acceleration**: Broadening our support for NPUs and
  improving performance across all major hardware accelerators.
- **Generative AI Optimizations**: Introducing new optimizations and features
  specifically for the next wave of on-device generative AI models.
- **Improving Developer Tools**: Building better tools for debugging, profiling,
  and optimizing models.
- **Platform Support**: Enhancing support for core platforms and exploring new
  ones.

## 🙌 Contributing

We welcome contributions to LiteRT. Please see the
[CONTRIBUTING.md](CONTRIBUTING.md) file for more information on how to
contribute.

## 💬 Getting Help

We encourage you to reach out if you need help.

- **GitHub Issues**: For bug reports and feature requests, please file a new
  issue on our [GitHub Issues](https://github.com/google-ai-edge/LiteRT/issues)
  page.
- **GitHub Discussions**: For questions, general discussions, and community
  support, please visit our
  [GitHub Discussions](https://github.com/google-ai-edge/LiteRT/discussions).

## 🔗 Related Products

LiteRT is part of a larger ecosystem of tools for on-device machine learning.
Check out these other projects from Google:

- **[LiteRT Samples](https://github.com/google-ai-edge/litert-samples/tree/main)**: A
  collection of LiteRT sample apps.
- **[LiteRT Torch Converter](https://github.com/google-ai-edge/litert-torch)**:
  A tool in LiteRT to convert PyTorch models into the LiteRT(.tflite) format for
  on-device deployment.
- **[LiteRT Generative Torch API](https://github.com/google-ai-edge/litert-torch/tree/main/litert_torch/generative)**: A
  library in LiteRT to reauthor LLMs for efficient conversion and on-device
  inference.
- **[LiteRT-LM](https://github.com/google-ai-edge/litert-lm)**: A library to
  efficiently run Large Language Models (LLMs) across edge platforms, built on
  top of LiteRT.
- **[XNNPACK](https://github.com/google/XNNPACK)**: A highly optimized library
  of neural network inference operators for ARM, x86, and WebAssembly
  architectures that provides high-performance CPU acceleration for LiteRT.
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)**: A framework for
  building cross-platform, customizable ML solutions for live and streaming
  media.

## ❤️ Code of Conduct

This project is dedicated to fostering an open and welcoming environment. Please
read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the standards of
behavior we expect from all participants in our community.

## 📜 License

LiteRT is licensed under the [Apache-2.0 License](LICENSE).
