# LiteRT Swift package

## Description

This repository contains the Swift package for
[LiteRT](https://ai.google.dev/edge/litert), Google's runtime for on-device
machine learning, on iOS and macOS. It provides two modules:

-   **`LiteRT`**: a type-safe Swift wrapper over the LiteRT C API for loading,
    compiling and running models with hardware acceleration (CPU, GPU, NPU).
    Supports iOS and macOS.
-   **`TensorFlowLite`**: the legacy TensorFlow Lite Swift API (`Interpreter`,
    `SignatureRunner`). Supports iOS.

The Swift sources are developed in the
[LiteRT repository](https://github.com/google-ai-edge/LiteRT/tree/main/litert/swift)
and mirrored here, so that Swift Package Manager only has to clone this small
repository. The native runtimes are prebuilt `.xcframework` archives published
with each [LiteRT release](https://github.com/google-ai-edge/LiteRT/releases),
downloaded on demand and verified by checksum.

## Requirements

-   [Xcode](https://developer.apple.com/xcode/) version 15.0 or later
-   An app targeting iOS 15 or later, or macOS 12 or later
-   An Apple silicon Mac to run in the iOS Simulator, as the Simulator slices
    are arm64 only

## Installation

1.  Follow the instructions for
    [adding package dependencies to your app in Xcode](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app).

2.  In the search field of the package dialog, enter this GitHub repository:

    ```
    https://github.com/google-ai-edge/LiteRT-Swift
    ```

3.  Select the version of LiteRT that you want to use. The package versions
    follow the
    [LiteRT releases](https://github.com/google-ai-edge/LiteRT/releases). For
    new projects, we recommend specifying the latest version and using the
    **Up to Next Major Version** dependency rule.

4.  Choose the products to add to your app target:

    | Product                  | Contents                                                               | Platforms  |
    | ------------------------ | ---------------------------------------------------------------------- | ---------- |
    | `LiteRT`                 | `LiteRT` Swift module; SwiftPM picks the linkage                       | iOS, macOS |
    | `LiteRT_static`          | `LiteRT` with the Swift wrapper linked statically                      | iOS, macOS |
    | `LiteRT_dynamic`         | `LiteRT` with the Swift wrapper linked dynamically                     | iOS, macOS |
    | `LiteRtMetalAccelerator` | Prebuilt Metal GPU accelerator, as a dynamic framework                 | iOS, macOS |
    | `CLiteRT_static`         | Static LiteRT C API for C and Objective-C, module `CLiteRT_static`     | iOS        |
    | `TensorFlowLite`         | `TensorFlowLite` Swift module, without the Core ML and Metal delegates | iOS        |

    Most apps add `LiteRT` and, to run models on the GPU,
    `LiteRtMetalAccelerator`.

5.  Add your `.tflite` model to the app target, so that it is copied into the
    app bundle, and `import LiteRT` in your Swift code.

## Sample App

See the
[iOS image segmentation app](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/image_segmentation/swift_litert)
in the [litert-samples](https://github.com/google-ai-edge/litert-samples)
repository, which uses the `LiteRT` and `LiteRtMetalAccelerator` products and
lets you switch between the CPU and GPU backends.

## Documentation

Documentation for LiteRT is available on
[ai.google.dev/edge/litert](https://ai.google.dev/edge/litert). The
[LiteRT Swift README](https://github.com/google-ai-edge/LiteRT/blob/main/litert/swift/README.md)
describes the Swift API, has a usage example, and explains how the package is
built and released.

## Contributing

The sources in this repository are mirrored from the
[LiteRT repository](https://github.com/google-ai-edge/LiteRT). See the
[Contributing guide](CONTRIBUTING.md) before opening a pull request.

## Support

For issues with LiteRT, including this package,
[file an issue](https://github.com/google-ai-edge/LiteRT/issues) on the LiteRT
issue tracker.

## Versioning

This repository adheres to [semantic versioning](https://semver.org/) to
indicate when backwards-incompatible changes are introduced. Each version tag
corresponds to the LiteRT release of the same version.

## Code of Conduct

This project is dedicated to fostering an open and welcoming environment.
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the
standards of behavior we expect from all participants.

## Security

To report a security issue, follow [SECURITY.md](SECURITY.md). Eligibility for
the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

## License

LiteRT is licensed under the [Apache 2.0 License](LICENSE).
