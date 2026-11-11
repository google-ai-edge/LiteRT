# LiteRT Swift

This directory contains the Swift APIs for LiteRT on Apple platforms:

-   **`LiteRT`**: a type-safe Swift wrapper over the LiteRT C API
    (`litert/c`) for loading, compiling and running
    models with hardware acceleration (CPU, GPU, NPU). Supports iOS and macOS.
-   **`TensorFlowLite`**: the legacy TensorFlow Lite Swift API (`Interpreter`,
    `SignatureRunner`, Core ML and Metal delegates), copied from TensorFlow's
    `tensorflow/lite/swift`. Supports iOS.

Both are built with Bazel. Apps consume them as a Swift package from the
[LiteRT-Swift](https://github.com/google-ai-edge/LiteRT-Swift) repository, which
mirrors the Swift sources of this directory together with its own
`Package.swift`. The `Package.swift` at the root of the LiteRT repository is a
development-only manifest that builds the same package from this directory
against locally provided archives.

This README has two parts:

-   [App Developer Guide](#app-developer-guide): use LiteRT Swift in an
    iOS or macOS app, and run the sample app.
-   [LiteRT Developer Guide](#litert-developer-guide): build, test and develop
    the LiteRT Swift package itself.

## App Developer Guide

### Swift Package Products

The package provides these products:

| Product                  | Contents                                                                 | Platforms  |
| ------------------------ | ------------------------------------------------------------------------ | ---------- |
| `LiteRT`                 | `LiteRT` Swift module; SwiftPM picks the linkage                         | iOS, macOS |
| `LiteRT_static`          | `LiteRT` with the Swift wrapper linked statically                        | iOS, macOS |
| `LiteRT_dynamic`         | `LiteRT` with the Swift wrapper linked dynamically                       | iOS, macOS |
| `LiteRtMetalAccelerator` | Prebuilt Metal GPU accelerator, as a dynamic framework                   | iOS, macOS |
| `CLiteRT_static`         | Static LiteRT C API for C and Objective-C, module `CLiteRT_static`       | iOS        |
| `TensorFlowLite`         | `TensorFlowLite` Swift module, without the Core ML and Metal delegates   | iOS        |

The `LiteRT_static` and `LiteRT_dynamic` product types only affect the Swift
wrapper. The C runtime always comes from `CLiteRT.xcframework` (a dynamic
framework) on iOS and `CLiteRT_mac.xcframework` (a dylib) on macOS.

> [!TIP]
> The `TensorFlowLite` product does not ship with the Core ML and Metal
> delegates. To run models on the GPU, we recommend switching to the `LiteRT`
> and `LiteRtMetalAccelerator` products, and migrating from `Interpreter` to
> `CompiledModel`. See [Adding LiteRT to Your App](#adding-litert-to-your-app)
> and the [Usage Example](#usage-example).

### Adding LiteRT to Your App

The package requires Xcode 15 or later, and an app that targets iOS 15 or
later, or macOS 12 or later. The iOS Simulator slices are arm64 only, so
running in the Simulator requires an Apple silicon Mac.

1.  In Xcode, open your app project and choose **File > Add Package
    Dependencies...**.
2.  Enter `https://github.com/google-ai-edge/LiteRT-Swift` in the search field.
    The package versions follow the
    [LiteRT releases](https://github.com/google-ai-edge/LiteRT/releases), for
    example `2.3.0`. Set **Dependency Rule** to **Up to Next Major Version**
    with the latest release as the minimum version. Then click **Add Package**.
3.  Choose the products to add to your app target: `LiteRT` (see
    [Swift Package Products](#swift-package-products)), and, to use the GPU,
    `LiteRtMetalAccelerator`. Then click **Add Package**.
4.  Add your `.tflite` model to the app target, so that it is copied into the
    app bundle, and `import LiteRT` in your Swift code.

> [!NOTE]
> Before LiteRT-Swift, the package was added from
> `https://github.com/google-ai-edge/LiteRT` with a `release/<version>` branch.
> Existing release branches keep working, but the `Package.swift` on `main` is
> now development-only (see [Architecture](#architecture)), and the LiteRT
> repository is large and uses Git LFS, which can make Xcode fail to check it
> out with `git-lfs: command not found`. To switch, remove the `LiteRT`
> package dependency from the project and add LiteRT-Swift as above.

### Key Types

-   **`Environment`**: Holds runtime environment options (such as compiler or
    dispatch plugin library paths) and the registered hardware accelerators.
-   **`CompiledModel`**: Loads a model from a file path, `Data` buffer or file
    descriptor and compiles it. Runs inference synchronously (`run`) or
    dispatches it asynchronously when supported (`dispatch`), and exposes
    buffer requirements and tensor metadata.
-   **`Options`**: Selects hardware accelerators (CPU, GPU, NPU) and other
    compilation settings. Accelerator-specific settings, such as
    `CpuOptions`, are attached through `ConcreteOptions` and `OpaqueOptions`.
-   **`TensorBuffer`**: Tensor memory backed by host memory or a Metal buffer
    (`MTLBuffer`), with typed `read` and `write` methods.
-   **`TensorType`, `Layout` and `Quantization`**: Tensor element types, shapes
    and quantization parameters.

### Usage Example

The following example loads a model that adds two tensors, compiles it for the
CPU or the GPU, runs inference and reads the result.

```swift
import Foundation
import LiteRT

func runInference(useGpu: Bool) throws {
  // 1. Initialize the LiteRT environment. It locates the Metal accelerator
  //    automatically when the app links the `LiteRtMetalAccelerator` product.
  let environment = try Environment()

  // 2. Configure compilation options.
  let options = try Options()
  if useGpu {
    // Run on the Metal GPU, and fall back to the CPU for unsupported ops.
    try options.setHardwareAccelerators([.gpu, .cpu])
  } else {
    try options.setHardwareAccelerators([.cpu])
    // Optionally tune the CPU backend: XNNPACK delegate with 4 threads.
    let cpuOptions = try CpuOptions()
    try cpuOptions.setKernelMode(.delegate)
    try cpuOptions.setNumThreads(4)
    try options.addConcreteOptions(cpuOptions)
  }

  // 3. Load and compile the model, which the app bundles as a resource.
  guard
    let modelPath = Bundle.main.path(forResource: "simple_add_model", ofType: "tflite")
  else {
    throw CocoaError(.fileNoSuchFile)
  }
  let compiledModel = try CompiledModel(
    filePath: modelPath,
    environment: environment,
    options: options
  )

  // 4. Allocate input and output buffers from the model requirements.
  let inputBuffers = try compiledModel.createInputBuffers()
  let outputBuffers = try compiledModel.createOutputBuffers()

  // 5. Populate the input buffers.
  let input0Data: [Float] = [1.0, 2.0]
  let input1Data: [Float] = [10.0, 20.0]
  try inputBuffers[0].write(input0Data)
  try inputBuffers[1].write(input1Data)

  // 6. Run inference.
  try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

  // 7. Read the output.
  let outputData: [Float] = try outputBuffers[0].read()
  print("Output elements: \(outputData)")  // Expected: [11.0, 22.0]
}
```

For the GPU path, the app also needs the `LiteRtMetalAccelerator` product (see
[Adding LiteRT to Your App](#adding-litert-to-your-app)).

### Running the Sample App

The [litert-samples](https://github.com/google-ai-edge/litert-samples)
repository contains an
[iOS image segmentation app](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/image_segmentation/swift_litert),
built on the `LiteRT` and `LiteRtMetalAccelerator` products. It lets you switch
between the CPU (XNNPACK) and GPU (Metal) backends in its UI.

The app's Xcode project depends on the LiteRT Swift package from GitHub, so
Xcode downloads it when you open the project; nothing has to be built locally.

You need Xcode 15 or later, an iPhone running iOS 15 or later, and an Apple
development team to sign the app with.

1.  Clone the samples repository:

    ```shell
    git clone https://github.com/google-ai-edge/litert-samples.git
    ```

2.  Download the model file into the sample app directory:

    ```shell
    cd litert-samples/samples/litert/image_segmentation/swift_litert
    curl -L -o selfie_multiclass_256x256.tflite \
      https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
    ```

3.  Open Xcode, choose **Open Existing Project...**, and select
    `ImageSegmentation.xcodeproj` in
    `litert-samples/samples/litert/image_segmentation/swift_litert`. Xcode
    resolves the package dependency. Then select the **ImageSegmentation**
    target and, under **Signing & Capabilities**, choose your team.

4.  Connect a physical iPhone, select it as the run destination, and build and
    run the app (**Product > Run**). Check that segmentation works with both
    the CPU and the GPU backend.

To run the sample against unreleased changes, clone the LiteRT repository,
populate `prebuilt/` as described in
[Populating `prebuilt/`](#populating-prebuilt), and replace the package
dependency of the sample project with the LiteRT clone added as a local package
(see
[Developing with Swift Package Manager](#developing-with-swift-package-manager)).

## LiteRT Developer Guide

### Directory Layout

```
swift/
├── BUILD                        Bazel targets (libraries, tests, xcframeworks)
├── Sources/
│   ├── CLiteRT/                 CLiteRT C module (Bazel only)
│   │   ├── CLiteRT.h            Umbrella header of the CLiteRT C module
│   │   └── Info.plist           Info.plist template of the CLiteRT framework
│   ├── LiteRT/                  LiteRT Swift API
│   │   ├── Environment.swift    Runtime environment and its options
│   │   ├── Options.swift        Compilation options, accelerator selection
│   │   ├── ConcreteOptions.swift, CpuOptions.swift, OpaqueOptions.swift
│   │   │                        Accelerator-specific options
│   │   ├── CompiledModel.swift  Model loading, compilation and execution
│   │   ├── TensorBuffer.swift   Host and Metal tensor memory
│   │   ├── TensorType.swift     Element types and layouts
│   │   ├── Quantization.swift   Tensor quantization parameters
│   │   └── LiteRtError.swift    Error type
│   ├── TensorFlowLite/          TensorFlow Lite Swift API
│   └── TensorFlowLiteC/         TensorFlowLiteC C module (Bazel only)
│       └── TensorFlowLiteC.h    Umbrella header of the TensorFlowLiteC C module
└── Tests/
    ├── LiteRT/                  LiteRT unit and integration tests
    └── TensorFlowLite/          TensorFlowLite tests (own BUILD file)
```

`Tests/TensorFlowLite/` has its own `BUILD` file because its all-delegates
build of the wrapper also declares `module_name = "TensorFlowLite"`; keeping it
in a separate package avoids an output collision with `:TensorFlowLite_Swift`.

### Architecture

The Swift code is compiled from source, while the C runtimes it wraps are
consumed as prebuilt `.xcframework` archives:

| Swift module     | C module(s) imported | Prebuilt archive(s)                                                    |
| ---------------- | -------------------- | ---------------------------------------------------------------------- |
| `LiteRT`         | `CLiteRT`            | `CLiteRT.xcframework.zip` (iOS), `CLiteRT_mac.xcframework.zip` (macOS) |
| `TensorFlowLite` | `TensorFlowLiteC`    | `TensorFlowLiteC.xcframework.zip`                                      |

In Bazel, the Swift libraries depend on the C libraries directly. In the Swift
package, each archive is declared as a `binaryTarget`. There are two package
manifests for the same Swift sources:

-   `Package.swift` at the root of the LiteRT repository is the development
    manifest. Its binary targets use `path:` to resolve the archives from the
    `prebuilt/` directory at the repository root, which is not checked in, so
    the package only resolves after the archives are built or downloaded there
    (see [Populating `prebuilt/`](#populating-prebuilt)). It declares no test
    targets, as the tests need Bazel runfiles. It is not meant to be added to
    apps.
-   `Package.swift` at the root of the
    [LiteRT-Swift](https://github.com/google-ai-edge/LiteRT-Swift) repository
    is the release manifest, which apps depend on. LiteRT-Swift is mirrored
    from this directory and holds only the Swift sources (`Sources/LiteRT/`,
    `Sources/TensorFlowLite/`) with its own `Package.swift` and README; the
    Bazel files, the Bazel-only C modules and the tests are left out. Its
    binary targets use `url:` and `checksum:`, pointing at the archives
    published with the matching LiteRT release, and each version tag (for
    example `v2.3.0`) corresponds to the LiteRT release of the same version.
    Keeping the package in a small repository without Git LFS files or
    submodules makes it fast for SwiftPM to clone.

### Prerequisites

-   macOS with Xcode 15 or later (`Package.swift` requires Swift tools 5.9).
-   [Bazelisk](https://github.com/bazelbuild/bazelisk), invoked as `bazel`. The
    Bazel version is pinned in `.bazelversion`.

All commands below run from the repository root.

### Building and Testing with Bazel

The main targets in `//litert/swift` are:

| Target                         | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `:LiteRT_Swift`                | `LiteRT` Swift library                               |
| `:LiteRT_Swift_Tests`          | `LiteRT` tests, run on the macOS host                |
| `:TensorFlowLite_Swift`        | `TensorFlowLite` Swift library                       |
| `Tests/TensorFlowLite:Tests`   | `TensorFlowLite` tests, run in the iOS simulator     |

Run the tests:

```shell
bazel test //litert/swift:LiteRT_Swift_Tests
bazel test //litert/swift/Tests/TensorFlowLite:Tests
```

`:TensorFlowLite_Swift` compiles `CoreMLDelegate.swift` and
`MetalDelegate.swift` only when the corresponding define is set:

-   `--define=use_coreml_delegate=1` for the Core ML delegate.
-   `--define=use_metal_delegate=1` for the Metal delegate.

The TensorFlowLite tests always build with both delegates.

The Swift package does not ship the delegate C libraries, so both package
manifests exclude the two files from the `TensorFlowLite` target.

### Populating `prebuilt/`

The development `Package.swift` of the LiteRT repository expects the following
archives in `prebuilt/` at the repository root. The same archives are published
as assets of each
[LiteRT release](https://github.com/google-ai-edge/LiteRT/releases), which the
LiteRT-Swift manifest downloads.

| Archive                                  | Bazel target                | Extra flags                        |
| ---------------------------------------- | --------------------------- | ---------------------------------- |
| `CLiteRT.xcframework.zip`                | `:CLiteRT`                  |                                    |
| `CLiteRT_static.xcframework.zip`         | `:CLiteRT_static`           |                                    |
| `LiteRtMetalAccelerator.xcframework.zip` | `:LiteRtMetalAccelerator`   |                                    |
| `TensorFlowLiteC.xcframework.zip`        | `:TensorFlowLiteC`          |                                    |
| `CLiteRT_mac.xcframework.zip`            | `:CLiteRT_mac` (see below)  | `--config=macos_arm64`             |

There are two ways to get them:

-   **Download a release.** Sufficient for changes to the Swift sources alone,
    which do not require new archives. Replace `2.3.0` with the release to
    use:

    ```shell
    VERSION=2.3.0
    mkdir -p prebuilt
    for ARCHIVE in CLiteRT CLiteRT_static CLiteRT_mac LiteRtMetalAccelerator \
        TensorFlowLiteC; do
      curl -L -o "prebuilt/${ARCHIVE}.xcframework.zip" \
        "https://github.com/google-ai-edge/LiteRT/releases/download/v${VERSION}/${ARCHIVE}.xcframework.zip"
    done
    ```

-   **Build them**, as described in
    [Building the Prebuilt Archives](#building-the-prebuilt-archives). Required
    whenever the C API, its headers or the C runtime change.

### Building the Prebuilt Archives

`:LiteRtMetalAccelerator` repackages the prebuilt Metal accelerator dylibs from
the `@litert_prebuilts` repository into an `.xcframework`.

#### iOS archives

```shell
bazel build -c opt --config=ios --copt=-fembed-bitcode=off \
  //litert/swift:CLiteRT \
  //litert/swift:CLiteRT_static \
  //litert/swift:LiteRtMetalAccelerator \
  //litert/swift:TensorFlowLiteC
```

As in the release build, `--copt=-fembed-bitcode=off` overrides the
`-fembed-bitcode` that `--config=ios` adds, so it must come after
`--config=ios`. Nothing uses the embedded LLVM bitcode, and it makes up about
two-thirds of the static libraries.

The archives are written to
`bazel-bin/litert/swift/`. Because the output directory
depends on the build flags, look it up with the same flags when copying, for
example:

```shell
IOS_BAZEL_BIN="$(bazel info -c opt --config=ios --copt=-fembed-bitcode=off bazel-bin)"
mkdir -p prebuilt
cp "${IOS_BAZEL_BIN}/litert/swift/CLiteRT.xcframework.zip" prebuilt/
```

#### macOS archive

`:CLiteRT_mac` does not build an `.xcframework` directly. It builds
`CLiteRT_mac.zip`, which holds the universal `libCLiteRT_mac.dylib`, the C
headers with their module map, and the accelerator dylibs it packages (such as
`libLiteRtMetalAccelerator.dylib`). Convert it into
`prebuilt/CLiteRT_mac.xcframework.zip` as follows:

```shell
bazel build -c opt --config=macos_arm64 \
  //litert/swift:CLiteRT_mac
MACOS_BAZEL_BIN="$(bazel info -c opt --config=macos_arm64 bazel-bin)"

mkdir -p prebuilt
cd prebuilt
rm -rf CLiteRT_mac CLiteRT_mac.xcframework CLiteRT_mac.xcframework.zip
unzip -q "${MACOS_BAZEL_BIN}/litert/swift/CLiteRT_mac.zip"
chmod -R u+w CLiteRT_mac

xcodebuild -create-xcframework \
  -library CLiteRT_mac/libCLiteRT_mac.dylib \
  -headers CLiteRT_mac/Headers \
  -output CLiteRT_mac.xcframework

# xcodebuild only copies the main library; copy the other dylibs into the
# macOS slice as well.
for DYLIB in CLiteRT_mac/*.dylib; do
  if [[ "$(basename "${DYLIB}")" != "libCLiteRT_mac.dylib" ]]; then
    cp "${DYLIB}" CLiteRT_mac.xcframework/macos-*/
  fi
done

# On Apple Silicon, dyld refuses to load unsigned dylibs, so ad-hoc sign them,
# then the xcframework itself.
xattr -cr CLiteRT_mac.xcframework
for DYLIB in CLiteRT_mac.xcframework/macos-*/*.dylib; do
  codesign --force -s - "${DYLIB}"
done
codesign --force -s - CLiteRT_mac.xcframework

zip -q -r CLiteRT_mac.xcframework.zip CLiteRT_mac.xcframework
rm -rf CLiteRT_mac CLiteRT_mac.xcframework
cd ..
```

### Developing with Swift Package Manager

With `prebuilt/` populated (see [Populating `prebuilt/`](#populating-prebuilt)),
open the `Package.swift` at the LiteRT repository root in Xcode, or add the
repository root to an app project as a local package (**File > Add Package
Dependencies... > Add Local...**). Edits to the Swift sources are picked up
directly. The development manifest never points at published archives, so it
does not change between releases.

Changes to the Swift sources in this directory are mirrored to the LiteRT-Swift
repository. For each LiteRT release, its `Package.swift` is pointed at the
archives of that release and the repository is tagged with the release version,
which is what apps pick up through
[Adding LiteRT to Your App](#adding-litert-to-your-app).

The products the package provides are listed in
[Swift Package Products](#swift-package-products).
