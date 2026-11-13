# Image Segmentation with LiteRT in Google Play services

This sample is a variant of the
[`android`](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/image_segmentation/kotlin_cpu_gpu/android)
image segmentation sample. It runs the same selfie multiclass segmentation model
with the LiteRT CompiledModel API, but on the LiteRT runtime that ships in
**Google Play services** instead of a runtime bundled in the APK.

Using LiteRT in Google Play services:

*   Keeps your APK small: the LiteRT runtime and the GPU accelerator are not
    packaged in the app.
*   Keeps the runtime up to date: Google Play services updates it independently
    of your app.

## How it differs from the `android` sample

*   **Dependencies:** `com.google.android.gms:play-services-litert` replaces the
    bundled `com.google.ai.edge.litert:litert` runtime.
*   **Initialization:** before creating a `CompiledModel`, the app gets a LiteRT
    `Environment` from Google Play services and passes it to
    `CompiledModel.create(...)`:

    ```kotlin
    val environment =
      LiteRt.initialize(context, initializationOptions { setEnableGpu(true) }).await()
    val model =
      CompiledModel.create(
        context.assets,
        "selfie_multiclass.tflite",
        CompiledModel.Options(Accelerator.GPU, Accelerator.CPU),
        environment,
      )
    ```

    **Note:** listing `Accelerator.CPU` after `Accelerator.GPU` enables CPU
    fallback: ops that the GPU doesn't support run on the CPU, and the rest of
    the model still runs on the GPU. With `Accelerator.GPU` alone,
    `CompiledModel.create` throws if any op in the model is not supported on the
    GPU. We recommend always adding `Accelerator.CPU` when you request the GPU.

    See `ImageSegmentationHelper.kt`. The environment is created once and shared
    by every `CompiledModel`, and it is closed when the `ViewModel` is cleared.
    If the model cannot run on the GPU, the app reports the error and falls back
    to the CPU.

*   **Toolchain:** `play-services-litert` is built with Kotlin 2.4.0, so the
    sample uses Kotlin 2.4.0 (see `gradle/libs.versions.toml`). If you enable
    minification in your own app, use R8 9.1 or newer.

## Requirements

*   An Android device or emulator (API level 24 or higher) with Google Play
    services. A physical arm64 device is recommended to try the GPU.
*   Android Studio, or the Android SDK and JDK 17 to build from the command
    line.

## Build and run

To build and install a debug version:

```sh
./gradlew :app:installDebug
```

To build and install an optimized release version:

```sh
./gradlew :app:installRelease
```

Use the release version for any size or performance measurements; the debug
version is not optimized. For convenience, the sample signs release builds with
the debug key; your own app must use its own release signing key.

The model is downloaded into `app/src/main/assets` during the build (see
`app/download_model.gradle`).

In some cases, Google Play services may need to download the LiteRT module the
first time the app uses LiteRT, so the first initialization can take longer, or
fail if the device is offline. In that case the app shows an error.

## Test

The instrumentation tests in `app/src/androidTest` initialize LiteRT from Google
Play services and segment an image on the CPU and on the GPU (or on the CPU
fallback, on devices without a supported GPU). To run them on a connected device
or emulator with Google Play services:

```sh
./gradlew :app:connectedDebugAndroidTest
```
