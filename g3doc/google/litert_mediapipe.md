# Using LiteRT in MediaPipe

LiteRT can be selected as an `InferenceCalculator` option:

```c++
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<std::vector<Tensor>> input_tensors =
      graph.In(0).SetName("input_tensors").Cast<std::vector<Tensor>>();
  SidePacket<TfLiteModelPtr> model =
      graph.SideIn(0).SetName("model").Cast<TfLiteModelPtr>();

  auto& inference_node = graph.AddNode("InferenceCalculator");
  auto& inference_opts =
      inference_node.GetOptions<InferenceCalculatorOptions>();
  // Requesting LiteRT.
  inference_opts.mutable_delegate()->mutable_litert();
  input_tensors.ConnectTo(inference_node.In("TENSORS"));
  model.ConnectTo(inference_node.SideIn("MODEL"));
  Stream<std::vector<Tensor>> output_tensors =
      inference_node.Out("TENSORS").Cast<std::vector<Tensor>>();

  // Graph outputs.
  output_tensors.SetName("output_tensors").ConnectTo(graph.Out(0));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

This change requires that the target binary specifies a dependency on the
Inference Calculator LiteRT:

```python
cc_test(
    name = ...
    srcs = ...
    deps = [
      "//third_party/mediapipe/calculators/tensor:inference_calculator_litert"
    ],
)
```

Please see the `LiteRT` message in the [`inference_calculator.proto`](http://google3/third_party/mediapipe/calculators/tensor/inference_calculator.proto)
for all possible configuration options.

## Using LiteRT GPU acceleration

Through the Inference Calculator LiteRT we can also run models on GPU. To enable
GPU acceleration, configure the delegate in `InferenceCalculatorOptions`:

```c++
InferenceCalculatorOptions::Delegate::LiteRt* litert =
  options->mutable_delegate()->mutable_litert();
litert->mutable_gpu(); // Uses default GPU options (AUTOMATIC backend, FP16 precision)
```

Or via `.pbtxt` config:

```proto
node {
  calculator: "InferenceCalculator"
  ...
  options {
    [drishti.InferenceCalculatorOptions.ext] {
      delegate {
        litert {
          gpu {
            # Optional: choose backend, default is AUTOMATIC.
            # Options: AUTOMATIC, OPENGL, OPENCL, WEBGPU.
            backend: OPENGL
            # Optional: choose precision, default is FP16.
            # Options: DEFAULT, FP16, FP32.
            precision: FP16
          }
        }
      }
    }
  }
}
```

### GPU Dependency Requirements

Using GPU acceleration requires linking a LiteRT GPU accelerator in your target
binary.

*   **Static Linking** (recommended for C++ binaries): Add the static library
    target to your `deps`:

    ```python
    deps = [
        "//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator",
    ]
    ```

*   **Dynamic Linking** (common for Android apps to reduce base APK size): Add
    the shared library target to your binary `deps` and ensure it is loaded:

    ```python
    deps = [
        "//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator_shared_lib",
    ]
    ```

    And load the library in Java/Kotlin:

    ```java
    System.loadLibrary("LiteRtClGlAccelerator");
    ```

## Using LiteRT NPU acceleration

Through the Inference Calculator LiteRT we can also run models on an NPU.
Assuming we have ahead-of-time compiled the model for a specific NPU, then we
need to set the so-called 'dispatch library path' to the folder that contains
the `libLiteRtDispatch_<npu_vendor>.so` file of the NPU that we want to target:

```c++
InferenceCalculatorOptions::Delegate::LiteRt* litert =
  options->mutable_delegate()->mutable_litert();
  litert->mutable_npu()->set_dispatch_library_path("/path/to/folder/containing/dispatch_lib");
```

Note that if you define your config via a .pbtxt file you'd specify it like so:

```
...
  options: {
    [drishti.InferenceCalculatorOptions.ext] {
      delegate {
        litert {
          npu {
            dispatch_library_path : "/path/to/folder/containing/dispatch_lib"
          }
        }
      }
    }
  }
...
```

> [!TIP] The `dispatch_library_path` can be omitted from the options if you
> configure the `LiteRtService` in your MediaPipe graph. If available, the
> calculator will automatically retrieve the path from the service.
>
> In Android, you can configure the service with the app's native library
> directory (where packaged `.so` files are extracted):
>
> ```java
> String nativeLibPath = this.getApplicationInfo().nativeLibraryDir;
> if (nativeLibPath != null) {
>   processor.setServiceObject(
>       new LiteRtService(),
>       LiteRtConfig.builder().setDispatchLibraryPath(nativeLibPath).build());
> }
> ```
>
> See
> [MainActivity.java](http://google3/third_party/mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic/MainActivity.java#L144-L149)
> for a complete example.

See also
[Selfie Segmentation NPU config](http://google3/third_party/mediapipe/modules/selfie_segmentation/selfie_segmentation_litertnpu.pbtxt).

We'd need to ensure that we package the `libLiteRtDispatch_<npu_vendor>.so` into
this folder before running the app. The vendor's NPU shared library also depends
on LiteRT's C API shared library file, so we need to ensure to package that too.

For a simple command-line program this could look like so:

```
DEVICE_BIN_PATH=/data/local/tmp
blaze build --config=android_arm64 //litert/c:litert_runtime_c_api_so
adb push --sync "blaze-bin/litert/c/libLiteRt.so" "${DEVICE_BIN_PATH}"
blaze --blazerc=/dev/null build -c opt --config=android_arm64 litert/vendors/mediatek/dispatch:dispatch_api_so
adb push blaze-bin/litert/vendors/mediatek/dispatch/libLiteRtDispatch_Mediatek.so ${DEVICE_BIN_PATH}
```

A production app would probably copy the shared library into a folder within the
APK. This can be achieved by specifying the library targets in the `deps`
attribute of the `android_binary`.

For **Google Tensor (Pixel TPU)**:

*   **Static Linking** (recommended for C++ binaries):

    ```python
    deps = [
        "//litert/vendors/google_tensor/dispatch:dispatch_api_static",
    ]
    ```

*   **Dynamic Linking** (for Android apps):

    ```python
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/vendors/google_tensor/google:dispatch_api_so",
    ]
    ```

For **MediaTek NPU**:

```python
deps = [
    "//litert/c:litert_runtime_c_api_shared_lib",
    "//litert/vendors/mediatek/google:dispatch_api_so",
]
```

For **Qualcomm NPU**:

```python
deps = [
  "//litert/c:litert_runtime_c_api_shared_lib",
  "//litert/vendors/qualcomm/google:dispatch_api_so",
  # Include the QNN runtime libraries for the target HTP version(s) of the devices you want to support:
  "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v73", # Snapdragon 8 Gen 2
  "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v75", # Snapdragon 8 Gen 3
  "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v79", # Snapdragon 8 Elite
# (Optional) "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v69" for Snapdragon 8 Gen 1
]
```

And then load these libraries from the app:

*   Java (Google Tensor):
    [MainActivity.java](http://google3/third_party/mediapipe/examples/android/src/java/com/google/mediapipe/apps/selfiesegmentationlitertdarwinn/MainActivity.java#L21-L24)
*   Java (Qualcomm):
    [MainActivity.java](http://google3/third_party/mediapipe/examples/android/src/java/com/google/mediapipe/apps/selfiesegmentationlitertnpu/MainActivity.java#L21-L24)
*   Kotlin:
    [FastVlmViewModel.kt](http://google3/third_party/ai_edge_gallery/Android/src/app/src/main/java/com/google/ai/edge/gallery/customtasks/fastvlm/FastVlmViewModel.kt#L59)
