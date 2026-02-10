# Changes since release 2.1.0

<!---
INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES
-->

### Breaking Changes

<!---
* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API, ABI AND BEHAVIORAL BREAKING CHANGES>
-->

### Known Caveats

<!---
* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM, SHOULD GO HERE>
-->

### Major Features and Improvements

<!---
* <IF RELEASE CONTAINS MULTIPLE FEATURES FROM SAME AREA, GROUP THEM TOGETHER>
-->

### Bug Fixes and Other Changes

<!---
* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>
-->

* Removed methods from `litert::Event` which uses C type `LiteRtEnvironment`.
  All C++ API should uses C++ `litert::Environment` instead.
  Also removed method `CreateFromSyncFenceFd()` that doesn't accept
  `litert::Environment`.

* Updated `litert::Event::Type()` to return C++ types instead of C types.

* The LiteRT headers no longer define the following OpenCL type names in the
  global namespace when OpenCL is _not_ supported: `cl_mem`, `cl_event`.
  These have been replaced with the type aliases `LiteRtClMem` and
  `LiteRtClEvent`, defined in a new header `litert/c/litert_opencl_types.h`.
  All of these symbols that didn't include the `LiteRt` prefix in their name
  were never intended to be part of the LiteRT API, and their presence
  in the global namespace risked conflicts with header files from other
  packages.

* Likewise, and for the same reason, the LiteRT headers no longer define the
  following WebGPU type names in the global namespace when WebGPU _is_
  supported: `struct WGPUBufferImpl`, `WGPUBuffer`. These have been
  replaced with the type alias `LiteRtWGPUBuffer` which is defined in a
  new header file `litert/c/litert_webgpu_types.h`. Alternatively, apps
  using these symbols can get them from WebGPU's `webgpu.h` header file.

# Release 2.1.0

**Release 2.1.0 is the LiteRT** beta release.

LiteRT APIs are stable and have achieved feature parity. This milestone marks a significant step forward, introducing full feature parity with TensorFlow Lite, stable LiteRT APIs, and critical performance enhancements for GPU and NPU acceleration. With this release, we are officially recommending that developers begin their transition to LiteRT.

## Major Features and Improvements

### LiteRT Runtime

* Custom ops are supported through [custom op dispatcher](https://github.com/google-ai-edge/LiteRT/blob/main/g3doc/apis/Custom_Op_Dispatcher.md).
* CMake Build is supported in addition to Bazel
* Released LiteRT C++ SDK using prebuilt libLiteRt.so file
* Added Profiler API in CompiledModel
* Added ErrorReporter API in CompiledModel
* Added ResizeInputTensor API in CompiledModel

### LiteRT NPU

* Introduced LiteRT Accelerator Test Suite for coverage and regression testing
* Introduced LiteRT graph transformation APIs for compiler plugins
* Qualcomm
  * Added support for Qualcomm Snapdragon Gen5
  * Added support for NPU JIT mode
  * LiteRT Op coverage improvements
* MediaTek
  * Added support for NPU JIT mode
  * LiteRT Op coverage improvements

### LiteRT GPU

* Increased GPU coverage with WebGPU/Dawn and OpenCL including Android, Linux, MacOS, Windows, iOS, IoT devices
* Added asynchronous execution to Metal, WebGPU backends
* Improved performance and memory footprint
* Added an option to control GPU inference priority
* Better error handling (without crashing) on Delegation errors

### LLM Support

* Provided Desktop GPU backends prebuilt for Linux (x64, arm64), MacOS (arm64), Windows (x64)
* Improved memory utilization when executing on GPUs
* Published new LLMs on [https://huggingface.co/litert-community](https://huggingface.co/litert-community)
  * litert-community/FastVLM-0.5B
  * litert-community/Qwen3-0.6B
  * litert-community/embeddinggemma-300m with new NPU precompiled models
  * litert-community/gemma-3-270m-it with new NPU precompiled model
* Published Function Gemma on [https://huggingface.co/google](https://huggingface.co/google)
  * google/functiongemma-270m-it

### LiteRT on Android

* Added Interpreter API (CPU only) in the Maven v2.1.0+ packages
* Added [Instruction](https://ai.google.dev/edge/litert/next/android_cpp_sdk) to use pre-built CompiledModel C++ API from the Maven package.

## Bug Fixes and Other Changes

Fixes Android min SDK version and it’s 23 now.

LiteRT NPU: Fixes partition algorithm when the full model cannot be offloaded to NPU.

## Breaking Changes

* Removed direct C headers usage. Users no longer need to include C headers.
* TensorBuffer::CreateManaged() requires Environment always.
* All TensorBuffer creation requires Environment except HostMemory types.
* LiteRT C++ constructors are hidden. All LiteRT C++ objects should be created by Create() methods.
* Moved internal only C++ APIs(such as litert\_logging.h) to litert/cc/internal
* Removed Tensor, Subgraph, Signature access from litert::Model. Instead users can access SimpleTensor, SimpleSignature.
* The CompiledModel::Create() API no longer needs litert::Model. They can be created from filename, model buffers directly.
* Users can access SimpleTensor and SimpleSignature from CompiledModel.
* Annotation, Metrics APIs are removed from CompiledModel.
* Removed individual OpaqueOptions creation. These OpaqueOptions objects are obtained by Options directly.
  *  Options::GetCpuOptions()
  *  Options::GetGpuOptions()
  *  Options::GetRuntimeOptions()
  * …

# Release 2.0.2a1

## LiteRT

### Breaking Changes

* `com.google.ai.edge.litert.TensorBufferRequirements`
  * It becomes a data class, so all fields could be accessed directly without getter methods.
  * The type of field `strides` changes from `IntArry` to `List<Int>` to be immutable.
* `com.google.ai.edge.litert.Layout`
  * The type of field `dimensions` and `strides` changes from `IntArry` to `List<Int>` to be immutable.
* Rename GPU option `NoImmutableExternalTensorsMode` to `NoExternalTensorsMode`

### Known Caveats

### Major Features and Improvements

* Added Profiler API in Compiled Model: [source](https://github.com/google-ai-edge/LiteRT/blob/main/litert/cc/litert_profiler.h).
* Added Error reporter API in Compiled Model: [source](https://github.com/google-ai-edge/LiteRT/blob/d65ffb98ce708a7fb40640546af0c3a6f0f8a763/litert/cc/options/litert_runtime_options.h#L44).
* Added resize input tensor API in Compiled Model: [source](https://github.com/google-ai-edge/LiteRT/blob/main/litert/cc/litert_compiled_model.h#L431).

* [tflite] Add error detection in TfLiteRegistration::init(). When a Delegate
kernel returns `TfLiteKernelInitFailed()`, it is treated
as a critical failure on Delegate. This error will be detected in
SubGraph::ReplaceNodeSubsetsWithDelegateKernels() will cause
Delegate::Prepare() to fail, ultimately leading
InterpreterBuilder::operator() or Interpreter::ModifyGraphWithDelegate() to
return an error.

#### LiteRT GPU Accelerator

* Added WebGPU support with GPU Accelerator.
* Added an option to control GPU inference priority.

#### LiteRT API Refactoring

* Introduced target `litert/cc:litert_api_with_dynamic_runtime`
  This is a convenience Bazel target contains LiteRt C++ and C APIs. Users
  of this library are responsible to bundle LiteRT C API Runtime
  `libLiteRt.so`.
* C++ APIs that need LiteRT C API Runtime are moved to
  litert/cc/dynamic_runtime/
  Note: This is for internal usage. If you want to use dynamic API, use
  `litert/cc:litert_api_with_dynamic_runtime`.
* All static C++ APIs are moved to litert/cc/
* You shouldn't mix static API targets with dynamic API targets.

### Bug Fixes and Other Changes

* The Android `minSdkVersion` has increased to 23.
* Update tests to provide `kLiteRtHwAcceleratorNpu` for fully AOT compiled
models.
