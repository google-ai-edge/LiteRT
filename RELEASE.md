# Release 2.0.2a1

## LiteRT

<!---
INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES
-->

### Major Features and Improvements

<!---
INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES
-->

### Breaking Changes

<!---
* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API, ABI AND BEHAVIORAL BREAKING CHANGES>
-->

* `com.google.ai.edge.litert.TensorBufferRequirements`
  * It becomes a data class, so all fields could be accessed directly without getter methods.
  * The type of field `strides` changes from `IntArry` to `List<Int>` to be immutable.
* `com.google.ai.edge.litert.Layout`
  * The type of field `dimensions` and `strides` changes from `IntArry` to `List<Int>` to be immutable.
* Rename GPU option `NoImmutableExternalTensorsMode` to `NoExternalTensorsMode`

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

* [tflite] Add error detection in TfLiteRegistration::init(). When a Delegate
kernel returns `TfLiteKernelInitFailed()`, it is treated
as a critical failure on Delegate. This error will be detected in
SubGraph::ReplaceNodeSubsetsWithDelegateKernels() will cause
Delegate::Prepare() to fail, ultimately leading
InterpreterBuilder::operator() or Interpreter::ModifyGraphWithDelegate() to
return an error.
* Added Profiler API in Compiled Model: [source](https://github.com/google-ai-edge/LiteRT/blob/main/litert/cc/litert_profiler.h).
* Added Error reporter API in Compiled Model: [source](https://github.com/google-ai-edge/LiteRT/blob/d65ffb98ce708a7fb40640546af0c3a6f0f8a763/litert/cc/options/litert_runtime_options.h#L44).
* Added resize input tensor API in Compiled Model: [source](https://github.com/google-ai-edge/LiteRT/blob/main/litert/cc/litert_compiled_model.h#L431).

### Bug Fixes and Other Changes

* The Android `minSdkVersion` has increased to 23.
* Update tests to provide `kLiteRtHwAcceleratorNpu` for fully AOT compiled
models.
<!---
* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>
-->
