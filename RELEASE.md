# Release 1.4.0

## LiteRT Next

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

### Bug Fixes and Other Changes

* Update tests to provide `kLiteRtHwAcceleratorNpu` for fully AOT compiled
models.
<!---
* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>
-->
