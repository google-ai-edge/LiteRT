<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Arm Vulkan ML integration testing

The supported developer commands are the [Arm diagnostic CLI](../devtool/README.md)
and the [shared vision evaluator](../../../../tools/vision_eval/README.md).

## Real LiteRT model lifecycle

`//litert/vendors/arm_vulkan_ml/tools/testing:lifecycle_test` generates tiny
TFLite models and exercises the real compiler plugin, dispatch, and public
LiteRT compiled-model API (`CompiledModel` + `HwAccelerators::kNpu`). It checks:

* Complete NPU acceleration and resource ownership across repeated and
  simultaneous `CompiledModel` instances.
* Quantized `SOFTMAX` execution through the JIT compiler.
* Scalar and `COMPLEX64` buffer layouts reaching the dispatch API.

Build and run the test on an ML-capable Vulkan device (opt in with
`LITERT_ARM_VULKAN_ML_RUN_INTEGRATION_TESTS=1`; without opt-in the test skips):

```sh
blaze test -c opt \
  --test_env=LITERT_ARM_VULKAN_ML_RUN_INTEGRATION_TESTS=1 \
  //litert/vendors/arm_vulkan_ml/tools/testing:lifecycle_test
```

For end-to-end vision dataset evaluation and smoke checks, use
[`litert_vision_eval`](../../../../tools/vision_eval/README.md).
