# TensorFlow Dependencies Analysis

## 1. .bzl files loaded from @org_tensorflow

The following `.bzl` files are loaded from `@org_tensorflow` in the `WORKSPACE` or `BUILD`/`.bzl` files:

*   `tensorflow:workspace0.bzl` (WORKSPACE)
*   `tensorflow:workspace1.bzl` (WORKSPACE)
*   `tensorflow:workspace2.bzl` (WORKSPACE)
*   `tensorflow:workspace3.bzl` (WORKSPACE)
*   `tensorflow:tensorflow.bzl` (and `tensorflow.default.bzl`)
    *   Used in: `tflite/testing/tflite_model_test.bzl`, `tflite/build_def.bzl`, `tflite/experimental/acceleration/mini_benchmark/special_rules.bzl`, `tflite/testing/build_def.bzl`, `tflite/testing/kernel_test/BUILD`, `tflite/testing/BUILD`, `tflite/internal/BUILD`, `tflite/java/jni/BUILD`, `tflite/tools/BUILD`, `tflite/delegates/flex/BUILD`, etc.
*   `tensorflow:strict.default.bzl`
    *   Used in: `tflite/build_def.bzl`, `tflite/testing/BUILD`, `tflite/g3doc/tools/BUILD`, `tflite/tools/signature/BUILD`, etc.
*   `tensorflow:pytype.default.bzl`
    *   Used in: `tflite/g3doc/tools/BUILD`, `tflite/tools/optimize/debugging/python/BUILD`
*   `tensorflow/java:build_defs.bzl`
    *   Used in: `tflite/java/ovic/BUILD`, `tflite/java/BUILD`
*   `third_party/protobuf/bazel/common:proto_info.bzl`
    *   Used in: `tflite/tools/benchmark/experimental/delegate_performance/android/proto.bzl`
*   `tensorflow/core/platform:build_config_root.bzl`
    *   Used in: `tflite/testing/BUILD`, `tflite/experimental/acceleration/mini_benchmark/metrics/BUILD`
*   `tensorflow/core/platform:build_config.bzl`
    *   Used in: `tflite/delegates/gpu/common/transformations/BUILD`, `tflite/delegates/gpu/common/BUILD`

## 2. External Repositories used in LiteRT

The following external repositories, likely defined in `tf_workspace*`, are referenced in LiteRT `BUILD` files:

*   `@com_google_absl`
*   `@com_google_googletest`
*   `@com_googlesource_code_re2`
*   `@tsl`
*   `@xla`
*   `@ml_dtypes_py`
*   `@eigen_archive`
*   `@org_tensorflow` (The workspace itself)

## 3. BUILD file dependencies on @org_tensorflow//...

The following is a list of all dependencies on `@org_tensorflow` targets found in `BUILD` and `.bzl` files (excluding `third_party`, `g3doc`, and `.git`):

*   `@org_tensorflow//:LICENSE`
*   `@org_tensorflow//:requirements_lock_3_10.txt`
*   `@org_tensorflow//:requirements_lock_3_11.txt`
*   `@org_tensorflow//:requirements_lock_3_12.txt`
*   `@org_tensorflow//:requirements_lock_3_13.txt`
*   `@org_tensorflow//:WORKSPACE`
*   `@org_tensorflow//tensorflow:android`
*   `@org_tensorflow//tensorflow:android_arm`
*   `@org_tensorflow//tensorflow:android_arm64`
*   `@org_tensorflow//tensorflow:android_armeabi`
*   `@org_tensorflow//tensorflow:arm_any`
*   `@org_tensorflow//tensorflow/c:c_api_internal`
*   `@org_tensorflow//tensorflow/c:kernels`
*   `@org_tensorflow//tensorflow/c:tf_datatype`
*   `@org_tensorflow//tensorflow/c:tf_status_headers`
*   `@org_tensorflow//tensorflow/c:tf_tensor_internal`
*   `@org_tensorflow//tensorflow/cc/saved_model:loader`
*   `@org_tensorflow//tensorflow/cc/saved_model:signature_constants`
*   `@org_tensorflow//tensorflow:chromiumos`
*   `@org_tensorflow//tensorflow:chromiumos_x86_64`
*   `@org_tensorflow//tensorflow/compiler/mlir:init_mlir`
*   `@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper`
*   `@org_tensorflow//tensorflow/compiler/mlir:passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/common:func`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/common/ir:QuantOps`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/common:uniform_quantized_types`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo/cc:context`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto_py`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantize_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization:__subpackages__`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:representative_dataset`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__`
*   `@org_tensorflow//tensorflow/compiler/mlir:register_common_dialects`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:fold_broadcast_pass`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:fuse_convolution_pass`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:legalize_tf`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:rename_entrypoint_to_main`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:__subpackages__`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:tf_stablehlo`
*   `@org_tensorflow//tensorflow/compiler/mlir/stablehlo:unfuse_batch_norm_pass`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:cluster_util`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_attr`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_tensor`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:file_tf_mlir_translate`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:import_utils`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:location_utils`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mangling_util`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_import_options`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_traits`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:lower_tf_lib`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_optimize_td_files`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_device_pass_inc_gen`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_graph_optimization_pass`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:unroll_batch_matmul_pass`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow/translate/tools:parsers`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_utils`
*   `@org_tensorflow//tensorflow/compiler/mlir/tensorflow:verification_utils`
*   `@org_tensorflow//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor`
*   `@org_tensorflow//tensorflow/compiler/mlir/tf2xla:compile_mlir_util`
*   `@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes`
*   `@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf`
*   `@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf_with_tf2xla`
*   `@org_tensorflow//tensorflow/compiler/mlir/tools/optimize:quantization_utils`
*   `@org_tensorflow//tensorflow/compiler/tf2xla/kernels:xla_ops`
*   `@org_tensorflow//tensorflow/core/common_runtime/eager:context`
*   `@org_tensorflow//tensorflow/core/common_runtime/eager:core_no_xla`
*   `@org_tensorflow//tensorflow/core:core_cpu`
*   `@org_tensorflow//tensorflow/core:core_cpu_base`
*   `@org_tensorflow//tensorflow/core:core_cpu_lib`
*   `@org_tensorflow//tensorflow/core/example:example_protos_cc_impl`
*   `@org_tensorflow//tensorflow/core/example:feature_util`
*   `@org_tensorflow//tensorflow/core:feature_util`
*   `@org_tensorflow//tensorflow/core:framework`
*   `@org_tensorflow//tensorflow/core:framework:fake_input`
*   `@org_tensorflow//tensorflow/core:framework_internal`
*   `@org_tensorflow//tensorflow/core:framework_lite`
*   `@org_tensorflow//tensorflow/core:framework:tensor_testutil`
*   `@org_tensorflow//tensorflow/core:graph`
*   `@org_tensorflow//tensorflow/core:image_testdata`
*   `@org_tensorflow//tensorflow/core/ir/types:Dialect`
*   `@org_tensorflow//tensorflow/core:jpeg_internal`
*   `@org_tensorflow//tensorflow/core/kernels:ops_testutil`
*   `@org_tensorflow//tensorflow/core/kernels:portable_all_ops_textual_hdrs`
*   `@org_tensorflow//tensorflow/core/kernels:portable_core_ops`
*   `@org_tensorflow//tensorflow/core/kernels:portable_extended_ops`
*   `@org_tensorflow//tensorflow/core/kernels:tensor_list`
*   `@org_tensorflow//tensorflow/core:lib`
*   `@org_tensorflow//tensorflow/core:lib_internal`
*   `@org_tensorflow//tensorflow/core/lib/jxl:jxl_io`
*   `@org_tensorflow//tensorflow/core/lib/math:math_util`
*   `@org_tensorflow//tensorflow/core/lib/png:png_io`
*   `@org_tensorflow//tensorflow/core:lib_proto_parsing`
*   `@org_tensorflow//tensorflow/core/lib/webp:webp_io`
*   `@org_tensorflow//tensorflow/core:ops`
*   `@org_tensorflow//tensorflow/core:__pkg__`
*   `@org_tensorflow//tensorflow/core/platform`
*   `@org_tensorflow//tensorflow/core/platform:build_config.bzl`
*   `@org_tensorflow//tensorflow/core/platform:build_config_root.bzl`
*   `@org_tensorflow//tensorflow/core/platform:errors`
*   `@org_tensorflow//tensorflow/core/platform:hash`
*   `@org_tensorflow//tensorflow/core/platform:jpeg`
*   `@org_tensorflow//tensorflow/core/platform:logging`
*   `@org_tensorflow//tensorflow/core/platform:mutex`
*   `@org_tensorflow//tensorflow/core/platform:platform_port`
*   `@org_tensorflow//tensorflow/core/platform:protobuf`
*   `@org_tensorflow//tensorflow/core/platform:resource_loader`
*   `@org_tensorflow//tensorflow/core/platform:rules_cc.bzl`
*   `@org_tensorflow//tensorflow/core/platform:status`
*   `@org_tensorflow//tensorflow/core/platform:statusor`
*   `@org_tensorflow//tensorflow/core/platform:strong_hash`
*   `@org_tensorflow//tensorflow/core/platform:thread_annotations`
*   `@org_tensorflow//tensorflow/core/platform:tstring`
*   `@org_tensorflow//tensorflow/core:portable_gif_internal`
*   `@org_tensorflow//tensorflow/core:portable_jpeg_internal`
*   `@org_tensorflow//tensorflow/core:portable_op_registrations_and_gradients`
*   `@org_tensorflow//tensorflow/core:portable_tensorflow_lib`
*   `@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite`
*   `@org_tensorflow//tensorflow/core:portable_tensorflow_test_lib`
*   `@org_tensorflow//tensorflow/core/protobuf:error_codes_proto_impl_cc`
*   `@org_tensorflow//tensorflow/core/protobuf:for_core_protos_cc`
*   `@org_tensorflow//tensorflow/core:protos_all_cc`
*   `@org_tensorflow//tensorflow/core:protos_all_cc_impl`
*   `@org_tensorflow//tensorflow/core:protos_all_py`
*   `@org_tensorflow//tensorflow/core/public:release_version`
*   `@org_tensorflow//tensorflow/core:session_options`
*   `@org_tensorflow//tensorflow/core:tensorflow`
*   `@org_tensorflow//tensorflow/core:test`
*   `@org_tensorflow//tensorflow/core:testlib`
*   `@org_tensorflow//tensorflow/core:test_main`
*   `@org_tensorflow//tensorflow/core:tflite_portable_logging`
*   `@org_tensorflow//tensorflow/core/tfrt/fallback:op_kernel_runner`
*   `@org_tensorflow//tensorflow/core/util:stats_calculator_portable`
*   `@org_tensorflow//tensorflow/c:tf_datatype`
*   `@org_tensorflow//tensorflow/c:tf_status_headers`
*   `@org_tensorflow//tensorflow/c:tf_tensor_internal`
*   `@org_tensorflow//tensorflow:debug`
*   `@org_tensorflow//tensorflow:disable_tf_lite_py`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:fuchsia`
*   `@org_tensorflow//tensorflow:internal`
*   `@org_tensorflow//tensorflow:ios`
*   `@org_tensorflow//tensorflow:ios_x86_64`
*   `@org_tensorflow//tensorflow/java:build_defs.bzl`
*   `@org_tensorflow//tensorflow:libtensorflow_framework.`
*   `@org_tensorflow//tensorflow:libtensorflow_framework.so.`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:linux_aarch64`
*   `@org_tensorflow//tensorflow:linux_ppc64le`
*   `@org_tensorflow//tensorflow:linux_s390x`
*   `@org_tensorflow//tensorflow:linux_x86_64`
*   `@org_tensorflow//tensorflow:linux_x86_64_no_sse`
*   `@org_tensorflow//tensorflow:macos`
*   `@org_tensorflow//tensorflow:macos_arm64`
*   `@org_tensorflow//tensorflow:optimized`
*   `@org_tensorflow//tensorflow:__pkg__`
*   `@org_tensorflow//tensorflow/python/...`
*   `@org_tensorflow//tensorflow/python/client:session`
*   `@org_tensorflow//tensorflow/python/eager:context`
*   `@org_tensorflow//tensorflow/python/eager:def_function`
*   `@org_tensorflow//tensorflow/python/eager:function`
*   `@org_tensorflow//tensorflow/python/eager:monitoring`
*   `@org_tensorflow//tensorflow/python/framework:byte_swap_tensor`
*   `@org_tensorflow//tensorflow/python/framework:constant_op`
*   `@org_tensorflow//tensorflow/python/framework:convert_to_constants`
*   `@org_tensorflow//tensorflow/python/framework:dtypes`
*   `@org_tensorflow//tensorflow/python/framework:error_interpolation`
*   `@org_tensorflow//tensorflow/python/framework:errors`
*   `@org_tensorflow//tensorflow/python/framework:for_generated_wrappers`
*   `@org_tensorflow//tensorflow/python/framework:graph_util`
*   `@org_tensorflow//tensorflow/python/framework:importer`
*   `@org_tensorflow//tensorflow/python/framework:load_library`
*   `@org_tensorflow//tensorflow/python/framework:ops`
*   `@org_tensorflow//tensorflow/python/framework:tensor_shape`
*   `@org_tensorflow//tensorflow/python/framework:tensor_spec`
*   `@org_tensorflow//tensorflow/python/framework:tensor_util`
*   `@org_tensorflow//tensorflow/python/framework:test_lib`
*   `@org_tensorflow//tensorflow/python/framework:versions`
*   `@org_tensorflow//tensorflow/python/grappler:tf_optimizer`
*   `@org_tensorflow//tensorflow/python/layers`
*   `@org_tensorflow//tensorflow/python/lib/core:ndarray_tensor`
*   `@org_tensorflow//tensorflow/python/lib/core:pybind11_lib`
*   `@org_tensorflow//tensorflow/python/lib/core:py_func_lib`
*   `@org_tensorflow//tensorflow/python/lib/io:file_io`
*   `@org_tensorflow//tensorflow/python/ops:array_ops`
*   `@org_tensorflow//tensorflow/python/ops:array_ops_stack`
*   `@org_tensorflow//tensorflow/python/ops:control_flow_ops`
*   `@org_tensorflow//tensorflow/python/ops:image_ops`
*   `@org_tensorflow//tensorflow/python/ops:io_ops`
*   `@org_tensorflow//tensorflow/python/ops:linalg_ops`
*   `@org_tensorflow//tensorflow/python/ops:list_ops`
*   `@org_tensorflow//tensorflow/python/ops:logging_ops`
*   `@org_tensorflow//tensorflow/python/ops:losses`
*   `@org_tensorflow//tensorflow/python/ops:map_ops`
*   `@org_tensorflow//tensorflow/python/ops:math_ops`
*   `@org_tensorflow//tensorflow/python/ops:nn_ops`
*   `@org_tensorflow//tensorflow/python/ops:ragged:ragged_tensor`
*   `@org_tensorflow//tensorflow/python/ops:random_ops`
*   `@org_tensorflow//tensorflow/python/ops:rnn`
*   `@org_tensorflow//tensorflow/python/ops/signal:window_ops`
*   `@org_tensorflow//tensorflow/python/ops:string_ops`
*   `@org_tensorflow//tensorflow/python/ops:variables`
*   `@org_tensorflow//tensorflow/python/ops:variable_scope`
*   `@org_tensorflow//tensorflow/python/ops:while_loop`
*   `@org_tensorflow//tensorflow/python/platform:client_testlib`
*   `@org_tensorflow//tensorflow/python/platform:gfile`
*   `@org_tensorflow//tensorflow/python/platform:resource_loader`
*   `@org_tensorflow//tensorflow/python/platform:test`
*   `@org_tensorflow//tensorflow/python/platform:tf_logging`
*   `@org_tensorflow//tensorflow/python:_pywrap_tensorflow`
*   `@org_tensorflow//tensorflow/python:pywrap_tensorflow`
*   `@org_tensorflow//tensorflow/python:pywrap_tfe`
*   `@org_tensorflow//tensorflow/python:_pywrap_toco_api`
*   `@org_tensorflow//tensorflow/python/saved_model`
*   `@org_tensorflow//tensorflow/python/saved_model:constants`
*   `@org_tensorflow//tensorflow/python/saved_model:load`
*   `@org_tensorflow//tensorflow/python/saved_model:loader`
*   `@org_tensorflow//tensorflow/python/saved_model:save`
*   `@org_tensorflow//tensorflow/python/saved_model:save_options`
*   `@org_tensorflow//tensorflow/python/saved_model:signature_constants`
*   `@org_tensorflow//tensorflow/python/saved_model:tag_constants`
*   `@org_tensorflow//tensorflow/python:__subpackages__`
*   `@org_tensorflow//tensorflow/python:tf2`
*   `@org_tensorflow//tensorflow/python/tools:print_selective_registration_header`
*   `@org_tensorflow//tensorflow/python/trackable:autotrackable`
*   `@org_tensorflow//tensorflow/python/training:saver`
*   `@org_tensorflow//tensorflow/python/training:training_util`
*   `@org_tensorflow//tensorflow/python/util:all_util`
*   `@org_tensorflow//tensorflow/python/util:compat`
*   `@org_tensorflow//tensorflow/python/util:deprecation`
*   `@org_tensorflow//tensorflow/python/util:dispatch`
*   `@org_tensorflow//tensorflow/python/util:keras_deps`
*   `@org_tensorflow//tensorflow/python/util:lazy_loader`
*   `@org_tensorflow//tensorflow/python/util:nest`
*   `@org_tensorflow//tensorflow/python/util:tf_export`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`
*   `@org_tensorflow//tensorflow:strict.default.bzl`
*   `@org_tensorflow//tensorflow:__subpackages__`
*   `@org_tensorflow//tensorflow:tensorflow.bzl`
*   `@org_tensorflow//tensorflow:tensorflow_bzl`
*   `@org_tensorflow//tensorflow:tensorflow.default.bzl`
*   `@org_tensorflow//tensorflow:tensorflow_py`
*   `@org_tensorflow//tensorflow:tensorflow_py_no_contrib`
*   `@org_tensorflow//tensorflow/tools/toolchains/android:arm64-v8a`
*   `@org_tensorflow//tensorflow/tools/toolchains/android:armeabi-v7a`
*   `@org_tensorflow//tensorflow/tools/toolchains/android:x86`
*   `@org_tensorflow//tensorflow/tools/toolchains/android:x86_64`
*   `@org_tensorflow//tensorflow/tools/toolchains/ios:ios_armv7`
*   `@org_tensorflow//tensorflow/tools/toolchains/linux:linux_aarch64`
*   `@org_tensorflow//tensorflow/tools/toolchains/linux:linux_armhf`
*   `@org_tensorflow//tensorflow:windows`
*   `@org_tensorflow//tensorflow:workspace`
*   `@org_tensorflow//tensorflow:workspace0.bzl`
*   `@org_tensorflow//tensorflow:workspace1.bzl`
*   `@org_tensorflow//tensorflow:workspace2.bzl`
*   `@org_tensorflow//tensorflow:workspace3.bzl`
*   `@org_tensorflow//third_party/absl/...`
*   `@org_tensorflow//third_party/bazel_platforms/cpu:aarch64`
*   `@org_tensorflow//third_party/bazel_platforms/cpu:armv7`
*   `@org_tensorflow//third_party/bazel_platforms/cpu:x86_64`
*   `@org_tensorflow//third_party/bazel_platforms/os:linux`
*   `@org_tensorflow//third_party/deepmind/lyria_live/internal/odml:__subpackages__`
*   `@org_tensorflow//third_party/eigen3:LICENSE`
*   `@org_tensorflow//third_party/fft2d:fft2d_headers`
*   `@org_tensorflow//third_party/flatcc`
*   `@org_tensorflow//third_party/flatcc:runtime`
*   `@org_tensorflow//third_party/GL:EGL_headers`
*   `@org_tensorflow//third_party/GL:GLES3_headers`
*   `@org_tensorflow//third_party/hexagon_nn_skel:libhexagon_nn_skel`
*   `@org_tensorflow//third_party/icu/data:conversion_data`
*   `@org_tensorflow//third_party/java/android/android_sdk_linux/extras/android/compatibility/multidex`
*   `@org_tensorflow//third_party/java/mockito`
*   `@org_tensorflow//third_party/odml/infra/genai/conversion:__subpackages__`
*   `@org_tensorflow//third_party/odml/infra/genai/inference/executor/google_tensor:__subpackages__`
*   `@org_tensorflow//third_party/odml/litert/litert/python:__subpackages__`
*   `@org_tensorflow//third_party/odml/litert/litert:__subpackages__`
*   `@org_tensorflow//third_party/odml/litert:__subpackages__`
*   `@org_tensorflow//third_party/odml/model_customization/quantization:__subpackages__`
*   `@org_tensorflow//third_party/opencl_icd_loader`
*   `@org_tensorflow//third_party/protobuf/bazel/common:proto_info.bzl`
*   `@org_tensorflow//third_party/py/ai_edge_torch:__subpackages__`
*   `@org_tensorflow//third_party/py/litert_torch:__subpackages__`
*   `@org_tensorflow//third_party/py/numpy`
*   `@org_tensorflow//third_party/py/numpy:headers`
*   `@org_tensorflow//third_party/py/numpy:numpy`
*   `@org_tensorflow//third_party/py/PIL:pil`
*   `@org_tensorflow//third_party/py/tensorflow`
*   `@org_tensorflow//third_party/py/tensorflow_addons`
*   `@org_tensorflow//third_party/py/tensorflow_federated:__subpackages__`
*   `@org_tensorflow//third_party/tflite_micro:__subpackages__`
*   `@org_tensorflow//third_party/unzip`

## Analysis of Workspace Files

*   `workspace0.bzl`: Defines models (`inception_v1`, etc.), `rules_proto`, and `grpc` binds.
*   `workspace1.bzl`: Defines `bazel_toolchains`, `local_config_android`. Loads `grpc_deps`, `benchmark_deps`, `closure_repositories`, `rules_pkg_dependencies`.
*   `workspace2.bzl`: Extensive list of third-party dependencies (`absl`, `eigen3`, `flatbuffers`, `gemmlowp`, `ruy`, `stablehlo`, etc.) via `_initialize_third_party` and `_tf_repositories`.
*   `workspace3.bzl`: Defines `local_xla`, `local_tsl`, `io_bazel_rules_closure`, `rules_pkg`, `rules_jvm_external`, `llvm-project`.

## 4. Client Targets and their TensorFlow Dependencies

### `//ci/tools/python/vendor_sdk:<file_scope>`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//ci/tools/python/wheel/utils:<file_scope>`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//ci/tools/python/wheel:tflite_converter_protos`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/ats:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/build_common:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/build_common:linux_x86_64_grte`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/build_common:linux_x86_64_ungrte`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/c/internal:litert_logging`
*   `@org_tensorflow//tensorflow:android`

### `//litert/c/options:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/c:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/cc/dynamic_runtime/options:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/cc/dynamic_runtime/options:litert_compiler_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_cpu_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_darwinn_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_google_tensor_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_gpu_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_intel_openvino_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_mediatek_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_qualcomm_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime/options:litert_runtime_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/cc/dynamic_runtime:litert_builder`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_compiled_model`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_compiled_model_gpu_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc/dynamic_runtime:litert_custom_op_kernel`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_environment`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_environment_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_environment_options_test`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_environment_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc/dynamic_runtime:litert_event`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_extended_model`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_model`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_op_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_opaque_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_options`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_profiler`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_tensor_buffer`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_tensor_buffer_requirements`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/dynamic_runtime:litert_tensor_buffer_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc/dynamic_runtime:litert_tensor_buffer_types`
*   `@org_tensorflow//tensorflow:emscripten`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/cc/internal:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/cc/options:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/cc:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/cc:litert_any_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc:litert_compiled_model_google_tensor_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc:litert_compiled_model_gpu_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc:litert_compiled_model_jetgpu_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc:litert_environment_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/cc:litert_tensor_buffer_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/compiler/mlir/dialects/litert:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/compiler/mlir:converter_api_core`
*   `@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc`

### `//litert/compiler/plugin:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/compiler:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/core/cache:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/core/model:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/core/util:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/core:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/core:dynamic_loading`
*   `@org_tensorflow//tensorflow:windows`

### `//litert/experimental/mlir:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/experimental:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/integration_test/models:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/integration_test:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/kotlin:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/python/aot/ai_pack:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot/core:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot/core:apply_plugin`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/python/aot/vendors/example:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot/vendors/google_tensor:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot/vendors/mediatek:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot/vendors/qualcomm:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot/vendors:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/aot:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/internal:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/litert_wrapper/common:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:strict.default.bzl`
*   `@org_tensorflow//tensorflow:tensorflow.default.bzl`

### `//litert/python/litert_wrapper/compiled_model_wrapper:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:strict.default.bzl`
*   `@org_tensorflow//tensorflow:tensorflow.default.bzl`

### `//litert/python/litert_wrapper/tensor_buffer_wrapper:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:strict.default.bzl`
*   `@org_tensorflow//tensorflow:tensorflow.default.bzl`

### `//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer`
*   `@org_tensorflow//third_party/py/numpy:headers`

### `//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer_test`
*   `@org_tensorflow//third_party/py/numpy:headers`

### `//litert/python/litert_wrapper:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:strict.default.bzl`

### `//litert/python/litert_wrapper:litert_wrapper_test`
*   `@org_tensorflow//tensorflow/python/platform:resource_loader`
*   `@org_tensorflow//third_party/py/numpy:headers`

### `//litert/python/mlir/_mlir_libs:<file_scope>`
*   `@org_tensorflow//tensorflow:tensorflow.default.bzl`

### `//litert/python/tools/model_utils/test:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/tools/model_utils/test:match_test`
*   `@org_tensorflow//third_party/py/numpy:headers`

### `//litert/python/tools/model_utils/test:numpy_to_elements_attr_test`
*   `@org_tensorflow//third_party/py/numpy:headers`

### `//litert/python/tools/model_utils:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`

### `//litert/python/tools/model_utils:model_utils`
*   `@org_tensorflow//third_party/py/numpy:headers`

### `//litert/python/tools:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:pytype.default.bzl`
*   `@org_tensorflow//tensorflow:strict.default.bzl`

### `//litert/python:<file_scope>`
*   `@org_tensorflow//tensorflow:license`
*   `@org_tensorflow//tensorflow:strict.default.bzl`
*   `@org_tensorflow//tensorflow:tensorflow.default.bzl`

### `//litert/runtime/accelerators/dispatch:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/runtime/accelerators/xnnpack:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/runtime/accelerators:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/runtime/compiler:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/runtime/dispatch:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/runtime:gl_buffer_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/runtime:tensor_buffer`
*   `@org_tensorflow//tensorflow:android`

### `//litert/runtime:tensor_buffer_registry_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/samples/semantic_similarity:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/sdk_util:check_sdk_deps_test`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/sdk_util:mtk_sdk_v7_android`
*   `@org_tensorflow//tensorflow:android`

### `//litert/sdk_util:mtk_sdk_v7_host`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/sdk_util:mtk_sdk_v8_android`
*   `@org_tensorflow//tensorflow:android`

### `//litert/sdk_util:mtk_sdk_v8_host`
*   `@org_tensorflow//tensorflow:linux_x86_64`

### `//litert/test/generators:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/test:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/tools/culprit_finder:culprit_finder_lib`
*   `@org_tensorflow//tensorflow:ios`

### `//litert/tools/flags/vendors:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/tools/flags:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/tools:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/tools:npu_numerics_check`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/c:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/cc:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/examples:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/examples:example_transformations_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/google_tensor/compiler:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/google_tensor/compiler:compiler_plugin_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/google_tensor/dispatch:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/google_tensor:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/google_tensor:adapter`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/intel_openvino/compiler:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/intel_openvino/dispatch:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/mediatek/compiler/legalizations:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/mediatek/compiler:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/mediatek/compiler:compile_model`
*   `@org_tensorflow//tensorflow:android`
*   `@org_tensorflow//tensorflow:chromiumos`

### `//litert/vendors/mediatek/compiler:compiler_plugin_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/mediatek/dispatch:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/mediatek/schema:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/mediatek:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/compiler:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/core/backends:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/builders:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/dump:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/schema:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/transformation:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/utils:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/utils:log`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/core/wrappers/tests:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core/wrappers:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/core:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/dispatch:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/qnn_backend_test/builder_test:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/qnn_backend_test/builder_test:elementwise_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/qnn_backend_test/builder_test:fully_connected_int2_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/qnn_backend_test/builder_test:relu_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/qnn_backend_test/builder_test:topk_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/qnn_backend_test:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm/qnn_backend_test:qnn_model_test`
*   `@org_tensorflow//tensorflow:android`

### `//litert/vendors/qualcomm/tools:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm:<file_scope>`
*   `@org_tensorflow//tensorflow:license`

### `//litert/vendors/qualcomm:qnn_manager_test`
*   `@org_tensorflow//tensorflow:android`

