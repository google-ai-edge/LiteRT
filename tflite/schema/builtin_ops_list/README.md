# Builtin Ops List Generator.

This directory contains a code generator to generate a pure C header for
builtin ops lists.

Whenever you add a new builtin op, please execute:

```sh
bazel run \
  //tflite/schema/builtin_ops_header:generate > \
  tflite/builtin_ops.h &&
bazel run \
  //tflite/schema/builtin_ops_list:generate > \
  tflite/kernels/builtin_ops_list.inc
```
