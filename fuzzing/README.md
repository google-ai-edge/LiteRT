# LiteRT kernel fuzzers

This is a standalone CMake project for developing LiteRT/TFLite kernel
fuzzers. It does not modify LiteRT's main CMake target graph. The default
FuzzTest path is `/data/bt/os/fuzztest`; override it when using another local
checkout.

Configure with Clang and run the PAD fuzzer:

```bash
cmake -S fuzzing -B /tmp/litert-fuzz-build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DTENSORFLOW_SOURCE_DIR=/path/to/tensorflow \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build /tmp/litert-fuzz-build \
  --target tflite_pad_fuzz_test -j

ctest --test-dir /tmp/litert-fuzz-build \
  -R tflite_pad_fuzz_test --output-on-failure
```

Disable sanitizers for a faster local smoke build with
`-DLITERT_FUZZ_SANITIZERS=OFF`. Sanitizers should remain enabled for fuzzing.

For an interactive run:

```bash
/tmp/litert-fuzz-build/tflite_pad_fuzz_test \
  --fuzz=PadFuzzTest.PadNeverCrashes --fuzz_duration=60s
```

The initial sanitizer smoke run found an out-of-bounds read in TFLite's PAD
resize path for a scalar input paired with a rank-one padding tensor. This is
an intended security-fuzzing failure and should be triaged with sanitizers
enabled.

Please note we have assumptions that: 1. all input buffers are properly aligned according to their data type.  2. all Boolean input tensors have valid elements that are either 0 or 1. They won't contain other values.
