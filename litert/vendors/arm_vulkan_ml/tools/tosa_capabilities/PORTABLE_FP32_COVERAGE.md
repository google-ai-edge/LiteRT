# Portable FP32 lowering qualification

This matrix describes implemented subsets, not all type combinations from the
specification. Static, dense, non-scalar FP32 tensors are required, with ranks
and dimensions bounded by the backend. Constant INT32 controls are permitted.
The reference database remains a development input, never runtime JSON.

| Operators | Implemented subset | Qualification |
| --- | --- | --- |
| RELU, RELU6, ABS, NEG, SQUARE, SQRT, RSQRT, EXP, LOG, FLOOR, CEIL, TANH | FP32 unary; finite numerical test outputs, appropriate mathematical domains | SPIR-V + emulator |
| SUB, DIV, MAXIMUM, MINIMUM | Static broadcasting; SUB/DIV NONE/RELU/RELU6 activation | SPIR-V + emulator |
| SLICE, PADV2 | Constant bounds/padding/fill; positive output dimensions | SPIR-V + emulator |
| CONCATENATION, PACK | Compatible shapes; constant axis; CONCAT has no fused activation | SPIR-V + emulator |
| SQUEEZE, EXPAND_DIMS, TILE | Static shape changes and constant controls; non-scalar outputs | SPIR-V + emulator |
| UNPACK, SPLIT, SPLIT_V | Multiple outputs, constant axis/sizes, one inferred SPLIT_V extent | SPIR-V + emulator |
| STRIDED_SLICE | Positive strides, begin/end/shrink masks; no ellipsis/new-axis/offset | SPIR-V + emulator |
| MEAN, REDUCE_PROD | Constant general axes, negative axis normalization, both keep-dims modes, non-scalar output | SPIR-V + emulator |
| SOFTMAX | Last-axis normalization, finite positive beta, stable max-subtraction composition | SPIR-V + emulator |
| BATCH_MATMUL | Static batch broadcasting, both transpose flags, bounded explicit broadcast materialization | SPIR-V + emulator |
| MAX_POOL_2D, AVERAGE_POOL_2D | Static NHWC VALID/SAME, cropped unused trailing input, NONE/RELU/RELU6 | SPIR-V + emulator |

`qualify_portable_ops.py` covers 39 positive fixtures with four tensor cases each,
plus four explicit rejection fixtures. Every positive fixture uses the developer
tool for inspection, SPIR-V export/validation, capability matching, CPU comparison
and both trace modes. Generated models and reports are not checked in.

No new runtime INT32/BOOL interfaces, quantized variants, dynamic shapes, FFT or
CONV3D are enabled. Existing quantized lowerings remain separate. Engine profiles,
extensions and tensor limits are checked at runtime, and pipeline creation must
succeed. Missing advertisements remain blocking even under emulation opt-in.
Native GPU numerical/performance qualification is outstanding; emulator success
does not establish hardware conformance. Nonfinite outputs fail the development
tool's numerical gate. The known DenseNet/MobileNetV3 quantized pooling failures
are not waived by this coverage expansion.
