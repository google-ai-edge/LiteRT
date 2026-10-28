# TOSA lowering audit

66 specification operators; 35 referenced by production lowering.

Source references are not per-type qualification or device support. All device support remains query-dependent.
TFSC helper presence does not establish support for every type row.

| TOSA operator | Type rows | TFSC helper | Lowering source |
| --- | ---: | --- | --- |
| ARGMAX | 7 | present | Not referenced |
| AVG_POOL2D | 8 | present | compiler/operation.cc:562, compiler/operation.cc:647, compiler/portable_ops.cc:214 |
| CONV2D | 8 | present | compiler/conv2d.cc:278, compiler/operation.cc:585 |
| CONV3D | 8 | present | Not referenced |
| DEPTHWISE_CONV2D | 8 | present | compiler/conv2d.cc:277, compiler/operation.cc:585 |
| FFT2D | 1 | present | Not referenced |
| MATMUL | 8 | present | compiler/extended_ops.cc:267, compiler/operation.cc:675, compiler/portable_ops.cc:207 |
| MAX_POOL2D | 7 | present | compiler/operation.cc:658, compiler/portable_ops.cc:213 |
| RFFT2D | 1 | present | Not referenced |
| TRANSPOSE_CONV2D | 8 | present | compiler/extended_ops.cc:278 |
| CLAMP | 5 | present | compiler/conv2d.cc:290, compiler/extended_ops.cc:272, compiler/extended_ops.cc:282, compiler/operation.cc:538, compiler/operation.cc:555, compiler/operation.cc:565, compiler/operation.cc:572, compiler/operation.cc:587, compiler/operation.cc:91, compiler/portable_ops.cc:199 |
| ERF | 3 | present | Not referenced |
| SIGMOID | 3 | present | compiler/operation.cc:545 |
| TANH | 3 | present | Not referenced |
| ADD | 4 | present | compiler/extended_ops.cc:269, compiler/operation.cc:533, compiler/operation.cc:535, compiler/operation.cc:537, compiler/operation.cc:553, compiler/operation.cc:571, compiler/operation.cc:637, compiler/operation.cc:678 |
| ARITHMETIC_RIGHT_SHIFT | 3 | present | compiler/conv2d.cc:197 |
| BITWISE_AND | 3 | present | Not referenced |
| BITWISE_OR | 3 | present | Not referenced |
| BITWISE_XOR | 3 | present | Not referenced |
| INTDIV | 1 | present | Not referenced |
| LOGICAL_AND | 1 | present | Not referenced |
| LOGICAL_LEFT_SHIFT | 3 | present | Not referenced |
| LOGICAL_RIGHT_SHIFT | 3 | present | Not referenced |
| LOGICAL_OR | 1 | present | Not referenced |
| LOGICAL_XOR | 1 | present | Not referenced |
| MAXIMUM | 4 | present | Not referenced |
| MINIMUM | 4 | present | Not referenced |
| MUL | 6 | present | compiler/operation.cc:523, compiler/operation.cc:621, compiler/portable_ops.cc:225, compiler/portable_ops.cc:235, compiler/portable_ops.cc:249, compiler/portable_ops.cc:253, compiler/portable_ops.cc:257 |
| POW | 3 | present | Not referenced |
| SUB | 4 | present | compiler/conv2d.cc:195, compiler/operation.cc:527, compiler/operation.cc:534, compiler/operation.cc:535, compiler/portable_ops.cc:255 |
| TABLE | 2 | present | compiler/operation.cc:607, compiler/operation.cc:610, compiler/operation.cc:613 |
| ABS | 4 | present | Not referenced |
| BITWISE_NOT | 3 | present | Not referenced |
| CEIL | 3 | present | Not referenced |
| CLZ | 1 | present | Not referenced |
| COS | 3 | present | Not referenced |
| EXP | 3 | present | compiler/portable_ops.cc:255 |
| FLOOR | 3 | present | compiler/operation.cc:533, compiler/operation.cc:535 |
| LOG | 3 | present | Not referenced |
| LOGICAL_NOT | 1 | present | Not referenced |
| NEGATE | 6 | present | compiler/portable_ops.cc:233 |
| RECIPROCAL | 3 | present | compiler/portable_ops.cc:225, compiler/portable_ops.cc:234, compiler/portable_ops.cc:257 |
| RSQRT | 3 | present | compiler/portable_ops.cc:234 |
| SIN | 3 | present | Not referenced |
| SELECT | 7 | present | compiler/conv2d.cc:196, compiler/operation.cc:536 |
| EQUAL | 4 | present | Not referenced |
| GREATER | 4 | present | compiler/conv2d.cc:193 |
| GREATER_EQUAL | 4 | present | compiler/operation.cc:532 |
| REDUCE_ALL | 1 | present | Not referenced |
| REDUCE_ANY | 1 | present | Not referenced |
| REDUCE_MAX | 6 | present | compiler/extended_ops.cc:251, compiler/portable_ops.cc:254 |
| REDUCE_MIN | 6 | present | compiler/extended_ops.cc:252 |
| REDUCE_PRODUCT | 3 | present | compiler/portable_ops.cc:248 |
| REDUCE_SUM | 4 | present | compiler/extended_ops.cc:249, compiler/operation.cc:665, compiler/portable_ops.cc:248, compiler/portable_ops.cc:256 |
| CONCAT | 9 | present | compiler/conv2d.cc:158, compiler/conv2d.cc:247, compiler/conv2d.cc:248, compiler/operation.cc:603, compiler/portable_ops.cc:219, compiler/portable_ops.cc:242 |
| PAD | 9 | present | compiler/operation.cc:543, compiler/operation.cc:628, compiler/portable_ops.cc:239 |
| RESHAPE | 9 | present | compiler/extended_ops.cc:228, compiler/operation.cc:96, compiler/portable_ops.cc:196 |
| REVERSE | 9 | present | Not referenced |
| SLICE | 9 | present | compiler/conv2d.cc:152, compiler/conv2d.cc:240, compiler/conv2d.cc:257, compiler/operation.cc:577, compiler/operation.cc:602, compiler/operation.cc:642, compiler/operation.cc:655, compiler/portable_ops.cc:212, compiler/portable_ops.cc:219, compiler/portable_ops.cc:238, compiler/portable_ops.cc:245 |
| TILE | 9 | present | compiler/portable_ops.cc:205, compiler/portable_ops.cc:237 |
| TRANSPOSE | 9 | present | compiler/extended_ops.cc:264, compiler/operation.cc:542, compiler/operation.cc:626, compiler/portable_ops.cc:205 |
| GATHER | 8 | present | compiler/extended_ops.cc:240 |
| SCATTER | 8 | present | Not referenced |
| RESIZE | 7 | present | compiler/extended_ops.cc:257 |
| CAST | 46 | present | compiler/operation.cc:526, compiler/operation.cc:539 |
| RESCALE | 12 | present | compiler/conv2d.cc:182, compiler/conv2d.cc:203, compiler/conv2d.cc:207, compiler/operation.cc:86 |

Source SHA-256 fingerprints:

- `compiler/compiler_plugin.cc`: `ca77012e020c0e6ea4e152057d7bca23d9e3d7cea6d6e278e0f5e8849319a3ec`
- `compiler/conv2d.cc`: `17cfa68e38a6a448809e892e006fca06559547169816d93631e4c423b0705724`
- `compiler/conv2d_adapter.cc`: `fedb0a2aa5a571b0101c304e2be5009afe4993368c8968da8e14675fc4922b16`
- `compiler/extended_ops.cc`: `d3567fe248517d66b4c8b919a57319851485422b99559809a2f93481bc9cbd7c`
- `compiler/operation.cc`: `508022980027b064dffb32acc56635719ed348b1da3b90e3d3562e8c9579cbd4`
- `compiler/partition.cc`: `39c4ca39b0a34f37148bb0d08e9f0e7ca725be7291c6a7747d75740f2fe3e54d`
- `compiler/portable_ops.cc`: `4bec45efbb04cd7a1cc58c0f4bb915c39b49f38bb93cc2eab1d24b32d9fc6147`
- `compiler/quantization.cc`: `4474eafbb74122343329b0b51372162d86cd5ea69ce24d4dbbab2cd9bf522da0`
- `include/tosa/OperatorEnum.hpp`: `cba5458b03154132be02eccbe29e349cad9f3c8453364ab2ec461d67b4dcc5be`
- `include/tosa/GraphOperatorHelpers.hpp`: `233b61fe60392b66d59a529afeb0b53842134cb3a885d578ec8b484afdc4d7c3`
