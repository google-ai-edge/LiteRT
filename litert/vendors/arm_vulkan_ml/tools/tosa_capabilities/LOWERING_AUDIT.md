# TOSA lowering audit

66 specification operators; 29 referenced by production lowering.

Source references are not per-type qualification or device support. All device support remains query-dependent.
TFSC helper presence does not establish support for every type row.

| TOSA operator | Type rows | TFSC helper | Lowering source |
| --- | ---: | --- | --- |
| ARGMAX | 7 | present | Not referenced |
| AVG_POOL2D | 8 | present | compiler/operation.cc:554, compiler/operation.cc:639 |
| CONV2D | 8 | present | compiler/conv2d.cc:278, compiler/operation.cc:577 |
| CONV3D | 8 | present | Not referenced |
| DEPTHWISE_CONV2D | 8 | present | compiler/conv2d.cc:277, compiler/operation.cc:577 |
| FFT2D | 1 | present | Not referenced |
| MATMUL | 8 | present | compiler/operation.cc:667 |
| MAX_POOL2D | 7 | present | compiler/operation.cc:650 |
| RFFT2D | 1 | present | Not referenced |
| TRANSPOSE_CONV2D | 8 | present | compiler/extended_ops.cc:226 |
| CLAMP | 5 | present | compiler/conv2d.cc:290, compiler/extended_ops.cc:230, compiler/operation.cc:530, compiler/operation.cc:547, compiler/operation.cc:557, compiler/operation.cc:564, compiler/operation.cc:579, compiler/operation.cc:90 |
| ERF | 3 | present | Not referenced |
| SIGMOID | 3 | present | compiler/operation.cc:537 |
| TANH | 3 | present | Not referenced |
| ADD | 4 | present | compiler/operation.cc:525, compiler/operation.cc:527, compiler/operation.cc:529, compiler/operation.cc:545, compiler/operation.cc:563, compiler/operation.cc:629, compiler/operation.cc:670 |
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
| MUL | 6 | present | compiler/operation.cc:515, compiler/operation.cc:613 |
| POW | 3 | present | Not referenced |
| SUB | 4 | present | compiler/conv2d.cc:195, compiler/operation.cc:519, compiler/operation.cc:526, compiler/operation.cc:527 |
| TABLE | 2 | present | compiler/operation.cc:599, compiler/operation.cc:602, compiler/operation.cc:605 |
| ABS | 4 | present | Not referenced |
| BITWISE_NOT | 3 | present | Not referenced |
| CEIL | 3 | present | Not referenced |
| CLZ | 1 | present | Not referenced |
| COS | 3 | present | Not referenced |
| EXP | 3 | present | Not referenced |
| FLOOR | 3 | present | compiler/operation.cc:525, compiler/operation.cc:527 |
| LOG | 3 | present | Not referenced |
| LOGICAL_NOT | 1 | present | Not referenced |
| NEGATE | 6 | present | Not referenced |
| RECIPROCAL | 3 | present | Not referenced |
| RSQRT | 3 | present | Not referenced |
| SIN | 3 | present | Not referenced |
| SELECT | 7 | present | compiler/conv2d.cc:196, compiler/operation.cc:528 |
| EQUAL | 4 | present | Not referenced |
| GREATER | 4 | present | compiler/conv2d.cc:193 |
| GREATER_EQUAL | 4 | present | compiler/operation.cc:524 |
| REDUCE_ALL | 1 | present | Not referenced |
| REDUCE_ANY | 1 | present | Not referenced |
| REDUCE_MAX | 6 | present | compiler/extended_ops.cc:215 |
| REDUCE_MIN | 6 | present | compiler/extended_ops.cc:216 |
| REDUCE_PRODUCT | 3 | present | Not referenced |
| REDUCE_SUM | 4 | present | compiler/operation.cc:657 |
| CONCAT | 9 | present | compiler/conv2d.cc:158, compiler/conv2d.cc:247, compiler/conv2d.cc:248, compiler/operation.cc:595 |
| PAD | 9 | present | compiler/operation.cc:535, compiler/operation.cc:620 |
| RESHAPE | 9 | present | compiler/extended_ops.cc:194, compiler/operation.cc:95 |
| REVERSE | 9 | present | Not referenced |
| SLICE | 9 | present | compiler/conv2d.cc:152, compiler/conv2d.cc:240, compiler/conv2d.cc:257, compiler/operation.cc:569, compiler/operation.cc:594, compiler/operation.cc:634, compiler/operation.cc:647 |
| TILE | 9 | present | Not referenced |
| TRANSPOSE | 9 | present | compiler/operation.cc:534, compiler/operation.cc:618 |
| GATHER | 8 | present | compiler/extended_ops.cc:206 |
| SCATTER | 8 | present | Not referenced |
| RESIZE | 7 | present | compiler/extended_ops.cc:221 |
| CAST | 46 | present | compiler/operation.cc:518, compiler/operation.cc:531 |
| RESCALE | 12 | present | compiler/conv2d.cc:182, compiler/conv2d.cc:203, compiler/conv2d.cc:207, compiler/operation.cc:85 |

Source SHA-256 fingerprints:

- `compiler/compiler_plugin.cc`: `ca77012e020c0e6ea4e152057d7bca23d9e3d7cea6d6e278e0f5e8849319a3ec`
- `compiler/conv2d.cc`: `17cfa68e38a6a448809e892e006fca06559547169816d93631e4c423b0705724`
- `compiler/conv2d_adapter.cc`: `fedb0a2aa5a571b0101c304e2be5009afe4993368c8968da8e14675fc4922b16`
- `compiler/extended_ops.cc`: `cbd290c2d8f0cf95754aefdb15c8e192f27c05cfca5aa442dc04a5fd8058303f`
- `compiler/operation.cc`: `2c660bcccf1fa770a61bfabddd52a33ed8754cf38d87cfc62adde710bc481526`
- `compiler/partition.cc`: `bc0c72399b9073b2e10c6dbe12a372ec742dcc196c980b1f4c163491fde00ee8`
- `compiler/quantization.cc`: `4474eafbb74122343329b0b51372162d86cd5ea69ce24d4dbbab2cd9bf522da0`
- `include/tosa/OperatorEnum.hpp`: `cba5458b03154132be02eccbe29e349cad9f3c8453364ab2ec461d67b4dcc5be`
- `include/tosa/GraphOperatorHelpers.hpp`: `233b61fe60392b66d59a529afeb0b53842134cb3a885d578ec8b484afdc4d7c3`
