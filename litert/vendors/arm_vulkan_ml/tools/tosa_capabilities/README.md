# TOSA capability reference database

[`tosa_001000_1.json`](tosa_001000_1.json) is the structured specification layer of
the Arm Vulkan ML capability matrix: **66 operators and 350 type combinations**.
It is checked against [`tosa_capabilities.schema.json`](tosa_capabilities.schema.json).
It does not declare that LiteRT, TFSC, the emulator, or a GPU supports every row.

The database is a development and validation input, **not a runtime dependency**.
Neither the compiler plugin nor dispatch loads this JSON. Extraction, schema
validation, and the Python query tool run on the development host. The generated JSON keeps
the specification facts reviewable; its source pins, extractor and schema are
versioned with the backend. Regenerate it locally using the command below.
The large generated database and execution reports are not checked in.

Compiler legality checks enforce the constraints implemented for each lowering.
The compiled partition carries its required instruction set, profiles, and
extensions; dispatch must check those against public Vulkan device queries.
The reference database cannot establish device support. If compiler tables are
generated from it later, that generation belongs in the build/development
workflow, without a runtime JSON parser or registry checkout dependency.

## Sources and provenance

| Source | Pinned snapshot | What it supplies |
| --- | --- | --- |
| `/data/bt/os/SPIRV-Registry/extended/TOSA.001000.1.html` | `3b5c31fdd42ed166ef174bd7f1b3613672b6cc46` | Mapped operations/opcodes, SPIR-V types, operand order, constants, type combinations, profiles/extensions and enum values |
| Arm `tosa-specification`, tag `v1.0.1` | commit `7010bd19cb1e5468f5a60f578b961fc476bd6ae3` | Logical shapes, rank ranges, level bounds, base-profile requirements, numerical prose and validation expressions |
| Vulkan SPIR-V environment | VUIDs 09919, 09920, 09921 and 09923; checked October 26, 2026 | Manually recorded graph interface/constant shape and resource-description requirements |

The local TOSA source is available in
`/data/bt/os/tosa-for-spirv-codegen/external/tosa_specification`. Its working tree
currently uses a newer specification. The extractor reads the **pinned 1.0.1 Git
objects**, without switching that checkout. It also reads the pinned registry
objects, so local working edits do not silently change the database. Updating
either pin is an explicit operation followed by regeneration and review.

The primary operator list and type rows always come from the requested registry
checkout. TOSA is a supplement: operations or extensions present only in general
TOSA are not imported into the target instruction set. The generated file records
repository URLs, commit IDs, SHA-256 file hashes and per-operator references; it
contains no absolute checkout paths. Specification excerpts retain the source
license; see [the notice](tosa_001000_1.json.license) and
[TOSA's license](TOSA_SPECIFICATION_LICENSE.txt).

## Data contract

Each operator has argument declarations, complete registry type-table rows,
semantic constraints, and provenance. An argument's type comes from its row's
`types` entry, its `fixed_element_type`, or the row's `tosa_type_bindings` applied
to `element_type_parameter`/`tosa_type`. For example, CONCAT's variadic inputs
share the output's `in_out_t`; SELECT's condition is always Boolean and is not
repeated in the source type table. Constant attributes and fixed-type inputs are
kept in `arguments`, rather than duplicated across every row.

| Field | Meaning |
| --- | --- |
| `variants[].types` | Exact operand combinations from one table row; never a Cartesian product of independent type lists |
| `variants[].parameters` | Row-specific accumulator enum or TABLE_SIZE constraints |
| `variants[].requirements` | OR over `any_of`, AND within each `all_of`; extension requirements include an allowed base profile |
| `enums[].values[].additional_requirements` | Extra requirements imposed by an attribute value, ANDed with the row requirements |
| `arguments[].logical_shape` | TOSA dimensions and logical axes, scoped to that operator; distinct from Vulkan strides, tiling and format |
| `arguments[].rank_range`, `level_limits` | Symbolic rank bounds and level-dependent expressions, resolved against `levels` |
| `arguments[].constant_required` | Constant requirement from the SPIR-V mapping, including zero points and shape operands |
| `validation_expressions` | 155 indexed pseudocode assertions, with source lines and explicit dependence on their containing control flow/helpers |
| `semantics` | Operator prose and precision rules, preserved as AsciiDoc with source references |
| `coverage` | Separate specification-evaluator status and unassessed implementation/device coverage; see the source audit below |

`i48_t` remains a 48-bit TOSA semantic type even though its SPIR-V storage is
`OpTypeInt 64 0`. `PRO-INT, PRO-FP` permits either profile; a BF16-to-FP8 cast
requiring two extensions requires both. `DOUBLE_ROUND` on RESCALE adds
`EXT-DOUBLEROUND`, independent of its input/output type row.

This is a reference database, **not an executable legality checker**. Shape
equations and assertions are recorded as expressions, not evaluated as Python.
Assertions inside loops or guarded blocks retain their source context. Global
tensor/broadcast rules and pseudocode helpers remain normative. A missing rule,
unresolved expression or unreviewed mapping cannot imply support. In particular,
RESCALE's declaration encodes its multiplier in 32-bit SPIR-V elements while its
TOSA `scale32` description distinguishes 16/32-bit multiplier semantics; both
facts are retained rather than silently choosing an interpretation.

The database deliberately excludes `EXT-DYNAMIC`, `EXT-INT4`, `EXT-VARIABLE` and
`EXT-CONTROLFLOW`, which are absent from this SPIR-V mapping. Separately, Vulkan
requires fixed interface and constant shapes for each compiled graph pipeline.
Legal specification rows still need a correct LiteRT lowering, TFSC coverage,
and the selected device's profile, extension, level, tensor-limit and format
checks before delegation. Emulator execution reports are local development artifacts; they are not
proof of support on a native GPU.

## Use and reproduce

From the worktree root, create the host environment once (it can be shared with
the emulator test):

```bash
uv venv --python 3.13 .arm-vulkan-preflight/venv
uv pip install --python .arm-vulkan-preflight/venv/bin/python \
  --only-binary=:all: \
  -r litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/requirements.txt
```

Regenerate from the local sources, then validate:

```bash
.arm-vulkan-preflight/venv/bin/python \
  litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/extract_capabilities.py \
  --registry /data/bt/os/SPIRV-Registry \
  --tosa-spec /data/bt/os/tosa-for-spirv-codegen/external/tosa_specification

.arm-vulkan-preflight/venv/bin/python \
  litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/capabilities_test.py -v
```

Add `--check` to the extraction command to compare the generated bytes without
writing. There are 23 targeted checks, including schema validation, source pins,
completeness, shape/constant metadata, exact type combinations, profile OR/AND,
enum requirements, and rejection of unknown queries.

Query FP16 CONV2D with FP32 accumulation:

```bash
python3 litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/capability_query.py \
  --op CONV2D --type input=fp16_t --enum acc_type=FP32 \
  --capability PRO-FP
```

Use `--json` for the full operator record and filtered rows. Type filters name
columns in the registry's type table. Supplying capabilities filters only the
profile/extension conditions; it does not evaluate shapes, attribute legality,
level limits or hardware support. Unknown names are errors, and no matching row
is reported as zero candidates.

## Full lowering audit and opportunities

The specification database does not encode implementation status. Its
`litert_lowering` field is `not_assessed`, rather than the stale
`not_implemented` value used before the backend existed. The separate
[full source audit](LOWERING_AUDIT.md) covers every mapped operator and records
TFSC helper presence, production source references, and source fingerprints.
Neither a helper nor a source reference means that all type rows are supported.
Generate the joined machine-readable view with `--json`; no new large generated
JSON file is needed in source control:

```bash
python3 litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/audit_lowering.py \
  --tfsc /data/bt/os/tosa-for-spirv-codegen \
  --output litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/LOWERING_AUDIT.md
```

The registry pin was checked against upstream on October 27, 2026 and remains
current. Tensor API Revision 2 does not change this operator inventory.

| Opportunity | Current implementation / proposed use | Conditions before use |
| --- | --- | --- |
| CLAMP for INT8 RELU/RELU6 | Implemented when input/output quantization is identical | Emulator matched all 256 codes; TABLE retained when scale or zero point changes |
| RESCALE for INT8 QUANTIZE | Identity conversion eliminated; nonidentity conversions still use a 256-entry TABLE | Validate range, rounding and saturation for all 256 inputs; DOUBLE_ROUND additionally requires advertised EXT-DOUBLEROUND |
| RESCALE DOUBLE_ROUND | Current composed base-profile rounding | Emulator diagnostic passed ResNet18/MobileNetV2/DenseNet121 but advertises no extension support; do not bypass capability checks |
| AVG_POOL2D | Now direct for INT8 as well as FP32 | Existing static shape/quantization checks remain; accepted INT8 rounding differences are documented in the backend README |
| SIGMOID / TANH | FP32 SIGMOID already direct; INT8 sigmoid is TABLE; TANH not lowered | Floating-point rows do not authorize direct INT8 activation; use TABLE for quantized activation or a separately validated conversion path |
| MAXIMUM / MINIMUM / ABS / NEGATE / SUB | Available for new elementwise lowering | Match exact type row, broadcasting, quantization and integer overflow behavior |
| REDUCE_MAX / REDUCE_MIN / REDUCE_PRODUCT / REDUCE_ALL / REDUCE_ANY | INT8/FP32 MIN/MAX now implemented with constant axes and non-scalar outputs; SUM already used; PRODUCT/ALL/ANY remain future work | Check supported types, axes/keep-dims, accumulator semantics and Boolean types |
| RESIZE | Implemented for INT8/FP32 nearest and FP32 bilinear with constant NHWC sizes; aligned singleton sizes rejected | Match interpolation, coordinate transformation, border behavior and supported scale/offset representation |
| GATHER / SCATTER / TILE / REVERSE | GATHER implemented for INT8/FP32 values, constant INT32 indices and batch_dims=0; others remain future work | Check index types, negative-index behavior, axis normalization and static output shapes |
| TRANSPOSE_CONV2D / CONV3D | FP32 TRANSPOSE_CONV2D implemented for canonical SAME/VALID shapes with constant weights and optional bias; quantized transpose convolution and CONV3D remain future work | Validate layouts, padding/output shapes, accumulation and quantization; helper presence alone is insufficient |
| EXP / RECIPROCAL / RSQRT / LOG / POW | Building blocks for softmax, normalization and other composite functions | Check floating-point rows and numerical stability; these do not provide an atomic softmax/layer-normalization op |
| ARGMAX / FFT2D / RFFT2D / trigonometric and logical ops | Additional mapped coverage, outside current vision-classifier priorities | Add only for a concrete model need, with type/shape/device checks |

There is no dedicated HARD_SWISH, SILU, SOFTMAX, FULLY_CONNECTED, QUANTIZE,
DEQUANTIZE or normalization instruction in this mapping. Existing TABLE,
MATMUL, CAST and arithmetic sequences therefore cannot all be replaced by a
single same-named TOSA instruction. Unused mapped operators generally expand
model coverage; they do not automatically simplify currently supported graphs.

Device eligibility remains the conjunction of compiler legality, the queried
TOSA profile/extensions, tensor-format/size limits, and successful graph pipeline
creation. The current dispatch does not yet prevalidate every TOSA-level
constraint; this is an explicit gap, not a claim of universal shape support.
The database remains a development input, not a runtime dependency.
