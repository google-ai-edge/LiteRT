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
| Vulkan SPIR-V environment | Selected VUIDs 09919, 09920, 09921 and 09923; not exhaustive. Full inventory uses pinned Vulkan-Headers 1.4.349 validusage.json | Manually recorded graph interface/constant shape and resource-description requirements |

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
license; see [TOSA's license](TOSA_SPECIFICATION_LICENSE.txt).

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

The C++ developer tool can query this database directly on Linux and Android.
Generate the JSON on the host, then deploy it with `litert_arm_vulkan_ml_tool`;
no Python or specification checkout is needed on the phone. See the
[query usage guide](../devtool/README.md#query-tosa-specification-candidates-on-linux-or-android).
The database is an explicit diagnostic input, not an inference runtime dependency.

Query FP16 CONV2D with FP32 accumulation:

```bash
litert_arm_vulkan_ml_tool tosa-query --database tosa_001000_1.json \
  --op CONV2D --type input=fp16_t --enum acc_type=FP32 --capability PRO-FP
```

The Python query remains available as the host reference:

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
generated full source audit (`audit_lowering.py`) covers every mapped operator and records
TFSC helper presence, production source references, and source fingerprints.
Here, "full" means every mapped specification operator is listed; it does not
mean every type combination is implemented or qualified. The source audit uses
text matching and may miss indirect helper calls. Neither helper presence nor a
source reference establishes complete lowering or device support.

The current production inventory is **60/66 emitted TOSA operators**. Run
`litert_arm_vulkan_ml_tool coverage [--json]` for the reviewed implementation
manifest, compiled into the C++ executable. The source audit now recognizes
portable unary macros and enum aliases. Verify inventory consistency with
`audit_lowering.py --tfsc /path/to/tosa-for-spirv-codegen --check`.
Emission, SPIR-V validation, emulator qualification and native-device qualification
are separate fields. The six remaining operators are FFT2D, CLZ, BITWISE_AND,
BITWISE_OR, LOGICAL_LEFT_SHIFT and RFFT2D; see the backend README for the frontend
and extension constraints. New TFLite-reachable subsets and their fixtures are
recorded in `implementation_coverage.json`; the specification database itself
continues to describe specification legality only.

The table below separates implemented subsets, proposed optimizations and
remaining coverage gaps. It is an engineering inventory, not a prioritized or
committed TODO list. The [portable FP32 matrix](PORTABLE_FP32_COVERAGE.md) records
implemented subsets and their SPIR-V/emulator qualification evidence. Native GPU
qualification remains separate.
Generate the joined machine-readable view with `--json`; no new large generated
JSON file is needed in source control:

```bash
python3 litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/audit_lowering.py \
  --tfsc /data/bt/os/tosa-for-spirv-codegen \
  --output .arm-vulkan-preflight/lowering-audit.md
```

The registry pin was last checked against upstream on October 27, 2026. Tensor API Revision 2 does not change this operator inventory.

| Area | Status | Implemented subset or remaining work |
| --- | --- | --- |
| INT8 RELU/RELU6 | Implemented subset | Direct CLAMP when quantization is unchanged; TABLE when scale or zero point changes. All 256 input codes were checked on the emulator. |
| INT8 QUANTIZE | Implemented subset; proposed optimization | Identity conversion is eliminated; other conversions use TABLE. Replacing TABLE with RESCALE remains a proposal requiring range, rounding and saturation checks for all 256 codes. |
| RESCALE DOUBLE_ROUND | Capability-dependent proposal | Production uses composed base-profile rounding. The emulator diagnostic does not establish advertised EXT-DOUBLEROUND support; unadvertised extensions remain rejected. |
| AVG_POOL2D / MAX_POOL2D | Implemented subsets; numerical qualification gap | FP32 VALID/SAME pooling is implemented with fused activation handling. INT8 pooling retains its existing geometry/quantization restrictions. INT8 AVG_POOL2D rounding differences and the known DenseNet121/MobileNetV3 Large failures remain visible; see the backend README. |
| SIGMOID / TANH | Implemented FP32; partial INT8 | FP32 LOGISTIC emits SIGMOID, and FP32 TANH is implemented. INT8 LOGISTIC uses TABLE; INT8 TANH now uses TABLE with standard output quantization. Floating-point type rows do not authorize direct INT8 activation. |
| MAXIMUM / MINIMUM / ABS / NEGATE / SUB / DIV | Implemented FP32/INT8 subsets | Unary paths and static binary broadcasting are implemented; SUB/DIV support NONE/RELU/RELU6 fused activation. INT8 SUB, ABS and matching-quantization MAXIMUM/MINIMUM now have separately qualified subsets; DIV and affine NEG remain unsupported. |
| Reductions | Implemented subsets; remaining coverage | SUM and MIN/MAX support existing INT8/FP32 subsets. General-axis MEAN and REDUCE_PROD are implemented for FP32, including keep-dims modes and non-scalar outputs. BOOL REDUCE_ALL/ANY are implemented with constant axes. Runtime axes and scalar outputs remain unsupported. |
| RESIZE | Implemented subsets; remaining coverage | INT8/FP32 nearest and bilinear use constant NHWC sizes. Quantized bilinear now uses Q10 integer coordinates and signed rounding; aligned singleton sizes remain unsupported; coordinate, border and scale/offset restrictions still apply. |
| GATHER / TILE / SCATTER / REVERSE | Implemented subsets; remaining coverage | GATHER supports INT8/FP32 values with constant in-range INT32 indices and batch_dims=0. FP32/INT8 TILE and REVERSE use constant controls. FP32 SCATTER_ND supports constant unique indices. Runtime GATHER indices remain unsupported. |
| Shape manipulation and partition outputs | Implemented FP32/INT8 subsets | SQUEEZE, EXPAND_DIMS, PACK/UNPACK, SPLIT/SPLIT_V and positive-stride STRIDED_SLICE support FP32 and matching-quantization INT8, including multiple results. Controls are constant; scalar interfaces and unsupported slice masks/negative strides are rejected. See the FP32 matrix for exact restrictions. |
| TRANSPOSE_CONV2D / CONV3D | Implemented subset; remaining coverage | FP32 transpose convolution supports canonical SAME/VALID geometry, constant weights and optional bias. Quantized transpose convolution now supports symmetric constant weights, optional matching INT32 bias and proven rescaling. CONV3D has an FP32 subset; quantized CONV3D remains unsupported. |
| MATMUL-based lowering | Implemented subsets | FP32 BATCH_MATMUL supports static batch broadcasting and transpose flags with bounded materialization. FULLY_CONNECTED uses existing INT8 or FP32 paths; FP32 lowers to MATMUL plus bias and optional CLAMP. |
| EXP / RECIPROCAL / RSQRT / LOG / SOFTMAX | Implemented FP32 subsets/compositions | FP32 EXP, RSQRT and LOG are lowered; RECIPROCAL is used internally for DIV, SQRT and SOFTMAX. FP32 SOFTMAX uses stable last-axis max subtraction and finite positive beta. This does not imply a standalone LiteRT RECIPROCAL op or general normalization support. |
| POW / ARGMAX / FFT and logical ops | Implemented subsets; remaining coverage | FP32 POW/SIN/COS, exact GELU via ERF, FP32-to-INT32 ARGMAX and BOOL logical operations have qualified subsets. FFT2D, CLZ, BITWISE_AND/OR and LOGICAL_LEFT_SHIFT have no selected TFLite route; RFFT2D remains pending complex-intermediate partition handling and EXT-FFT qualification. |

There is no dedicated HARD_SWISH, SILU, SOFTMAX, FULLY_CONNECTED, QUANTIZE,
DEQUANTIZE or normalization instruction in this mapping. Existing TABLE,
MATMUL, CAST and arithmetic sequences therefore cannot all be replaced by a
single same-named TOSA instruction. In particular, implemented SOFTMAX and
FULLY_CONNECTED lowerings are compositions. The full FP32 MobileNetV2 graph
provides model-level evidence for its FULLY_CONNECTED path.
Unused mapped operators generally expand
model coverage; they do not automatically simplify currently supported graphs.

Device eligibility remains the conjunction of compiler legality, the queried
TOSA profile/extensions/level, tensor-format/size limits, and successful graph
pipeline creation. The compiler collects level requirements from final TFSC
instructions: rank, tensor byte size (including intermediates), effective
dilated kernel and padding extents, strides, resize scale ratios, tensor-list
lengths and the single flat graph nesting depth. Dispatch compares those values
with the selected engine's `8K` or `none` bounds; unknown levels and unchecked
artifacts are rejected. These checks do not replace operator shape/type legality
analysis or pipeline acceptance.
The database remains a development input, not a runtime dependency.

## Vulkan valid-usage audit

`vuid_audit.py` inventories the Vulkan commands and recursively referenced
structures used by dispatch, capability discovery and validation capture. It
also includes RuntimeSpirv rules for applicability review. Its source is the
`registry/validusage.json` and `vk.xml` from the pinned Vulkan-Headers v1.4.349
archive (the valid-usage metadata identifies Vulkan-Docs commit
`ab80b9e8dd1c08b14c9536c37a31182114ae72ee`).

`vuid_review.json` records explicitly reviewed exclusions and enforcement
arguments, bound to the registry hash and backend source hashes. It is a partial
review, **not a complete compliance declaration**. Unreviewed requirements,
stale source hashes, stale VUIDs, duplicate classifications and missing evidence
fail the audit. Tests of the audit gate are in `vuid_audit_test.py`.

```sh
python litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/vuid_audit.py \
  --registry .bazel-output/external/arm_vulkan_ml_dep_vulkan_headers/registry \
  --output .arm-vulkan-preflight/vuid-inventory.json --check
```

The complete applicable-coverage milestone remains open while this command
fails. Candidate rules for nested structures are deliberately included even
when their arrays are unused; exclusions require an explicit construction
argument. Numerical success or a validation-layer run does not automatically
classify a VUID as covered. Full resource-failure injection and complete
applicability review are still outstanding. TOSA-level constraints are enforced
against emitted graph requirements as described above.

### Five individually tested dispatch requirements

The C++ target
`//litert/vendors/arm_vulkan_ml/dispatch:vulkan_invocation_test` covers 04101,
09921, 09923, 09796 and 09951. It compiles tiny models through the production
compiler plugin, injects an instance-procedure resolver into production dispatch,
and substitutes typed fake Vulkan functions. No loader, GPU or emulator is
required. Allocations use synthetic handles and small host vectors.

The independent SPIR-V decoder is a test oracle for these single-entry-point
fixtures, not a general SPIR-V validator. Captured pipeline descriptions and
constant data are deep-copied while production pointers are valid. FP32 fixtures
have two constants, two inputs and differently shaped outputs; an INT8 fixture
checks integer compatibility. Descriptor layouts and writes are checked too.

Named production-path tests are recorded for the five VUIDs in `vuid_review.json`.
Harness self-tests that deliberately corrupt captured arguments or binding state
verify the oracle itself and are not counted as additional production coverage.
Only valid graphs are submitted to the real emulator/validation layer. Fake
execution does not compute model outputs and is not numerical qualification.

This establishes five individually mapped tests, not completion of the broader
VUID audit. Existing numerical and real Vulkan validation gates remain separate.

The implementation manifest also records `quantized_lowerings` keyed by LiteRT
operator. This avoids confusing INT8 TANH implemented through TABLE with a
nonexistent INT8 type row for the TOSA TANH instruction. Each quantized record
lists its constraints, fixtures and separate emulator/hardware evidence.

External-memory inventory now includes `external_buffer.cc/.h`, tensor-specific
external-handle queries, FD/AHardwareBuffer imports and dedicated allocations.
Five additional reviewed VUIDs have structural resolver-test evidence: tensor
binding 09896/09897/09717, dedicated tensor allocation size 09710, and
AHardwareBuffer allocation size 02383. The broader
VUID audit remains partial; unreviewed rules and stale source-review fingerprints
continue to fail `--check` rather than being automatically cleared.
