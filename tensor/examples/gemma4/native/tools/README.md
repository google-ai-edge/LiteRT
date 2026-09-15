# Native runner preparation and validation tools

The Python tools inspect/export the published E2B `.litertlm` bundle, regenerate
fixed histories, and compare saved outputs without model inference, builds or
device actions. The separate [LiteRT-LM adapter](litert_lm/README.md) is C++
source for the CPU performance baseline. Large model files and captures remain
outside the source tree. See the [comparison recipe](../REPRODUCE.md) for the
native build and matched performance workload.

The export code preserves the original coefficient mapping and exact INT2
widening checks. Path arguments replace the original experiment-specific
locations. It requires Python 3.10+, NumPy, FlatBuffers' Python package and a
`flatc` compatible with the repository schema. Generate the TFLite Python
bindings from this checkout, from the LiteRT root:

```sh
mkdir -p "$SCHEMA_DIR"
flatc --python -o "$SCHEMA_DIR" tflite/converter/schema/schema.fbs
```

`SCHEMA_DIR` must contain `tflite/Model.py`, including the schema's INT2 and
StableHLO composite support. The matching `.litertlm` container schema and
generated parser are retained in [schema](schema); no converter checkout or
LiteRT-LM executable is imported. The inventory omits decoding the unused
LlmMetadata protobuf payload, while retaining its section boundaries.

## Export workflow

Set `MODEL` to the original `.litertlm` bundle and choose new `AUDIT` and
`MATCHED_BUNDLE` directories. Run from the LiteRT root:

```sh
TOOLS=tensor/examples/gemma4/native/tools
python3 "$TOOLS/inspect_bundle.py" --model "$MODEL" \
  --schema-dir "$SCHEMA_DIR" --output-dir "$AUDIT"
python3 "$TOOLS/trace_kv.py" --inventory "$AUDIT/published-inventory.json" \
  --schema-dir "$SCHEMA_DIR" --output "$AUDIT/kv-graph-trace.json"
python3 "$TOOLS/summarize_kv.py" --trace "$AUDIT/kv-graph-trace.json" \
  --output "$AUDIT/kv-placement-summary.json"
python3 "$TOOLS/export_bundle.py" --schema-dir "$SCHEMA_DIR" \
  --inventory "$AUDIT/published-inventory.json" \
  --trace "$AUDIT/kv-graph-trace.json" --kv "$AUDIT/kv-placement-summary.json" \
  --output-dir "$MATCHED_BUNDLE"
python3 "$TOOLS/validate_export.py" --schema-dir "$SCHEMA_DIR" \
  --inventory "$AUDIT/published-inventory.json" \
  --trace "$AUDIT/kv-graph-trace.json" --kv "$AUDIT/kv-placement-summary.json" \
  --bundle-dir "$MATCHED_BUNDLE" --output "$AUDIT/export-validation.json"
```

The tools refuse to overwrite output artifacts. Run Python normally, without
`-O`: source-specific assertions are part of export and validation. The exporter
requires all 277 decode FC weights, 552 static activation scales, 227 learned
norms, 35 layer scalars, both embedding tables and the fixed constants. It
preserves numeric INT2 codes while widening their transport files to signed
INT4; the C++ loader reconstructs compact INT2 for supported static matrices.
The 35 per-layer embedding tables are assembled in token-major order, with
their original scale columns. Unknown variable float coefficients fail export.

The independent validator checks the source bundle SHA, every emitted tensor
and scale file, INT2 numeric codes, embedding partitions, and matching active
FC coefficients across graph signatures. The published model layout is a
deliberate precondition of these scripts; they are not a universal bundle
converter. Full export/readback of multi-gigabyte data is separate from the
small syntax/comparator checks and should be recorded when actually run.

## Fixed histories and saved-output comparisons

```sh
python3 tensor/examples/gemma4/native/tools/prepare_fixtures.py
python3 -B -m unittest discover \
  -s tensor/examples/gemma4/native/tools -p '*_test.py'
python3 tensor/examples/gemma4/native/tools/compare_live.py \
  "$REFERENCE_CAPTURE" "$NATIVE_CAPTURE" \
  --fixtures tensor/examples/gemma4/native/fixtures/manifest.json \
  --case native_boundary_4096_2 --no-cache --require-bitwise \
  --output "$NEW_COMPARISON_JSON"
```

`prepare_fixtures.py` verifies the copied seed manifest before repeating its
token body. It reproduces the five default TSVs byte for byte; no tokenizer or
dataset installation is needed. Generated metadata records the seed filename
and pinned SHA256 rather than an absolute checkout path, so moving the checkout
does not change the default manifest. To generate a different fixture set without
changing checked-in files, supply `--output-dir`. `--include-8192` only creates
an additional input file; it does not execute the long case.

`compare_live.py` uses [compare_runs.py](compare_runs.py) for full-vocabulary
metrics and adds logical INT8-cache comparison. It validates histories, run
completion, raw file sizes, finite values and argmax reports before comparing
numbers. With cache dumps, it aligns the fixed reference's transposed V layout
to native token-major values and compares only committed positions. Without
dumps, use `--no-cache`; this does not provide cache-code evidence.

Comparison success normally means the comparison completed. Exact equality is
enforced only with `--require-bitwise`. The scripts report distribution errors
even when the chosen token is unchanged. The 19 local comparator tests cover
malformed captures, history/target alignment, nonfinite/truncated vectors,
argmax drift, and logical-cache layout/code mismatches.

## Performance summaries

[summarize_performance.py](summarize_performance.py) reads completed native and
LiteRT-LM captures from the [comparison recipe](../REPRODUCE.md):

```bash
python3 tensor/examples/gemma4/native/tools/summarize_performance.py \
  "$RESULTS" --output "$RESULTS/summary.json"
```

By default, `$RESULTS/native` and `$RESULTS/litert-lm` contain the per-length
capture directories. The tool expects one warmup and three measured sessions,
excludes warmups from summaries, and reports each session plus median first-logit
seconds and decode tokens/s. It rejects incomplete or diagnostic captures,
invalid timings, and incompatible histories, capacities or thread settings.
It labels LiteRT-LM's XNNPACK and YNNPACK paths separately. For a separately
captured YNNPACK comparison, pass `--litert-lm-dir` with that results directory.

The JSON records cannot establish model/executable hashes, device identity or
CPU affinity; retain those with the campaign as described in the recipe. The
21 synthetic summarizer tests cover capture validation and timing aggregation.
