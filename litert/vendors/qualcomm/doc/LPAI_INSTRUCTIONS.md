# LPAI Backend

This guide covers the LiteRT QC compiler plugin + QNN Native path (using
`qnn-net-run` for on-device execution). Support for LiteRT QC compiler plugin +
LiteRT native execution is currently under development.

LPAI (Low Power AI) is a programmable ML engine optimized for low-area,
low-power applications. It is optimized for deeply embedded use cases such as:

-   Always-on voice use cases on mobile, XR or IoT platforms.
-   Voice and music use cases on IoT platforms.
-   Voice AI use cases such as Automatic Speech Recognition (ASR), Speech
    Caption, and etc.
-   Always-on camera use cases on mobile, XR or IoT platforms.
-   Qualcomm Sensor hubs.

## Execution Modes

| Mode                   | Mechanism              | When to use              |
| ---------------------- | ---------------------- | ------------------------ |
| **Traditional mode     | ARM process dispatches | General ARM applications |
: (cDSP)**               : inference to cDSP via  :                          :
:                        : FastRPC communication  :                          :
| **Direct mode (aDSP)** | Graph runs natively on | Audio/sensor workloads   |
:                        : aDSP, no per-inference : living directly on aDSP  :
:                        : FastRPC communication  :                          :

Both modes consume the same offline-compiled context binary, compile once and
pick mode at execution time.

## Known Limitations

-   **Quantized models only.** LPAI supports quantized 8-bit and quantized
    16-bit networks. Float models are rejected at compile time.
-   **Narrower op coverage than HTP.** Not all ops are supported. See
    `${QAIRT}/docs/QAIRT-Docs/QNN/OpDef/LpaiOpDefSupplement.html` for the full
    op support list.
-   **Limited aDSP memory.** The aDSP scratch and persistent memory budget is
    small. Large models may not fit.

--------------------------------------------------------------------------------

# Prerequisites

Download QAIRT SDK from
https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community
and unzip it.

Ensure that all `${}` variables are properly configured before proceeding:

Variable         | Description
---------------- | -------------------------------------------------------
`${QAIRT}`       | Root path of the unzipped QAIRT SDK
`${WORK_DIR}`    | Working directory for your model
`${MODEL}`       | Path to the source quantized `.tflite` model
`${SOC_MODEL}`   | Target SoC string (e.g. `SM8850`)
`${TEST_FOLDER}` | Test folder on device (e.g. `/data/local/tmp/lpai_run`)

## Signing Libraries for Production Devices

On production devices, the Hexagon DSP only loads signed binaries. All DSP-side
libraries must be signed before deployment.

### Required Libraries

Libraries marked ⚠️ are DSP-side and must be signed for production devices.
Replace `{N}` with the LPAI hw version and `{M}` with the Hexagon arch version
(see version table below).

Library                            | Path in `${QAIRT}/`          | Traditional mode (cDSP) | Direct mode (aDSP)
---------------------------------- | ---------------------------- | :---------------------: | :----------------:
`qnn-net-run`                      | `bin/aarch64-android/`       | ✓                       | ✓
`libQnnSystem.so`                  | `lib/aarch64-android/`       | ✓                       | ✓
`libQnnLpai.so` (ARM)              | `lib/aarch64-android/`       | ✓                       | —
`libQnnLpaiStub.so`                | `lib/aarch64-android/`       | ✓                       | —
`libQnnLpaiSkel.so` ⚠️             | `lib/lpai-v{N}/unsigned/`    | ✓                       | —
`libQnnLpai.so` (aDSP) ⚠️          | `lib/lpai-v{N}/unsigned/`    | —                       | ✓
`libQnnLpaiNetRunExtensions.so` ⚠️ | `lib/lpai-v{N}/unsigned/`    | —                       | ✓
`libQnnNetRunDirectV{M}Stub.so`    | `lib/aarch64-android/`       | —                       | ✓
`libQnnHexagonSkel_dspApp.so` ⚠️   | `lib/hexagon-v{M}/unsigned/` | —                       | ✓
`libQnnNetRunDirectV{M}Skel.so` ⚠️ | `lib/hexagon-v{M}/unsigned/` | —                       | ✓

### Install QPM and Hexagon SDK

1.  Download and install QPM3 from
    https://qpm.qualcomm.com/#/main/tools/details/QPM3.

    ```bash
    sudo apt install ./qpm3-<version>-linux.deb
    ```

2.  Log in and install the Hexagon SDK matching your target device (see version
    table below). Currently we use `6.6.0.0`.

    ```bash
    qpm-cli --login <your_username>
    qpm-cli --install HexagonSDK6.x
    ```

### Version Table

Use the tables below to find the versions for your device, then substitute `{N}`
and `{M}` throughout the rest of this guide (e.g. `lpai-v6`, `hexagon-v81`).

**SoC to LPAI and Hexagon version mapping:**

| SoC family | LPAI hw version | Hexagon arch (`{M}`) | SDK lib paths     |
:            : (`{N}`)         :                      :                   :
| ---------- | :-------------: | :------------------: | ----------------- |
| SM8850     | v6              | v81                  | `lib/lpai-v6`,    |
:            :                 :                      : `lib/hexagon-v81` :
| SM8750     | v5              | v79                  | `lib/lpai-v5`,    |
:            :                 :                      : `lib/hexagon-v79` :
| SM8650     | v5              | v75                  | `lib/lpai-v5`,    |
:            :                 :                      : `lib/hexagon-v75` :

For the full SoC-to-version mapping, see
`${QAIRT}/docs/QAIRT-Docs/QNN/general/lpai/lpai_backend.html`.

**Hexagon arch to required Hexagon SDK version (for signing):**

Hexagon arch (`{M}`) | Min. Hexagon SDK
:------------------: | :--------------:
v73                  | 5.5.5
v75                  | 5.5.5
v79                  | 6.0.0
v81                  | 6.4.0
v85                  | 6.5.0
v89                  | 6.5.0

### Sign with elfsigner.py

```bash
export HEXAGON_SDK_ROOT=<path_to_hexagon_sdk>

python $HEXAGON_SDK_ROOT/tools/elfsigner/elfsigner.py \
  -i <unsigned_lib.so> \
  -o <signed_output_dir>/
```

> **Production signing (CASS):** Deploying to locked production devices requires
> signing via Qualcomm's CASS server using your OEM certificate and capability
> ID. Please contact your Qualcomm account representative for access.

--------------------------------------------------------------------------------

# Compile and Execute

```
quantized .tflite
      │
      │  apply_plugin_main --qualcomm_backend ir
      ▼
.dlc  (QNN Deep Learning Container graph format)
      │
      │  qnn-context-binary-generator --backend libQnnLpai.so
      ▼
.bin  (LPAI context binary, compiled for aDSP)
      │
      │  [Production only] Sign DSP-side libraries with elfsigner.py
      │
      ▼
qnn-net-run on device
   ├── Traditional mode (cDSP)
   └── Direct mode (aDSP)
```

## Generate DLC

Use `apply_plugin_main` with `--qualcomm_backend ir` to emit a QNN DLC file
instead of compiling directly to HTP bytecode. For instructions on building
`apply_plugin_main`, see `BUILD_INSTRUCTIONS.md`.

```bash
export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang

apply_plugin_main \
  --cmd apply \
  --libs <path_to_compiler_plugin_libs> \
  --soc_model ${SOC_MODEL} \
  --soc_manufacturer Qualcomm \
  --model ${MODEL} \
  --qualcomm_backend ir \
  --qualcomm_dlc_dir ${WORK_DIR}/dlc
```

Output: one or more `qnn_partition_<N>.dlc` files in `${WORK_DIR}/dlc/`. LiteRT
may split the model into multiple partitions.

## Compile DLC to LPAI Context Binary

Create the following two configuration files before running
`qnn-context-binary-generator`.

**`${WORK_DIR}/lpaiParams.conf`** — sets the LPAI hardware version for your SoC:

```json
{
  "lpai_backend": {
    "target_env": "adsp",
    "enable_hw_ver": "v6"
  }
}
```

Replace `v6` with the LPAI hardware version for your SoC (e.g. `v5` for SM8650).

**`${WORK_DIR}/lpai_config.json`** — points to the backend extension library:

```json
{
  "backend_extensions": {
    "shared_library_path": "<QAIRT>/lib/x86_64-linux-clang/libQnnLpaiNetRunExtensions.so",
    "config_file_path": "<WORK_DIR>/lpaiParams.conf"
  }
}
```

Compile the DLC:

```bash
export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

mkdir -p ${WORK_DIR}/lpai_ctx_bin

${QAIRT}/bin/x86_64-linux-clang/qnn-context-binary-generator \
  --backend ${QAIRT}/lib/x86_64-linux-clang/libQnnLpai.so \
  --dlc_path ${WORK_DIR}/dlc/qnn_partition_0.dlc \
  --binary_file my_model_lpai \
  --output_dir ${WORK_DIR}/lpai_ctx_bin \
  --config_file ${WORK_DIR}/lpai_config.json \
  --log_level info
```

For models with multiple partitions, pass a comma-separated list: `--dlc_path
a.dlc,b.dlc`.

Output: `${WORK_DIR}/lpai_ctx_bin/my_model_lpai.bin`

## Run on Device with QNN Native Path

### Traditional Mode (cDSP)

```bash
adb root
adb shell "mkdir -p ${TEST_FOLDER}/lib ${TEST_FOLDER}/dsp"

# ARM-side
adb push ${QAIRT}/bin/aarch64-android/qnn-net-run        ${TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so    ${TEST_FOLDER}/lib/
adb push ${QAIRT}/lib/aarch64-android/libQnnLpai.so      ${TEST_FOLDER}/lib/
adb push ${QAIRT}/lib/aarch64-android/libQnnLpaiStub.so  ${TEST_FOLDER}/lib/

# DSP-side (sign required for production devices)
adb push ${QAIRT}/lib/lpai-v6/unsigned/libQnnLpaiSkel.so ${TEST_FOLDER}/dsp/

# Model and inputs
adb push ${WORK_DIR}/lpai_ctx_bin/my_model_lpai.bin      ${TEST_FOLDER}/
adb push ${WORK_DIR}/input_list.txt                      ${TEST_FOLDER}/

adb shell "
  cd ${TEST_FOLDER}
  chmod +x qnn-net-run
  export LD_LIBRARY_PATH=${TEST_FOLDER}/lib:\$LD_LIBRARY_PATH
  export ADSP_LIBRARY_PATH=${TEST_FOLDER}/dsp
  mkdir -p output
  ./qnn-net-run \
    --backend ./lib/libQnnLpai.so \
    --retrieve_context ./my_model_lpai.bin \
    --input_list ./input_list.txt \
    --output_dir ./output \
    --use_native_input_files --use_native_output_files \
    --log_level info
"
```

### Direct Mode (aDSP)

Direct mode requires `adb root`. The runtime config must set
`is_persistent_binary: true` (see below). The same context binary compiled above
works for both modes — no recompilation needed.

Create the config **on the host** and push it to device:

**`${WORK_DIR}/lpai_direct_config.json`:**

```json
{
  "backend_extensions": {
    "shared_library_path": "/data/local/tmp/lpai_direct/adsp/libQnnLpaiNetRunExtensions.so",
    "config_file_path": "/data/local/tmp/lpai_direct/lpaiParams.conf"
  },
  "context_configs": {
    "is_persistent_binary": true
  }
}
```

Update the paths above to match `${TEST_FOLDER}` on your device.

```bash
adb root
adb shell "mkdir -p ${TEST_FOLDER}/adsp"

# aDSP-side (sign required for production devices)
adb push ${QAIRT}/lib/lpai-v6/unsigned/libQnnLpai.so                    ${TEST_FOLDER}/adsp/
adb push ${QAIRT}/lib/lpai-v6/unsigned/libQnnLpaiNetRunExtensions.so    ${TEST_FOLDER}/adsp/
adb push ${QAIRT}/lib/hexagon-v81/unsigned/libQnnHexagonSkel_dspApp.so  ${TEST_FOLDER}/adsp/
adb push ${QAIRT}/lib/hexagon-v81/unsigned/libQnnNetRunDirectV81Skel.so ${TEST_FOLDER}/adsp/

# ARM-side
adb push ${QAIRT}/bin/aarch64-android/qnn-net-run                       ${TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so                   ${TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnNetRunDirectV81Stub.so      ${TEST_FOLDER}/

# Configs, model, and inputs
adb push ${WORK_DIR}/lpai_direct_config.json   ${TEST_FOLDER}/config.json
adb push ${WORK_DIR}/lpaiParams.conf           ${TEST_FOLDER}/
adb push ${WORK_DIR}/lpai_ctx_bin/my_model_lpai.bin  ${TEST_FOLDER}/
adb push ${WORK_DIR}/input_list.txt            ${TEST_FOLDER}/

adb shell "
  cd ${TEST_FOLDER}
  chmod +x qnn-net-run
  export LD_LIBRARY_PATH=${TEST_FOLDER}:${TEST_FOLDER}/adsp:\$LD_LIBRARY_PATH
  export ADSP_LIBRARY_PATH=${TEST_FOLDER}
  mkdir -p output
  ./qnn-net-run \
    --backend ./adsp/libQnnLpai.so \
    --direct_mode \
    --retrieve_context ./my_model_lpai.bin \
    --input_list ./input_list.txt \
    --config_file ./config.json \
    --output_dir ./output \
    --use_native_input_files --use_native_output_files
"
```

> **Important:** `ADSP_LIBRARY_PATH` must point to the **parent** of the `adsp/`
> subdirectory (i.e. `${TEST_FOLDER}`, not `${TEST_FOLDER}/adsp`). FastRPC
> appends `adsp/` automatically. Setting it to `${TEST_FOLDER}/adsp` causes
> FastRPC to look under `.../adsp/adsp/` and fall back to the system library in
> `/vendor/dsp/adsp/`, resulting in `dspApp->run FAILED res = 9`.

## Reference Logs

Successful execution produces the following output patterns.

### Traditional Mode (cDSP)

```
[INFO] [Qnn Runtime] qnn-net-run pid=...
...
[INFO] [Qnn Runtime] Finished Executing Graphs
```

### Direct Mode (aDSP)

```
[INFO] [Qnn Runtime] NonRPC net-run SUCCESS
```

--------------------------------------------------------------------------------

# References

Document               | Location
---------------------- | --------
LPAI backend overview  | `${QAIRT}/docs/QAIRT-Docs/QNN/general/lpai/lpai_backend.html`
Direct mode tutorial   | `${QAIRT}/docs/QAIRT-Docs/QNN/general/lpai/lpai_execution_direct_tutorial.html`
Model generation guide | `${QAIRT}/docs/QAIRT-Docs/QNN/general/lpai/lpai_backend_model_generation.html`
LPAI op support table  | `${QAIRT}/docs/QAIRT-Docs/QNN/OpDef/LpaiOpDefSupplement.html`
QPM3 download          | https://qpm.qualcomm.com/#/main/tools/details/QPM3
QAIRT SDK download     | https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community
