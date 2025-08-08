# QNN HTP Optrace Profiling

## Overview

QAIRT provides a set of serial tools that help users optimize their graphs. This page explains how to use these tools, integrated with a Python script and LiteRt, to identify performance bottlenecks.

## Usage

To obtain analytical files, please follow these steps:

1. Compile `apply_plugin_main`.
2. Generate Context Binary with Qualcomm Compiler Plugin
    
    Run the following command to compile the model and generate the context binary using `--qualcomm_profiling optrace`:
    ```
    bazel-bin/litert/tools/apply_plugin_main \
      --cmd compile \
      --libs bazel-bin/litert/vendors/qualcomm/compiler \
      --soc_model <soc_model> \
      --soc_manufacturer Qualcomm  \
      --model path/to/tflite \
      -o path/to/ctx_bin \
      --qualcomm_profiling optrace
    ```
    After running this command, you will also get a `<graph_name>_schematic.bin` file in your run directory.

3. To execute the QAIRT tools, use the following command:
    ```
    python3 run.py \
      --ctx_bin path/to/ctx_bin.bin \
      --schematic_bin path/to/schematic_bin.bin \
      --hostname <hostname> \
      --serial <serial> \
      --htp_arch <htp_arch, e.g., V79>
      --output_dir path/to/output_dir \
    ```

The `path/to/output_dir` directory contains a QNN HTP Analysis Summary (QHAS) HTML file and a Chrometrace JSON file.

* `chromeTrace_qnn_htp_analysis_summary.html`
![image](./assets/qhas.png)
* `chromeTrace.json` on [Perfetto](https://www.ui.perfetto.dev/)
![image](./assets/perfetto.png)

## Tools Reference
* [QNN HTP Optrace Profiling](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/htp_backend.html#qnn-htp-optrace-profiling)
* [qnn-net-run](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-net-run)
* [qnn-profile-viewer](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-profile-viewer)
* [qnn-context-binary-utility](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-context-binary-utilityqnn-context-binary-utility)
