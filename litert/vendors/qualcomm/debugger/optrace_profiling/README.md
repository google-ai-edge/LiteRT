# QNN HTP Optrace Profiling

## Overview

QAIRT provides a set of serial tools that help users optimize their graphs. This
page explains how to use these tools, integrated with a Python script and
LiteRt, to identify performance bottlenecks.

## Usage

To run LiteRT compilation and QAIRT profiling tools, execute the following
command: `python3 run.py \ --model path/to/model.tflite \ --soc_model <SoC
model, e.g., SM8650> \ --hostname <hostname> \ --serial <serial number> \
--htp_arch <HTP architecture, e.g., V75> \ --output_dir path/to/output_dir`

The `path/to/output_dir` directory contains a QNN HTP Analysis Summary (QHAS)
HTML file and a Chrometrace JSON file.

*   `chromeTrace_qnn_htp_analysis_summary.html` ![image](./assets/qhas.png)
*   `chromeTrace.json` on [Perfetto](https://www.ui.perfetto.dev/)
    ![image](./assets/perfetto.png)

## Tools Reference

*   [QNN HTP Optrace Profiling](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/htp_backend.html#qnn-htp-optrace-profiling)
*   [qnn-net-run](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-net-run)
*   [qnn-profile-viewer](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-profile-viewer)
*   [qnn-context-binary-utility](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-context-binary-utilityqnn-context-binary-utility)
