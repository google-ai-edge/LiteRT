# NeuroPilot Dependency Updater

This directory contains tools to automate the process of upgrading the MediaTek
NeuroPilot SDK dependency in LiteRT.

## upgrade_neuro_pilot.py

The `upgrade_neuro_pilot.py` script automates the update of the NeuroPilot SDK
URL across multiple configuration files in the repository, and updates the
internal NeuroPilot tool.

### Usage

To upgrade to a new version, run the following command from your workspace root:

```bash
blaze run //litert/google/neuro_pilot_updater:upgrade_neuro_pilot -- \
  --neuro_pilot_url="https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/<UUID>.gz"
```

### Files Updated

The script automatically updates the following files:

1.  `opensource_only/third_party/neuro_pilot/workspace.bzl`
2.  `opensource_only/ci/tools/python/vendor_sdk/mediatek/setup.py`
3.  `litert/vendors/CMakeLists.txt`

It also downloads the SDK and copies new folders to
`third_party/neuro_pilot`.

## Running Tests

To verify the script's logic, run the associated unit test:

```bash
blaze test //litert/google/neuro_pilot_updater:upgrade_neuro_pilot_test
```
