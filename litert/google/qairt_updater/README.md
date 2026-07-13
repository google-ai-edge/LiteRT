# QAIRT Dependency Updater

This directory contains tools to automate the process of upgrading the Qualcomm
AI Runtime (QAIRT) dependency in LiteRT.

## upgrade_qairt.py

The `upgrade_qairt.py` script automates the update of the QAIRT SDK version
across multiple configuration files in the repository.

### Usage

To upgrade to a new version, run the following command from your workspace root:

```bash
blaze run //litert/google/qairt_updater:upgrade_qairt -- \
  --qairt_url="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<VERSION>/v<VERSION>.zip"
```

Replace `<VERSION>` with the specific version string (e.g., `2.42.0.251225`).

### Files Updated

The script automatically updates the following files:

1.  `opensource_only/third_party/qairt/workspace.bzl`
2.  `litert/google/npu_runtime_libraries/fetch_qualcomm_library.sh`
3.  `litert/google/npu_runtime_libraries/fetch_qualcomm_library_jit.sh`
4.  `opensource_only/ci/tools/python/vendor_sdk/qualcomm/setup.py`
5.  `litert/vendors/CMakeLists.txt`

## Running Tests

To verify the script's logic, run the associated unit test:

```bash
blaze test //litert/google/qairt_updater:upgrade_qairt_test
```
