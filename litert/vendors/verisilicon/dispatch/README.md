# Verisilicon dispatch plug-in
This directory contains the Verisilicon NPU (IPs) dispatch plug-in implementation for LiteRT.

## Compile model

You can use Acuity toolkit to compile a model to LiteRT compiled model.

```python
from acuitylib.vsi_nn import VSInn

nn = VSInn()
net = nn.load_onnx('xxx.onnx')
nn.export_tflite(net, output_path='xxx', gen_litert=True, optimize='VIP9000NANOSI_PID0X1000005D', pack_nbg=True,
                  viv_sdk="your vsi_sdk path")
```

Note:
1. Your vsi_sdk must follow the same directory structure as "acuity/acuitynet/acuitylib/vsi_sdk/prebuilt-sdk/x86_64_linux/*".
2. The folder structure should be /your_path/vsi_sdk/prebuilt-sdk/x86_64_linux, meaning the root of vsi_sdk is /your_path/vsi_sdk.
3. Then set the export parameter : viv_sdk="/your_path/vsi_sdk".
4. gen_litert=True & pack_nbg=True must be set.

## Runtime Options

### device_index
  The device index of multi-device (multiple NPU devices on an SoC.).

### core_index
  The VIP-core index of multi-core(eg.VIP9400).

### time_out
  Timeout(ms) for NPU computation of a model.

### profile_level
  Execute operations(commands) one by one and show profile log.

### dump_nbg
  Dump NBG resource(nbg, input, output).

## Build Verisilicon dispatch

### bazel
```sh
bazel build //litert/c:litert_tflite_runtime_c_api_so
bazel build //litert/vendors/verisilicon/dispatch:dispatch_api_so
bazel build //litert/tools:run_model
```

### cmake
```sh
cmake --preset default
cmake --build cmake_build -j
```

## Run

Copy the libLiteRtDispatch_Verisilicon.so to $VIPLITE_SDK/drivers

```sh
cp /path_to/libLiteRtDispatch_Verisilicon.so $VIPLITE_SDK/drivers
export LD_LIBRARY_PATH=$VIPLITE_SDK/drivers:$LD_LIBRARY_PATH
run_model  --graph acuity_compiled.tflite --accelerator npu --dispatch_library_dir $VIPLITE_SDK/drivers

# run with options
run_model  --graph acuity_compiled.tflite --accelerator npu --dispatch_library_dir $VIPLITE_SDK/drivers --verisilicon_profile_level=2
```
