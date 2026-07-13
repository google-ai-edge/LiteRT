# How to doc

QNN SDK setup is required for each machine before running `litert_scripts.sh`
for the first time.

First, copy QNN setup script from google3:

```
cp third_party/qairt/latest/bin/check-linux-dependency.sh /tmp
```

Then, run:

```
sudo bash /tmp/check-linux-dependency.sh
```

## Usage of litert_scripts.sh

The script is for internal testing for LiteRt compiler plugin and runtime
(dispatch API, dispatch delegate)

### Simplest end2end example

```
./litert_scripts.sh pipeline Qualcomm V75 <path to tflite model> <path to output qnn model>
```

This command first compiles your input tflite model to output qnn model for your
Qualcomm V75 HTP SoC. Then it pushes all compiled model and needed shared
libraries to the connected phone. Then it compares results between XnnPack and
Npu runtime.

By default the scripts only works for single connected device. When there are
multiple devices connected, update `ADB_OPTION` in `script_init.sh` to `-s [your
decive hash]`.

### General use cases

#### Compile a tflite model to a npu tflite model

```
./litert_scripts.sh apply <SoC manufature> <Soc Model> <path to tflite model> <path to output qnn model> [--skip_numerics]
```

e.g.

```
./litert_scripts.sh apply apply Qualcomm V75 /tmp/add.tflite /tmp/add_npu_v75.tflite
```

#### Compile a tflite model to a npu bytecode

```
./litert_scripts.sh compile <SoC manufature> <Soc Model> <path to tflite model> <path to output qnn model>
```

e.g.

```
./litert_scripts.sh compile Qualcomm V75 /tmp/add.tflite /tmp/add_npu_v75.bin
```

#### Invoke a npu tflite model

```
./litert_scripts.sh invoke <path to qnn model>
```

e.g.

```
./litert_scripts.sh invoke /tmp/add_npu_v75.tflite
```

#### Invoke a npu tflite model and compare numerics with CPU

```
./litert_scripts.sh invoke <path to qnn model> <path to cpu model>
```

e.g.

```
./litert_scripts.sh invoke /tmp/add_npu_v75.tflite /tmp/add.tflite
```
