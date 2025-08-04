# 0. Prerequisites
Please download and set up these environment variables:
- $QNN_SDK_ROOT: Qualcomm AI Runtime SDK
- $HEXAGON_SDK_ROOT: Hexagon SDK
- $ANDROID_NDK_ROOT: Android NDK

Create a XML file to provide the op package information.
- $OP_PACKAGE_XML

Create a folder to store the generated source code.
- $OUTPUT_FOLDER

# 1. Code Generation
Generate the source code of the op package by `qnn-op-package-generator`.
```sh
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python/
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${OP_PACKAGE_XML} -o ${OUTPUT_FOLDER}
```

The source code of op package will be generated in `${OUTPUT_FOLDER}` like:
```log
${OUTPUT_FOLDER}
├── LiteRtQualcommOpPackage
│   ├── Makefile
│   ├── config
│   │   └── CustomOpPackageHtp.xml
│   ├── include
│   └── src
│       ├── LiteRtQualcommOpPackageInterface.cpp
│       └── ops
│           └── ElementWiseAdd.cpp
└── README.md
```


# 2. Makefile Patch
In QNN 2.34, we found that there are some link options missed in x86 platform and we will have undefined symbol in the shared objects. Please modify `X86_LDFLAGS` in the generated Makefile .
```sh
X86_TARGET_LIB := $(QNN_SDK_ROOT)/lib/x86_64-linux-clang
X86_LDFLAGS := -Wl,--whole-archive -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread -L$(X86_TARGET_LIB) -lQnnHtp
```


# 3. Op Implemnetations
Please implement op in `${OUTPUT_FOLDER}/LiteRtQualcommOpPackage/src/ops/`.


# 4. Build Op Package
Please make sure `clang++` can be found in your environment.
```sh
export QNN_SDK_ROOT=<Your Qualcomm AI Runtime SDK Root>
export HEXAGON_SDK_ROOT=<Your HEXAGON SDK Root>
export ANDROID_NDK_ROOT=<Your Android NDK Root>
cd ${OUTPUT_FOLDER}/LiteRtQualcommOpPackage
# build the target platform you need.
make htp_x86
make htp_aarch64
make htp_v75
```


# 5. Register Op Package
Please call `backendRegisterOpPackage` with proper parameters to register custom op package.

For x86 offline prepare:
- packagePath: `${OUTPUT_FOLDER}/build/x86_64-linux-clang/libQnnLiteRtQualcommOpPackage.so`
- interfaceProvider: `"ExamplePackageInterfaceProvider"`
- target: `"CPU"`

For V75 execution:
- packagePath: `${OUTPUT_FOLDER}/build/hexagon-v75/libQnnLiteRtQualcommOpPackage.so`
- interfaceProvider: `"ExamplePackageInterfaceProvider"`
- target: `"HTP"`

For android online prepare:
- packagePath: `${OUTPUT_FOLDER}/build/aarch64-android/libQnnLiteRtQualcommOpPackage.so`
- interfaceProvider: `"ExamplePackageInterfaceProvider"`
- target: `"HTP"`


# 6. Example Commands
apply_plugin_main
```sh
apply_plugin_main \
    --cmd apply \
    --libs bazel-bin/litert/vendors/qualcomm/compiler \
    --soc_model <SoC Model> \
    --soc_manufacturer Qualcomm \
    --model <tflite Model> \
    -o <Output Model> \
    --qualcomm_custom_op_package_path ./libQnnLiteRtQualcommOpPackage.so \
    --qualcomm_custom_op_package_target CPU \
    --qualcomm_custom_op_package_interface_provider LiteRtQualcommOpPackageInterfaceProvider
```

run_model
```sh
export LD_LIBRARY_PATH=${WD}:${LD_LIBRARY_PATH} && export ADSP_LIBRARY_PATH="${WD}" \
&& ${WD}/run_model \
    --graph=<Compiled Model> \
    --dispatch_library_dir=${WD} \
    --iterations <Iteration> \
    --compare_numerical \
    --print_tensors 
```