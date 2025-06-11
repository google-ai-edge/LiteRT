
# Environment Setting
- QNN_SDK_ROOT
- HEXAGON_SDK_ROOT
- ANDROID_NDK_ROOT
- OP_PACKAGE_XML
- GENERATE_FOLDER

# Code Generation
```sh
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python/
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${OP_PACKAGE_XML} -o ${GENERATE_FOLDER}
```

# File Structure
- op package source code will be generated in ${GENERATE_FOLDER} like:
```log
${GENERATE_FOLDER}
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

# Makefile patch
- Please modify Makefile with this patch to fix the undefined symbol error in x86, we need to link libQnnHtp.
```
X86_TARGET_LIB := $(QNN_SDK_ROOT)/lib/x86_64-linux-clang
X86_LDFLAGS:= -Wl,--whole-archive -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread -L$(X86_TARGET_LIB) -lQnnHtp
```

# Op Implemnetation
- Implement op in ${GENERATE_FOLDER}/LiteRtQualcommOpPackage/src/ops/.


# Build
- Make sure clang++ can be found in your environment.
```sh
export QNN_SDK_ROOT=<Your QNN Root>
export HEXAGON_SDK_ROOT=<Your HEXAGON SDK Root>
export ANDROID_NDK_ROOT=<Your Android NDK Root>
cd ${GENERATE_FOLDER}/LiteRtQualcommOpPackage
make htp_x86
make htp_aarch64
make htp_v75
```


# Register Op Package in Qnn backend
```cpp
- build/x86_64-linux-clang/libQnnLiteRtQualcommOpPackage.so
- - Interface provider: ExamplePackageInterfaceProvider
- - Target: CPU

- build/hexagon-v75/libQnnLiteRtQualcommOpPackage.so
- - Interface provider: ExamplePackageInterfaceProvider
- - Target: HTP

- build/aarch64-android/libQnnLiteRtQualcommOpPackage.so
- - Interface provider: ExamplePackageInterfaceProvider
- - Target: HTP

```