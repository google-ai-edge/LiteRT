# QAIRT SDK

## About QAIRT SDK

QAIRT is a suite of tools that help you develop, run, and optimize AI models for
Qualcomm-supported hardware.

-   Official Document:
    https://docs.qualcomm.com/doc/80-63442-10/topic/general_overview.html
-   Download link: please download from
    https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community.
    For example, users could download QAIRT-2.44.0 from
    https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.44.0.260225/v2.44.0.260225.zip
-   Interface Header Files: `${QAIRT}/include/QNN/`

## Required Libraries

Here are some standard concepts of "How QNN works on the Qualcomm devices" users
should know.

### Backend

Currently, LiteRT QC compiler plugin supports the following QNN backends: `Htp`,
`Dsp`, `Ir`, `Saver`, `Gpu`.

### Platform

Here are the supported operating system (OS) platforms of Qualcomm devices:

Platform                   | Target
-------------------------- | -----------------------------------
`aarch64-android`          | Android devices
`x86_64-linux-clang`       | x86 Linux devices
`x86_64-windows-msvc`      | x86 Windows devices
`aarch64-oe-linux-gcc11.2` | aarch64 Linux devices (IoT devices)
`aarch64-windows-msvc`     | aarch64 Windows devices

### Hexagon Arch

When using HTP backend, users need to know which architecture is used in their
devices. Please search "Supported Snapdragon devices" in
https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html and
find the "Hexagon Arch" column to get the Hexagon Arch of the target device. For
example, if you want to run on a Snapdragon 8 Elite Gen 5 (SM8850), the Hexagon
Arch is V81.

### QNN libraries

After we know the above information, users can locate all required libraries to
execute QNN:

-   `${QAIRT}/lib/{Platform}/libQnnSystem.so`
-   `${QAIRT}/lib/{Platform}/libQnn{Backend}*.so`
-   `${QAIRT}/lib/hexagon-v{HexagonArch}/unsigned/libQnnHtpV{HexagonArch}Skel.so`

For example, users want to execute QNN HTP backends on SM8850, they need to find
the following files:

-   `${QAIRT}/lib/aarch64-android/libQnnSystem.so`
-   `${QAIRT}/lib/aarch64-android/libQnnHtp.so`
-   `${QAIRT}/lib/aarch64-android/libQnnHtpPrepare.so`
-   `${QAIRT}/lib/aarch64-android/libQnnHtpV81Stub.so`
-   `${QAIRT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so`

For example, users want to compile a model for SM8850 with HTP backend in a x86
Linux host machine, they need to find the following files:

-   `${QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so`
-   `${QAIRT}/lib/x86_64-linux-clang/libQnnSystem.so`

## (Optional) Specify other QAIRT version in LiteRT

Modify `${LITERT}/third_party/qairt/workspace.bzl` and change `strip_prefix` and
`url`. Please get the url of QAIRT SDK from
https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community.

For example, the original workspace.bzl is using "qairt/2.42.0.251225" and users
want to use 2.44, then the workspace.bzl should be changed from:

```python
def qairt():
    configurable_repo(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        local_path_env = "LITERT_QAIRT_SDK",
        strip_prefix = "qairt/2.42.0.251225",
        url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip",
        file_extension = "zip",
    )
```

To:

```python
def qairt():
    configurable_repo(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        local_path_env = "LITERT_QAIRT_SDK",
        strip_prefix = "qairt/2.44.0.260225",
        url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.44.0.260225/v2.44.0.260225.zip",
        file_extension = "zip",
    )
```
