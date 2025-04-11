package(
    default_visibility = [
        "//visibility:public",
    ],
)

qnn_headers = [
    "include/QNN/QnnBackend.h",
    "include/QNN/QnnCommon.h",
    "include/QNN/QnnContext.h",
    "include/QNN/QnnDevice.h",
    "include/QNN/QnnGraph.h",
    "include/QNN/QnnInterface.h",
    "include/QNN/QnnLog.h",
    "include/QNN/QnnMem.h",
    "include/QNN/QnnOpDef.h",
    "include/QNN/QnnOpPackage.h",
    "include/QNN/QnnProfile.h",
    "include/QNN/QnnProperty.h",
    "include/QNN/QnnSdkBuildId.h",
    "include/QNN/QnnSignal.h",
    "include/QNN/QnnTensor.h",
    "include/QNN/QnnTypes.h",
    "include/QNN/Saver/QnnSaver.h",
    "include/QNN/Saver/QnnSaverCommon.h",
    "include/QNN/System/QnnSystemCommon.h",
    "include/QNN/System/QnnSystemContext.h",
    "include/QNN/System/QnnSystemInterface.h",
    "include/QNN/HTP/QnnHtpDevice.h",
    "include/QNN/HTP/QnnHtpGraph.h",
    "include/QNN/HTP/QnnHtpPerfInfrastructure.h",
    "include/QNN/HTP/QnnHtpProfile.h",
    "include/QNN/HTP/QnnHtpProperty.h",
]

cc_library(
    name = "qnn_lib_headers",
    hdrs = qnn_headers,
    includes = [
        "include/QNN",
    ],
    local_defines = select({
        "//conditions:default": [],
    }),
)
