VERS_1.0 {

  /*
    Version script for libLiteRt.so built with CMake.

    Same as export_litert_runtime_only_linux.lds, plus the C options API
    (Lrt*). The source-built CMake C++ API, including its no-Abseil variant,
    resolves the options API from libLiteRt.so because the options
    implementation depends on Abseil and can't be linked into no-Abseil
    consumers.

    Logging, status string, API version and layout helpers stay hidden.
    CMake consumers link them statically through
    litert_c_api_dynamic_runtime_helpers, like Bazel and cc_sdk do.
  */

  global:

    /* All LiteRt C APIs */
    LiteRt*;
    kLiteRtRuntimeBuiltin;

    /* C options API (c/options/*.h) */
    Lrt*;

  local:

    /* c/litert_layout.h */
    /* LiteRtLayout is non-opaque type (its struct is defined in the header) */
    /* and its APIs are not defined by litert_c_api_library rule. */
    LiteRtGetNumLayoutElements;
    LiteRtIsSameLayout;

    /* c/internal/litert_logging.h */
    LiteRtClearSinkLogger;
    LiteRtCreateLogger;
    LiteRtCreateSinkLogger;
    /* LiteRtDefaultLoggerLog; */
    LiteRtDestroyLogger;
    LiteRtGetDefaultLogger;
    LiteRtGetLoggerIdentifier;
    LiteRtGetLogSeverityName;
    LiteRtGetMinLoggerSeverity;
    LiteRtGetSinkLoggerMessage;
    LiteRtGetSinkLoggerSize;
    LiteRtLoggerLog;
    LiteRtSetDefaultLogger;
    LiteRtSetMinLoggerSeverity;
    LiteRtUseSinkLogger;
    LiteRtUseStandardLogger;
    /* c/litert_common.h */
    LiteRtCompareApiVersion;
    LiteRtGetStatusString;

    /* Hide everything else */
    *;
};
