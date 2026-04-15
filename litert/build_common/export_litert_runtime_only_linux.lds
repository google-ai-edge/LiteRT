VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.

    TODO: Add all vendor symbols. Also export qnn libc++ symbols
    (statically linked) as "protected" as needed.
  */

  global:

    /* All LiteRt C APIs */
    LiteRt*;

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
