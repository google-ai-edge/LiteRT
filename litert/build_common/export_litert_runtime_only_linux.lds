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

    /* Hide everything else */
    *;
};
