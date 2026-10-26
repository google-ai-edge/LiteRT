VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.
  */

  global:

    /* Compiler Plugin (glob required by lld --no-undefined-version when linking dispatch-only DSOs) */

    LiteRtCompilerPluginQueryInterface*;

    /* Dispatch (glob required by lld --no-undefined-version when linking compiler-plugin-only DSOs) */

    LiteRtDispatchQueryInterface*;
    LiteRtDispatch_Mock*;
    LiteRtDispatchExample*;

    /* Required for libunwind. This is needed if built and then run internally */
    google_find_phdr;

  local:

    /* Hide everything else */

    *;
};
