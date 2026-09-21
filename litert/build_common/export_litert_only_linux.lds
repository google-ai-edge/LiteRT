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

  local:

    /* Hide everything else */

    *;
};
