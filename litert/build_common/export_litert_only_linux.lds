VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.

    TODO: Add all vendor symbols. Also export qnn libc++ symbols
    (statically linked) as "protected" as needed.
  */

  global:

    /* Compiler Plugin */

    LiteRt*CompilerPlugin*;

    /* Compiled Result */

    LiteRt*CompiledResult*;

    /* Dispatch */

    LiteRtDispatch*;

    /* Required for libunwind. This is needed if built and then run internally */
    google_find_phdr;

  local:

    /* Hide everything else */

    *;
};
