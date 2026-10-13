/* Copyright (C) 2026 Amlogic, Inc. All rights reserved. */

VERS_1.0 {
  /*
    Export LiteRT compiler-plugin ABI + RTTI needed for cross-DSO
    dynamic_cast with libcompiler_adla.so (NodeBuilder::SetParameter).

    Hide everything else — especially embedded std::filesystem symbols
    such as path::_M_split_cmpts. Those must resolve from system
    libstdc++ / libc++; exporting them from this .so hijacks ADLA and
    causes segfaults inside Compiler::CompileInternal.
  */

  global:

    /* Compiler Plugin */

    LiteRt*CompilerPlugin*;

    /* Compiled Result */

    LiteRt*CompiledResult*;

    /* RTTI for cross-DSO dynamic_cast / typeid (adla::I*Operator, etc.) */

    _ZTI*;
    _ZTS*;
    _ZTV*;

  local:

    *;
};
