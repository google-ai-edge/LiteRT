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

    /*
      RTTI for cross-DSO dynamic_cast of adla:: types only.
      Do not use _ZTV* / _ZTI* / _ZTS*: those also match libc++ and
      libstdc++ iostream vtables. libcompiler_adla.so then binds its
      ostream to this .so and crashes in locale::locale inside Compile().
    */

    _ZTIN4adla*;
    _ZTSN4adla*;
    _ZTVN4adla*;

  local:

    /* libc++ (Android std::__ndk1) and libstdc++ iostream / locale. */
    _ZTVNSt6__ndk1*;
    _ZTINSt6__ndk1*;
    _ZTSNSt6__ndk1*;
    _ZTTNSt6__ndk1*;
    _ZTVN10__cxxabiv1*;
    _ZTIN10__cxxabiv1*;
    _ZTSN10__cxxabiv1*;
    _ZTVSt*;
    _ZTISt*;
    _ZTSSt*;
    _ZTTSt*;
    _ZTVNSt7__cxx11*;
    _ZTINSt7__cxx11*;
    _ZTSNSt7__cxx11*;
    _ZTTNSt7__cxx11*;

    *;
};
