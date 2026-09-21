/*
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
