# Flatbuffers

Header files in this directory are copied from https://github.com/google/flatbuffers.git at tag `v25.9.23`.

`ClassicLocale` in `util.h` has been modified to be fully inline so that no separate `.cpp` compilation is needed. The static member is defined out-of-class with the C++17 `inline` keyword to avoid ODR violations. This avoids POSIX filesystem dependencies (`mkdir`, `realpath`) from the upstream `util.cpp` that are unavailable on Hexagon QuRT.
