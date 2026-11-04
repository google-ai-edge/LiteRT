// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
#include "litert/vendors/nvidia/windows_compat.h"

#include <psapi.h>
#include <winioctl.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>
#pragma comment(lib, "psapi.lib")
#undef assert
#define assert(x)                                                             \
  do {                                                                        \
    if (!(x)) {                                                               \
      fprintf(stderr, "FAIL line %d: %s errno=%d winerr=%lu\n", __LINE__, #x, \
              errno, GetLastError());                                         \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

int main() {
  if (const char* temporary = std::getenv("TEST_TMPDIR")) {
    std::filesystem::current_path(temporary);
  }
  _set_error_mode(_OUT_TO_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  NvidiaUnlink("windows-compat2-published.bin");
  char path[] = "windows-compat2.XXXXXX";
  int fd = mkstemp(path);
  assert(fd >= 0);
  char absolute[PATH_MAX];
  assert(realpath(path, absolute));
  std::string generic(absolute);
  std::replace(generic.begin(), generic.end(), '\\', '/');
  int generic_fd = NvidiaOpen(generic.c_str(), _O_RDONLY);
  assert(generic_fd >= 0);
  _close(generic_fd);
  DWORD sparse_result = 0;
  assert(DeviceIoControl(reinterpret_cast<HANDLE>(_get_osfhandle(fd)),
                         FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
                         &sparse_result, nullptr));
  const int64_t at = (int64_t{5} << 30) + 123;
  assert(_lseeki64(fd, at, SEEK_SET) == at);
  assert(_write(fd, "native64", 8) == 8);
  const int64_t other_at = (int64_t{3} << 30) + 65;
  assert(_lseeki64(fd, other_at, SEEK_SET) == other_at);
  assert(_write(fd, "offset32", 8) == 8);
  assert(_lseeki64(fd, 777, SEEK_SET) == 777);
  OVERLAPPED old_style{};
  old_style.Offset = static_cast<DWORD>(other_at);
  old_style.OffsetHigh = static_cast<DWORD>(uint64_t(other_at) >> 32);
  char probe[8];
  DWORD probe_bytes = 0;
  assert(ReadFile(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), probe, 8,
                  &probe_bytes, &old_style));
  const auto old_position = _lseeki64(fd, 0, SEEK_CUR);
  printf("Original synchronous ReadFile position: before=777 after=%lld\n",
         old_position);
  assert(old_position == other_at + 8);
  assert(_lseeki64(fd, 777, SEEK_SET) == 777);
  NvidiaFileStat info{};
  assert(NvidiaFstat(fd, &info) == 0 && info.st_size == at + 8);
  auto reader = [&](int64_t where, const char* expected) {
    for (int i = 0; i < 100; ++i) {
      char bytes[8];
      assert(pread(fd, bytes, 8, where) == 8);
      assert(!memcmp(bytes, expected, 8));
    }
  };
  std::thread a(reader, at, "native64"), b(reader, other_at, "offset32");
  a.join();
  b.join();
  assert(_lseeki64(fd, 0, SEEK_CUR) == 777);
  assert(pread(fd, &info, 1, at + 8) == 0);
  assert(_chsize_s(fd, 65536) == 0);
  void* mapped = mmap(nullptr, 65536, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(mapped != MAP_FAILED);
  char source_name[PATH_MAX], mapped_name[PATH_MAX];
  assert(GetFinalPathNameByHandleA(reinterpret_cast<HANDLE>(_get_osfhandle(fd)),
                                   source_name, PATH_MAX,
                                   FILE_NAME_NORMALIZED | VOLUME_NAME_NT));
  assert(
      GetMappedFileNameA(GetCurrentProcess(), mapped, mapped_name, PATH_MAX));
  assert(_stricmp(source_name, mapped_name) == 0);
  assert(munmap(mapped, 65536) == 0);
  mapped = mmap(nullptr, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  assert(mapped != MAP_FAILED);
  static_cast<char*>(mapped)[4096] = 42;
  PSAPI_WORKING_SET_EX_INFORMATION cow{};
  cow.VirtualAddress = static_cast<char*>(mapped) + 4096;
  assert(QueryWorkingSetEx(GetCurrentProcess(), &cow, sizeof(cow)));
  assert(cow.VirtualAttributes.Valid && !cow.VirtualAttributes.Shared);
  char disk_byte = 1;
  assert(pread(fd, &disk_byte, 1, 4096) == 1 && disk_byte == 0);
  assert(munmap(mapped, 65536) == 0);
  void* zeros = mmap(nullptr, 65536, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  assert(zeros != MAP_FAILED && static_cast<char*>(zeros)[2345] == 0);
  static_cast<char*>(zeros)[2345] = 7;
  assert(munmap(zeros, 65536) == 0);
  assert(fchmod(fd, S_IRUSR) == 0);
  assert(link(path, "windows-compat2-published.bin") == 0);
  assert(NvidiaUnlink(path) == 0);
  NvidiaFileStat published{};
  assert(NvidiaStat("windows-compat2-published.bin", &published) == 0);
  assert(info.st_ino == published.st_ino);
  assert(_close(fd) == 0);
  fd = NvidiaOpen("windows-compat2-published.bin", _O_RDONLY | O_NOFOLLOW);
  assert(fd >= 0);
  assert(fchmod(fd, S_IRUSR | S_IWUSR) == 0);
  _close(fd);
  assert(NvidiaUnlink("windows-compat2-published.bin") == 0);
  puts(
      "PASS: distinct-offset concurrent 64-bit positional IO with unchanged "
      "file position, identity, file/anonymous mapping, immutable hard-link "
      "publication");
}
