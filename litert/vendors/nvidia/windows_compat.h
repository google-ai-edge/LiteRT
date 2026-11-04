// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// Native Windows equivalents for the NVIDIA backend's file operations.
#ifndef LITERT_VENDORS_NVIDIA_WINDOWS_COMPAT_H_
#define LITERT_VENDORS_NVIDIA_WINDOWS_COMPAT_H_
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

using ssize_t = intptr_t;
#ifndef PATH_MAX
#define PATH_MAX 32768
#endif
#define O_CLOEXEC _O_NOINHERIT
#define O_NOFOLLOW 0x10000000
#define S_IRUSR _S_IREAD
#define S_IWUSR _S_IWRITE
#define S_IWGRP 0
#define S_IWOTH 0
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 1
#define MAP_ANONYMOUS 2
#define MAP_NORESERVE 4
#define MAP_FAILED reinterpret_cast<void*>(-1)
#define MADV_DONTNEED 4
#define POSIX_FADV_DONTNEED 4
#define _SC_PAGESIZE 1

inline int NvidiaWindowsError() {
  switch (GetLastError()) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      errno = ENOENT;
      break;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
      errno = EEXIST;
      break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
      errno = EACCES;
      break;
    case ERROR_INVALID_HANDLE:
      errno = EBADF;
      break;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
      errno = ENOMEM;
      break;
    case ERROR_DISK_FULL:
      errno = ENOSPC;
      break;
    default:
      errno = EIO;
      break;
  }
  return -1;
}
inline int NvidiaOpen(const char* path, int flags, int mode = 0600) {
  const DWORD access =
      (flags & _O_RDWR) ? GENERIC_READ | GENERIC_WRITE
                        : ((flags & _O_WRONLY) ? GENERIC_WRITE : GENERIC_READ);
  DWORD creation = OPEN_EXISTING;
  if (flags & _O_CREAT) creation = (flags & _O_EXCL) ? CREATE_NEW : OPEN_ALWAYS;
  DWORD attributes = FILE_ATTRIBUTE_NORMAL;
  if (flags & O_NOFOLLOW) attributes |= FILE_FLAG_OPEN_REPARSE_POINT;
  HANDLE handle = CreateFileA(
      path, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, creation, attributes, nullptr);
  if (handle == INVALID_HANDLE_VALUE) return NvidiaWindowsError();
  FILE_ATTRIBUTE_TAG_INFO tag{};
  if ((flags & O_NOFOLLOW) &&
      (!GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &tag,
                                     sizeof(tag)) ||
       (tag.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))) {
    CloseHandle(handle);
    errno = ELOOP;
    return -1;
  }
  if (flags & _O_TRUNC) {
    if (!SetEndOfFile(handle)) {
      CloseHandle(handle);
      return NvidiaWindowsError();
    }
  }
  const int fd = _open_osfhandle(
      reinterpret_cast<intptr_t>(handle),
      _O_BINARY | _O_NOINHERIT | (flags & (_O_RDWR | _O_WRONLY | _O_APPEND)));
  if (fd < 0) CloseHandle(handle);
  return fd;
}
struct NvidiaFileStat {
  uint64_t st_dev = 0, st_ino = 0;
  int64_t st_size = 0;
  unsigned st_mode = 0;
  struct Time {
    int64_t tv_sec = 0, tv_nsec = 0;
  } st_mtim, st_ctim;
};
inline int NvidiaFstat(int fd, NvidiaFileStat* out) {
  HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  BY_HANDLE_FILE_INFORMATION info{};
  FILE_BASIC_INFO basic{};
  if (!GetFileInformationByHandle(h, &info) ||
      !GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic)))
    return NvidiaWindowsError();
  out->st_dev = info.dwVolumeSerialNumber;
  out->st_ino = (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
  out->st_size = (int64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
  out->st_mode =
      (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? _S_IFDIR : _S_IFREG;
  out->st_mode |= _S_IREAD;
  if (!(info.dwFileAttributes & FILE_ATTRIBUTE_READONLY))
    out->st_mode |= _S_IWRITE;
  auto time = [](int64_t ticks) {
    ticks -= 116444736000000000LL;
    return NvidiaFileStat::Time{ticks / 10000000, (ticks % 10000000) * 100};
  };
  out->st_mtim = time(basic.LastWriteTime.QuadPart);
  out->st_ctim = time(basic.ChangeTime.QuadPart);
  return 0;
}
inline int NvidiaStat(const char* path, NvidiaFileStat* out) {
  int fd = NvidiaOpen(path, _O_RDONLY);
  if (fd < 0) return -1;
  int result = NvidiaFstat(fd, out);
  _close(fd);
  return result;
}
inline ssize_t pread(int fd, void* out, size_t bytes, int64_t offset) {
  // OVERLAPPED offsets on a synchronous handle still update that handle's
  // current position. An independent asynchronous handle preserves pread's
  // file-position and concurrent-reader semantics, including CRT descriptors.
  HANDLE read_handle =
      ReOpenFile(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), GENERIC_READ,
                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                 FILE_FLAG_OVERLAPPED);
  if (read_handle == INVALID_HANDLE_VALUE) return NvidiaWindowsError();
  OVERLAPPED ov{};
  ov.Offset = static_cast<DWORD>(offset);
  ov.OffsetHigh = static_cast<DWORD>(uint64_t(offset) >> 32);
  // Each read owns its event, preserving concurrent positional reads.
  ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (!ov.hEvent) {
    DWORD error = GetLastError();
    CloseHandle(read_handle);
    SetLastError(error);
    return NvidiaWindowsError();
  }
  DWORD received = 0;
  BOOL ok = ReadFile(read_handle, out,
                     static_cast<DWORD>(std::min<size_t>(bytes, 1u << 30)),
                     &received, &ov);
  DWORD error = GetLastError();
  if (!ok && error == ERROR_IO_PENDING)
    ok = GetOverlappedResult(read_handle, &ov, &received, TRUE);
  error = GetLastError();
  CloseHandle(ov.hEvent);
  CloseHandle(read_handle);
  if (!ok && error != ERROR_HANDLE_EOF) {
    SetLastError(error);
    return NvidiaWindowsError();
  }
  return received;
}
inline char* realpath(const char* path, char* out) {
  HANDLE h =
      CreateFileA(path, FILE_READ_ATTRIBUTES,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    NvidiaWindowsError();
    return nullptr;
  }
  DWORD n = GetFinalPathNameByHandleA(h, out, PATH_MAX,
                                      FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  CloseHandle(h);
  if (!n || n >= PATH_MAX) {
    errno = ENAMETOOLONG;
    return nullptr;
  }
  return out;
}
inline int fchmod(int fd, int mode) {
  HANDLE h =
      ReOpenFile(reinterpret_cast<HANDLE>(_get_osfhandle(fd)),
                 FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES,
                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0);
  if (h == INVALID_HANDLE_VALUE) return NvidiaWindowsError();
  FILE_BASIC_INFO basic{};
  if (!GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic))) {
    DWORD error = GetLastError();
    CloseHandle(h);
    SetLastError(error);
    return NvidiaWindowsError();
  }
  if (mode & _S_IWRITE)
    basic.FileAttributes &= ~FILE_ATTRIBUTE_READONLY;
  else
    basic.FileAttributes |= FILE_ATTRIBUTE_READONLY;
  BOOL ok = SetFileInformationByHandle(h, FileBasicInfo, &basic, sizeof(basic));
  DWORD error = GetLastError();
  CloseHandle(h);
  SetLastError(error);
  return ok ? 0 : NvidiaWindowsError();
}
inline int link(const char* source, const char* destination) {
  return CreateHardLinkA(destination, source, nullptr) ? 0
                                                       : NvidiaWindowsError();
}
inline int NvidiaRename(const char* source, const char* destination) {
  return MoveFileExA(source, destination, MOVEFILE_REPLACE_EXISTING)
             ? 0
             : NvidiaWindowsError();
}
inline int NvidiaUnlink(const char* path) {
  // A read-only temporary hard link must be removable without changing the
  // immutable published file's attributes. POSIX delete semantics support it.
  HANDLE h = CreateFileA(
      path, DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) return NvidiaWindowsError();
  struct Disposition {
    DWORD Flags;
  } disposition{0x1 | 0x2 | 0x10};
  BOOL ok =
      SetFileInformationByHandle(h, static_cast<FILE_INFO_BY_HANDLE_CLASS>(21),
                                 &disposition, sizeof(disposition));
  DWORD error = GetLastError();
  CloseHandle(h);
  if (!ok) {
    SetLastError(error);
    return NvidiaWindowsError();
  }
  return 0;
}
inline int mkstemp(char* pattern) {
  const size_t n = strlen(pattern);
  for (unsigned attempt = 0; attempt < 100; ++attempt) {
    unsigned value = std::random_device{}();
    static constexpr char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    for (size_t i = n - 6; i < n; ++i) {
      pattern[i] = digits[value % 36];
      value /= 36;
    }
    int fd = NvidiaOpen(pattern, _O_RDWR | _O_CREAT | _O_EXCL);
    if (fd >= 0 || errno != EEXIST) return fd;
  }
  errno = EEXIST;
  return -1;
}
inline int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite && std::getenv(name)) return 0;
  const int result = _putenv_s(name, value);
  if (result) {
    errno = result;
    return -1;
  }
  return 0;
}
inline long sysconf(int) {
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize;
}
inline void* mmap(void*, size_t size, int prot, int flags, int fd,
                  int64_t offset) {
  if (flags & MAP_ANONYMOUS) {
    void* p =
        VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return p ? p : MAP_FAILED;
  }
  HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  const bool writable = (prot & PROT_WRITE) != 0;
  HANDLE mapping = CreateFileMappingW(
      file, nullptr, writable ? PAGE_WRITECOPY : PAGE_READONLY, 0, 0, nullptr);
  if (!mapping) {
    NvidiaWindowsError();
    return MAP_FAILED;
  }
  void* p = MapViewOfFile(mapping, writable ? FILE_MAP_COPY : FILE_MAP_READ,
                          static_cast<DWORD>(uint64_t(offset) >> 32),
                          static_cast<DWORD>(offset), size);
  CloseHandle(mapping);
  if (!p) {
    NvidiaWindowsError();
    return MAP_FAILED;
  }
  return p;
}
inline int munmap(void* p, size_t) {
  MEMORY_BASIC_INFORMATION info{};
  if (!VirtualQuery(p, &info, sizeof(info))) return NvidiaWindowsError();
  BOOL ok = info.Type == MEM_PRIVATE ? VirtualFree(p, 0, MEM_RELEASE)
                                     : UnmapViewOfFile(p);
  return ok ? 0 : NvidiaWindowsError();
}
// Windows has no safe per-range equivalent of Linux MADV_DONTNEED on mapped
// files. Report unsupported instead of discarding COW data or trimming the
// process working set (which would affect unrelated model pages).
inline int madvise(void*, size_t, int) {
  errno = ENOTSUP;
  return -1;
}
inline int posix_fadvise(int, int64_t, int64_t, int) { return ENOTSUP; }
#else
#include <sys/stat.h>

#include <cstdio>
using NvidiaFileStat = struct stat;
inline int NvidiaStat(const char* path, NvidiaFileStat* out) {
  return stat(path, out);
}
inline int NvidiaRename(const char* a, const char* b) {
  return std::rename(a, b);
}
#endif
#endif
