#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Runs a command and writes the lines of GNU time's -v accounting that the benchmark's

readers use (report.py, nvguard's parse_time): the peak resident set size and
the wall time.
bench.sh times the CLI with it on Windows, where there is no /usr/bin/time; the
peak is the
process's peak working set there, and the children's peak RSS elsewhere.

usage: process_time.py -o FILE COMMAND [ARG...]

Exits with the command's status, as GNU time does.
"""

import subprocess
import sys
import time


def peak_rss_kib(process):
  """The peak resident set of the finished process, KiB."""
  if sys.platform == 'win32':
    import ctypes

    class Counters(ctypes.Structure):
      _fields_ = [
          ('cb', ctypes.c_uint32),
          ('PageFaultCount', ctypes.c_uint32),
      ] + [
          (name, ctypes.c_size_t)
          for name in (
              'PeakWorkingSetSize',
              'WorkingSetSize',
              'QuotaPeakPagedPoolUsage',
              'QuotaPagedPoolUsage',
              'QuotaPeakNonPagedPoolUsage',
              'QuotaNonPagedPoolUsage',
              'PagefileUsage',
              'PeakPagefileUsage',
          )
      ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    # The process handle stays open while the Popen object lives, so the counters of the
    # exited process are still readable.
    handle = ctypes.c_void_p(int(process._handle))
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
      raise ctypes.WinError()
    return counters.PeakWorkingSetSize // 1024
  import resource

  return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def wall_text(seconds):
  """GNU time's wall-clock format: h:mm:ss above an hour, else m:ss.ss."""
  hours, rest = divmod(seconds, 3600)
  minutes, secs = divmod(rest, 60)
  if hours >= 1:
    return f'{int(hours)}:{int(minutes):02d}:{int(secs):02d}'
  return f'{int(minutes)}:{secs:05.2f}'


def main(argv):
  if len(argv) < 4 or argv[1] != '-o':
    sys.stderr.write(__doc__)
    return 2
  out, command = argv[2], argv[3:]
  start = time.monotonic()
  process = subprocess.Popen(command)
  status = process.wait()
  elapsed = time.monotonic() - start
  rss = peak_rss_kib(process)
  with open(out, 'w') as stream:
    stream.write(f'\tCommand being timed: "{" ".join(command)}"\n')
    stream.write(
        f'\tElapsed (wall clock) time (h:mm:ss or m:ss): {wall_text(elapsed)}\n'
    )
    stream.write(f'\tMaximum resident set size (kbytes): {rss}\n')
    stream.write(f'\tExit status: {status}\n')
  return status


if __name__ == '__main__':
  sys.exit(main(sys.argv))
