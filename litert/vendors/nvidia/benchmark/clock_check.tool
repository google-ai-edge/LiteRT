#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The host's clocks, for a measurement's record: the kernel's tick and frequency (adjtimex), the
status of the NTP discipline, and the rates of CLOCK_MONOTONIC and CLOCK_REALTIME against
CLOCK_MONOTONIC_RAW over a short window. The CLI's turn timers run on the monotonic clock, which
the kernel advances from the same adjusted timekeeper as the real-time clock (the tick and the
frequency scale both; only the raw clock is unadjusted): a tick other than 10000 us, or a
monotonic rate off the raw clock's by more than 0.1%, makes every rate the CLI reports wrong by
that much (cuda-wsl, 2026-10-20 to 28: tick 10503 and rising, the monotonic clock 5% fast, every
tokens/s cell 5% low). The real-time clock's rate against the raw clock shows the time daemon's
steps inside the window and is reported, not gated. Prints one JSON object; exits 1 when the
clock is not fit."""
import ctypes
import ctypes.util
import json
import sys
import time


class Timex(ctypes.Structure):
  _fields_ = [('modes', ctypes.c_int), ('offset', ctypes.c_long), ('freq', ctypes.c_long),
              ('maxerror', ctypes.c_long), ('esterror', ctypes.c_long), ('status', ctypes.c_int),
              ('constant', ctypes.c_long), ('precision', ctypes.c_long), ('tolerance', ctypes.c_long),
              ('time_sec', ctypes.c_long), ('time_usec', ctypes.c_long), ('tick', ctypes.c_long),
              ('ppsfreq', ctypes.c_long), ('jitter', ctypes.c_long), ('shift', ctypes.c_int),
              ('stabil', ctypes.c_long), ('jitcnt', ctypes.c_long), ('calcnt', ctypes.c_long),
              ('errcnt', ctypes.c_long), ('stbcnt', ctypes.c_long), ('tai', ctypes.c_int),
              ('pad', ctypes.c_int * 11)]


def kernel_clock():
  t = Timex()
  libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
  state = libc.adjtimex(ctypes.byref(t))
  return {'tick_us': t.tick, 'freq_ppm': round(t.freq / 65536, 3), 'status': t.status,
          'state': state, 'esterror_us': t.esterror}


def rates(seconds):
  r0, m0, w0 = time.time(), time.monotonic(), time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
  time.sleep(seconds)
  r, m, w = time.time() - r0, time.monotonic() - m0, time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - w0
  return {'window_s': seconds, 'realtime_s': round(r, 4), 'monotonic_s': round(m, 4), 'raw_s': round(w, 4),
          'monotonic_over_raw': round(m / w, 5), 'realtime_over_raw': round(r / w, 5)}


def main():
  seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
  report = {'kernel': kernel_clock(), 'rates': rates(seconds)}
  k, r = report['kernel'], report['rates']
  reasons = []
  if k['state'] >= 0 and k['tick_us'] != 10000:
    reasons.append(f"the kernel tick is {k['tick_us']} us, not 10000")
  if abs(r['monotonic_over_raw'] - 1.0) > 0.001:
    reasons.append(f"the monotonic clock runs at {r['monotonic_over_raw']:.4f} of the raw clock")
  report['fit'] = not reasons
  report['reasons'] = reasons
  print(json.dumps(report))
  return 0 if not reasons else 1


if __name__ == '__main__':
  sys.exit(main())
