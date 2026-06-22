# Qnn profiler
System preparation:

```
sudo apt install libc++-dev
```

For compiled conetxt binary `compiled.bin`, run profiler on android devices via

```
sh profiling_script.sh compiled.bin
```

The scripts does:

1. Read context binary info via qnn tool.
2. Generate input data.
3. Run model with `qnn-net-run` and pull profiling results back to your workstation.