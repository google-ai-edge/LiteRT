# Running Integration Tests

From within the `rules_apple` workspace, use the following command to run all
the tests:

```bash
bazel test //test/... //tools/...
```

When using Xcode 10 to run the tests, you may also be required to run the tests
with the `--spawn_strategy=local` flag. This will disable sandbox mode for the
integration tests, which adds too many component paths for the intermediate
files generated during the tests, and makes `xcodebuild` based tests fail to
run.
