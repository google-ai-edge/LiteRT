# Contributing to `debugging-litert-gpu-accuracy`

## Scope & Closed-Source Requirement

This skill lives under
`_agents/skills/debugging_litert_gpu_accuracy/`,
which is excluded from OSS export by the
`"**/_agents/**"` rule in
`copy.bara.sky`. Keep all internal script paths,
CNS/TFHub paths, and internal debugging workflows inside this `_agents/`
directory tree so they remain closed-source.

## Validation Before Submitting Changes

1.  Run the skill structural validator:

    ```bash
    /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
      //_agents/skills/debugging_litert_gpu_accuracy:validate_debugging_litert_gpu_accuracy_test
    ```

2.  Keep `SKILL.md` under 500 lines and move any long C++ probe snippets or
    op-table dumps into `references/` (linked one level deep from `SKILL.md`).
