# Plan to remove TensorFlow dependencies from LiteRT WORKSPACE

The goal is to make LiteRT independent of the `@org_tensorflow` Bazel workspace by copying necessary configuration files and defining dependencies directly.

## Phase 1: Analysis
1.  Identify all `.bzl` files loaded from `@org_tensorflow` in the project.
    -   `tensorflow:workspace0.bzl`, `workspace1.bzl`, `workspace2.bzl`, `workspace3.bzl`
    -   `tensorflow:tensorflow.bzl`, `tensorflow.default.bzl`, `strict.default.bzl`
    -   And others found via grep.
2.  Identify all external repositories loaded by `tf_workspace0/1/2/3()` that are actually needed by LiteRT.
3.  Identify all `BUILD` file dependencies on `@org_tensorflow//...` targets.

## Phase 2: Copying Necessary Files
1.  **2-1. Move Licenses:**
    -   Identify the definition of `@org_tensorflow//tensorflow:license`.
    -   Copy the `LICENSE` file and relevant `BUILD` rule to `litert/build_common/tensorflow`.
    -   Update `WORKSPACE` or `BUILD` files to point to the local license target.
2.  **2-2. Move Platform Targets:**
    -   Identify definitions for platform targets (e.g., `@org_tensorflow//tensorflow:emscripten`, `ios`, `linux_x86_64`, `macos`, etc.).
    -   Copy necessary `config_setting`s and `platform` definitions to `litert/build_common/tensorflow`.
    -   Update usages to point to local targets.
3.  **2-3. Move Remaining Dependencies:**
    -   Copy necessary `.bzl` and `BUILD` files from the current `third_party/tensorflow` (which is currently a full TF checkout).
    -   Ensure these files are accessible via local paths (e.g., `//litert/build_common/tensorflow:workspace0.bzl`) instead of `@org_tensorflow`.

## Phase 3: Refactoring Workspace and Build Files
1.  **Workspace Refactoring:**
    -   In `WORKSPACE`, replace `load("@org_tensorflow//tensorflow:workspace*.bzl", ...)` with local loads from `//litert/build_common/tensorflow/...`.
    -   Define LiteRT-specific workspace macros (e.g., `litert_workspace0/1/2/3`) that only include needed dependencies.
    -   Define all required `http_archive` and other repository rules directly or via LiteRT macros.
2.  **Build File Refactoring:**
    -   Update all `load` statements in `BUILD` files to point to local `.bzl` files.
    -   Replace dependencies on `@org_tensorflow//...` with either LiteRT-specific targets or direct external repository targets (e.g., `@com_google_absl//...` instead of `@org_tensorflow//third_party/absl/...` if applicable).
3.  **Config Settings:**
    -   Copy necessary `config_setting`s from TensorFlow's `BUILD` files to LiteRT.

## Phase 4: Verification and Cleanup
1.  Verify the build works without `USE_LOCAL_TF=true` and without downloading the full TensorFlow archive if possible (or at least without using the `@org_tensorflow` workspace name).
2.  Remove `tensorflow_source_repo` definition for `org_tensorflow` from `WORKSPACE`.
3.  Iteratively fix any remaining broken dependencies.

## Immediate Next Steps
-   Grep for all `@org_tensorflow` loads to get a complete list of files to copy.
-   Analyze `workspace*.bzl` to see which dependencies are core to LiteRT.
