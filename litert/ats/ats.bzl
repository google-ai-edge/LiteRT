# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Macros to define pre-configured ATS test suites and run through the litert_device* macros.
"""

load("//litert/integration_test:litert_device.bzl", "litert_device_exec")
load("//litert/integration_test:litert_device_common.bzl", "device_rlocation", "dispatch_device_rlocation", "host_rlocation", "is_gpu_backend", "is_npu_backend", "plugin_device_rlocation", "version_target_suffix")
load(
    "//litert/integration_test:litert_device_script.bzl",
    "litert_device_script",
    # copybara:uncomment_begin(google-only)
    # "make_cns_pull_model_provider",
    # copybara:uncomment_end
    "make_download_model_provider",
)

def _make_ats_args(init = [], **kwargs):
    def _fmt_re(re):
        if len(re) == 1:
            return re[0]
        return "\\'({})\\'".format("|".join(re))

    extra_flags = kwargs.get("extra_flags", [])
    exec_args = [
        "--quiet=false",
    ] + extra_flags + init

    backend = kwargs.get("backend", "cpu")
    if is_npu_backend(backend):
        backend_flag = "npu"
    elif is_gpu_backend(backend):
        backend_flag = "gpu"
    else:
        backend_flag = backend
    exec_args.append("--backend={}".format(backend_flag))

    dont_register = kwargs.get("dont_register", [])
    if dont_register:
        exec_args.append(
            "--dont_register={}".format(_fmt_re(dont_register)),
        )

    do_register = kwargs.get("do_register", [])
    if do_register:
        exec_args.append(
            "--do_register={}".format(_fmt_re(do_register)),
        )

    param_seeds = kwargs.get("param_seeds", {})
    if param_seeds:
        exec_args.append(
            "--seeds=\"{}\"".format(",".join(["{}:{}".format(k, v) for k, v in param_seeds.items()])),
        )
    return exec_args

def _resolve_extra_models(name, extra_models):
    """Resolves `extra_models` into data deps, model providers, and runtime paths."""
    model_providers = []
    data = []
    cns_models = []

    # Resolved runtime filesystem paths passed via `--extra_models=...` to the `ats`
    # binary on the target device (JIT) or host workstation (AOT), whereas `extra_models`
    # holds the input Bazel labels, CNS source paths, or `.tar.gz` download URLs.
    device_paths = []
    host_paths = []

    if extra_models:
        cns_models = [m for m in extra_models if m.startswith("/") and not m.startswith("//")]
        url_models = [m for m in extra_models if m.startswith("http://") or m.startswith("https://")]
        label_models = [m for m in extra_models if m not in cns_models and m not in url_models]

        num_source_types = (1 if cns_models else 0) + (1 if url_models else 0) + (1 if label_models else 0)
        if num_source_types > 1:
            fail("extra_models cannot mix CNS paths, download URLs (http(s)://...), and Bazel labels in the same target.")

        # Resolve Bazel target labels (e.g. `//path/to:model.tflite` or filegroups) to their
        # staged runfiles paths on the device and host (using the exact file path for
        # `.tflite` targets and the parent directory for filegroups), deduplicating paths.
        if label_models:
            data = label_models
            for m in label_models:
                get_parent = not m.endswith(".tflite")
                dev_loc = device_rlocation(m, get_parent = get_parent)
                if dev_loc not in device_paths:
                    device_paths.append(dev_loc)
                host_loc = host_rlocation(m, get_parent = get_parent)
                if host_loc not in host_paths:
                    host_paths.append(host_loc)

        for i, url in enumerate(url_models):
            url_provider_name = "{}_download_models_provider_{}".format(name, i)
            make_download_model_provider(
                name = url_provider_name,
                url = url,
            )
            model_providers.append(":" + url_provider_name)

        if cns_models:
            cns_provider_name = name + "_cns_models_provider"

            # copybara:uncomment make_cns_pull_model_provider(name = cns_provider_name, cns_paths = cns_models)
            model_providers.append(":" + cns_provider_name)
            for m in cns_models:
                if not m.endswith(".tflite"):
                    cns_dir_name = m.rstrip("/").rsplit("/", 1)[-1]
                    dev_dir = "/data/local/tmp/runfiles/user/tmp/litert_extras/" + cns_dir_name
                    if dev_dir not in device_paths:
                        device_paths.append(dev_dir)

    if model_providers:
        device_paths.append("/data/local/tmp/runfiles/user/tmp/litert_extras")

    return struct(
        data = data,
        model_providers = model_providers,
        cns_models = cns_models,
        device_paths = device_paths,
        host_paths = host_paths,
    )

def litert_define_ats(
        backend,
        name,
        jit_suffix,
        compile_only_suffix,
        compile_aot_and_run_suffix = None,
        dont_register = [],
        do_register = [],
        param_seeds = {},
        extra_flags = [],
        extra_models = [],
        platform = "android",
        aot_shard_count = None,
        aot_tags = []):
    """Defines a pre-configured ATS test suite.

    Args:
      name: The name of the test suite.
      backend: The backend to use for the test suite.
      jit_suffix: Suffix for the Just-In-Time execution target.
      compile_only_suffix: Suffix for the Compile-Only target.
      compile_aot_and_run_suffix: Suffix for the Compile AOT and Run target.
      dont_register: A list of regular expressions for tests that should be skipped
          (registered in GTest as SKIPPED so coverage stats remain complete).
      do_register: A list of regular expressions for tests that should be registered
          (non-matching tests are omitted from registration entirely).
      param_seeds: A dictionary of parameter seeds for the test suite.
      extra_flags: A list of extra flags to pass to the test suite.
      extra_models: A list of labels to directories or files containing .tflite models,
          CNS paths to .tflite models, or .tar.gz URLs (https://...) to
          download .tflite models from. Cannot mix source types.
      platform: Target OS platform ("android" or "macos").
      aot_shard_count: Optional shard_count for the host compile-only sh_test target.
      aot_tags: Additional tags for the host compile-only sh_test target.
    """
    if "append" not in dir(backend):
        backend = [backend]

    if compile_aot_and_run_suffix:
        fail("Compile aot and run on device is not supported yet.")

    resolved_models = _resolve_extra_models(name, extra_models)

    for b in backend:
        # TODO: Unify local workdir paths for scripting.
        version_suffix = "_" + version_target_suffix(b) if version_target_suffix(b) else ""

        init_run_args = []
        if platform != "macos" and resolved_models.device_paths:
            init_run_args.append("--extra_models={}".format(",".join(resolved_models.device_paths)))

        if is_npu_backend(b):
            init_run_args += [
                "--dispatch_dir=\"{}\"".format(dispatch_device_rlocation(b)),
                "--plugin_dir=\"{}\"".format(plugin_device_rlocation(b)),
            ]

        run_args = _make_ats_args(
            init = init_run_args,
            backend = b,
            dont_register = dont_register,
            do_register = do_register,
            param_seeds = param_seeds,
            extra_flags = extra_flags,
        )

        if jit_suffix != None:
            litert_device_exec(
                name = name + jit_suffix + version_suffix,
                target = "//litert/ats:ats",
                remote_suffix = "_remote",
                local_suffix = "",
                exec_args = run_args,
                backend_id = b,
                platform = platform,
                model_providers = resolved_models.model_providers,
                extra_models = resolved_models.cns_models,
                data = resolved_models.data,
            )

        init_compile_args = ["--compile_mode=true"]
        if resolved_models.host_paths:
            init_compile_args.append("--extra_models={}".format(",".join(resolved_models.host_paths)))

        compile_args = _make_ats_args(
            init = init_compile_args,
            backend = b,
            dont_register = dont_register,
            do_register = do_register,
            param_seeds = param_seeds,
            extra_flags = extra_flags,
        )

        if compile_only_suffix != None:
            litert_device_script(
                name = name + compile_only_suffix + version_suffix,
                script = "//litert/ats:ats_aot.sh",
                bin = "//litert/ats:ats",
                backend_id = b,
                exec_args = compile_args,
                build_for_host = True,
                build_for_device = False,
                model_providers = resolved_models.model_providers,
                data = resolved_models.data,
                is_test = True,
                shard_count = aot_shard_count,
                tags = [
                    "noasan",
                    "nomsan",
                    "nosan",
                    "notsan",
                ] + aot_tags,
            )
