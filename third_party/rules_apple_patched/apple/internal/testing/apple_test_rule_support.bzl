# Copyright 2019 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper methods for implementing test rules."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@bazel_skylib//lib:shell.bzl",
    "shell",
)
load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
    "AppleCodesigningDossierInfo",
    "AppleDsymBundleInfo",
    "AppleExtraOutputsInfo",
    "AppleTestInfo",
    "AppleTestRunnerInfo",
)

_CoverageFilesInfo = provider(
    doc = """
Internal provider used by the `coverage_files_aspect` aspect to propagate the
transitive closure of sources and binaries that a test depends on. These files
are then made available during the coverage action as they are required by the
coverage infrastructure. The sources are provided in the `coverage_files` field,
and the binaries in the `covered_binaries` field. This provider is only available
when coverage collecting is enabled.
""",
    fields = {
        "coverage_files": "`depset` of files required to be present during a coverage run.",
        "covered_binaries": """
`depset` of files representing the binaries that are being tested under a coverage run.
""",
    },
)

# Key to extract all values for inserting into the binary at load time.
_INSERT_LIBRARIES_KEY = "DYLD_INSERT_LIBRARIES"

def _coverage_files_aspect_impl(target, ctx):
    """Implementation for the `coverage_files_aspect` aspect."""

    # Skip collecting files if coverage is not enabled.
    if not ctx.configuration.coverage_enabled:
        return []

    coverage_files = []

    # Collect this target's coverage files.
    for attr in ["srcs", "hdrs", "non_arc_srcs"]:
        for files in [x.files for x in getattr(ctx.rule.attr, attr, [])]:
            coverage_files.append(files)

    # Collect the binaries themselves from the various bundles involved in the test. These will be
    # passed through the test environment so that `llvm-cov` can access the coverage mapping data
    # embedded in them.
    direct_binaries = []
    transitive_binaries_sets = []
    if AppleBundleInfo in target and target[AppleBundleInfo].binary:
        direct_binaries.append(target[AppleBundleInfo].binary)

    rule_attrs = ctx.rule.attr

    # Collect dependencies coverage files.
    for dep in getattr(rule_attrs, "deps", []):
        coverage_files.append(dep[_CoverageFilesInfo].coverage_files)

    for fmwk in getattr(rule_attrs, "frameworks", []):
        coverage_files.append(fmwk[_CoverageFilesInfo].coverage_files)
        transitive_binaries_sets.append(fmwk[_CoverageFilesInfo].covered_binaries)

    if hasattr(rule_attrs, "test_host") and rule_attrs.test_host:
        coverage_files.append(rule_attrs.test_host[_CoverageFilesInfo].coverage_files)
        transitive_binaries_sets.append(rule_attrs.test_host[_CoverageFilesInfo].covered_binaries)

    return [
        _CoverageFilesInfo(
            coverage_files = depset(transitive = coverage_files),
            covered_binaries = depset(
                direct = direct_binaries,
                transitive = transitive_binaries_sets,
            ),
        ),
    ]

coverage_files_aspect = aspect(
    attr_aspects = ["deps", "frameworks", "test_host"],
    doc = """
This aspect walks the dependency graph through the dependency graph and collects all the sources and
headers that are depended upon transitively. These files are needed to calculate test coverage on a
test run.

This aspect propagates a `_CoverageFilesInfo` provider.
""",
    implementation = _coverage_files_aspect_impl,
)

def _get_template_substitutions(
        *,
        test_bundle,
        test_bundle_dossier = None,
        test_coverage_manifest = None,
        test_env_inherit,
        test_environment,
        test_host_artifact = None,
        test_host_bundle_name = "",
        test_filter = None,
        test_host_dossier = None,
        test_type):
    """Dictionary with the substitutions to be applied to the template script.

    Args:
        test_bundle: File representing the test bundle's artifact that is with this rule.
        test_bundle_dossier: Optional. A File representing the dossier generated for the test
            bundle, if one exists.
        test_coverage_manifest: Optional. File representing the coverage manifest that is with
            this rule.
        test_env_inherit: List of environment variables to inherit from the external environment.
        test_environment: Dictionary representing the test environment for the current process
            running in the simulator.
        test_filter: Optional. The test filter received from the test rule implementation.
        test_host_artifact: Optional. A File representing the artifact found from the referenced
            test host rule if one was assigned.
        test_host_bundle_name: Optional. The bundle_name for the test host rule if one was assigned.
        test_host_dossier: Optional. A File representing the dossier generated for the test host, if
            one exists.
        test_type: String. The test type received from the test rule implementation.

    Returns:
        A new dictionary with substitutions suitable for ctx.actions.expand_template when applied to
        the rule's assigned test runner template.
    """
    substitutions = {
        "test_env_inherit": shell.array_literal(test_env_inherit),
        "test_bundle_path": test_bundle.short_path,
        "test_bundle_dossier_path": test_bundle_dossier.short_path if test_bundle_dossier else "",
        "test_coverage_manifest": test_coverage_manifest.short_path if test_coverage_manifest else "",
        "test_env": ",".join([k + "=" + v for (k, v) in test_environment.items()]),
        "test_filter": test_filter or "",
        "test_host_bundle_name": test_host_bundle_name,
        "test_host_path": test_host_artifact.short_path if test_host_artifact else "",
        "test_host_dossier_path": test_host_dossier.short_path if test_host_dossier else "",
        "test_type": test_type.upper(),
    }
    return {"%(" + k + ")s": substitutions[k] for k in substitutions}

def _get_coverage_execution_environment(*, covered_binaries):
    """Returns environment variables required for test coverage support.

    Args:
        covered_binaries: String. The `depset` of files representing the binaries that are being
            tested under a coverage run.

    Returns:
        A new dictionary with the required environment variables for retrieving Apple coverage
        information.
    """
    covered_binary_paths = [f.short_path for f in covered_binaries.to_list()]

    return {
        "APPLE_COVERAGE": "1",
        "TEST_BINARIES_FOR_LLVM_COV": ";".join(covered_binary_paths),
    }

def _get_main_thread_checker_test_environment(*, features):
    """Returns environment variables required for supporting Main Thread Checker during testing.

    Args:
        features: List of features enabled by the user. Typically from `ctx.features`.
    Returns:
        dict: A dictionary containing the required environment variables for enabling Main Thread Checker
        during testing.

    This function checks for the presence of the "apple.fail_on_main_thread_checker" feature in the list of features.
    If the feature is present, it returns a dictionary with the "MTC_CRASH_ON_REPORT" variable set to "1", enabling
    Main Thread Checker crash reporting, which may cause test execution to halt upon violations. If the feature is
    not present, an empty dictionary is returned, meaning that Main Thread Checker violations will still be reported,
    but they won't cause the test execution to crash.
    """

    if "apple.fail_on_main_thread_checker" in features:
        return {
            "MTC_CRASH_ON_REPORT": "1",
        }
    return {}

def _get_simulator_test_environment(
        *,
        command_line_test_env,
        features,
        rule_test_env,
        runner_test_env):
    """Returns the test environment for the current process running in the simulator

    All DYLD_INSERT_LIBRARIES key-value pairs are merged from the command-line, test
    rule and test runner.

    Args:
        command_line_test_env: Dictionary of fhe environment variables retrieved from the test
            invocation's command line arguments.
        features: List of features enabled by the user. Typically from `ctx.features`.
        rule_test_env: Dictionary of the environment variables retrieved from the test rule's
            attributes.
        runner_test_env: Dictionary of the environment variables retrieved from the assigned test
            runner.

    Returns:
        A new dictionary referencing the environment variables above, adjusted to handle additional
        information such as required DYLD_INSERT_LIBRARIES instructions.
    """

    # Get mutable copies of the different test environment dicts.
    command_line_test_env_copy = dicts.add(command_line_test_env)
    rule_test_env_copy = dicts.add(rule_test_env)
    runner_test_env_copy = dicts.add(runner_test_env)

    # Combine all DYLD_INSERT_LIBRARIES values in a list ordered as per the source:
    # 1. Command line test-env
    # 2. Test Rule test-env
    # 3. Test Runner test-env
    insert_libraries_values = []
    command_line_values = command_line_test_env_copy.pop(_INSERT_LIBRARIES_KEY, default = None)
    if command_line_values:
        insert_libraries_values.append(command_line_values)
    rule_values = rule_test_env_copy.pop(_INSERT_LIBRARIES_KEY, default = None)
    if rule_values:
        insert_libraries_values.append(rule_values)
    runner_values = runner_test_env_copy.pop(_INSERT_LIBRARIES_KEY, default = None)
    if runner_values:
        insert_libraries_values.append(runner_values)

    # Combine all DYLD_INSERT_LIBRARIES values in a single string separated by ":" and then save it
    # to a dict to be combined with other test_env pairs.
    insert_libraries_values_joined = ":".join(insert_libraries_values)
    test_env_dyld_insert_pairs = {}
    if insert_libraries_values_joined:
        test_env_dyld_insert_pairs = {_INSERT_LIBRARIES_KEY: insert_libraries_values_joined}

    # Combine all the environments with the DYLD_INSERT_LIBRARIES values merged together.
    return dicts.add(
        command_line_test_env_copy,
        rule_test_env_copy,
        runner_test_env_copy,
        test_env_dyld_insert_pairs,
        _get_main_thread_checker_test_environment(features = features),
    )

def _apple_test_rule_impl(*, ctx, requires_dossiers, test_type):
    """Generates an implementation for the Apple test rules, given arguments.

    Args:
        ctx: A rule context.
        requires_dossiers: A Boolean to indicate if the test rule depends on dossiers from the test
            bundle and the optional test host.
        test_type: A String indicating the test type. For example, "xctest" or "xcuitest".

    Returns:
        A full set of providers required to define an Apple test rule.
    """
    runner_attr = ctx.attr.runner
    runner_info = runner_attr[AppleTestRunnerInfo]
    execution_requirements = getattr(runner_info, "execution_requirements", {})

    # ctx.expand_make_variables is marked deprecated in the docs but every ruleset uses it. Not
    # sure how they're planning on getting rid of it for good.
    rule_test_env = {
        k: ctx.expand_make_variables("env", v, {})
        for k, v in ctx.attr.env.items()
    }

    test_bundle_target = ctx.attr.deps[0]
    test_bundle = test_bundle_target[AppleTestInfo].test_bundle

    direct_runfiles = [test_bundle]
    transitive_runfiles = [test_bundle_target[DefaultInfo].default_runfiles.files]

    test_bundle_dossier = None
    if requires_dossiers and AppleCodesigningDossierInfo in test_bundle_target:
        test_bundle_dossier = test_bundle_target[AppleCodesigningDossierInfo].dossier
        if test_bundle_dossier:
            direct_runfiles.append(test_bundle_dossier)

    # Environment variables to be set as the %(test_env)s substitution, which includes the
    # --test_env and env attribute values, but not the execution environment variables.
    test_environment = _get_simulator_test_environment(
        command_line_test_env = ctx.configuration.test_env,
        features = ctx.features,
        rule_test_env = rule_test_env,
        runner_test_env = getattr(runner_info, "test_environment", {}),
    )

    # Environment variables for the Bazel test action itself.
    execution_environment = dict(getattr(runner_info, "execution_environment", {}))

    # Include the test rule env in the execution environment, to match
    # `--test_env` behavior
    execution_environment = dicts.add(
        execution_environment,
        rule_test_env,
    )

    # Bundle name of the app under test (test host) if given
    test_host_bundle_name = ""
    test_host_dossier = None

    test_host_attr = ctx.attr.test_host
    if test_host_attr:
        if AppleBundleInfo in test_host_attr:
            test_host_bundle_name = test_host_attr[AppleBundleInfo].bundle_name
        if requires_dossiers and AppleCodesigningDossierInfo in test_host_attr:
            test_host_dossier = test_host_attr[AppleCodesigningDossierInfo].dossier
            if test_host_dossier:
                direct_runfiles.append(test_host_dossier)

    test_host_artifact = test_bundle_target[AppleTestInfo].test_host
    if test_host_artifact:
        direct_runfiles.append(test_host_artifact)

    test_coverage_manifest = None
    if ctx.configuration.coverage_enabled and ctx.attr.collect_code_coverage:
        test_coverage_manifest = ctx.file.test_coverage_manifest
        if test_coverage_manifest:
            direct_runfiles.append(test_coverage_manifest)

        apple_coverage_support_files = ctx.attr._apple_coverage_support.files
        covered_binaries = test_bundle_target[_CoverageFilesInfo].covered_binaries

        execution_environment = dicts.add(
            execution_environment,
            _get_coverage_execution_environment(
                covered_binaries = covered_binaries,
            ),
        )

        transitive_runfiles.extend([
            apple_coverage_support_files,
            covered_binaries,
            test_bundle_target[_CoverageFilesInfo].coverage_files,
        ])

    executable = ctx.actions.declare_file("%s" % ctx.label.name)
    ctx.actions.expand_template(
        template = runner_info.test_runner_template,
        output = executable,
        substitutions = _get_template_substitutions(
            test_bundle = test_bundle,
            test_bundle_dossier = test_bundle_dossier,
            test_coverage_manifest = test_coverage_manifest,
            test_env_inherit = ctx.attr.env_inherit,
            test_environment = test_environment,
            test_filter = ctx.attr.test_filter,
            test_host_artifact = test_host_artifact,
            test_host_bundle_name = test_host_bundle_name,
            test_host_dossier = test_host_dossier,
            test_type = test_type,
        ),
        is_executable = True,
    )

    transitive_runfile_objects = [
        runner_attr.default_runfiles,
        runner_attr.data_runfiles,
    ]

    # Add required data into the runfiles to make it available during test
    # execution.
    for data_dep in ctx.attr.data:
        transitive_runfiles.append(data_dep.files)
        transitive_runfile_objects.append(data_dep.default_runfiles)

    return [
        # Repropagate the AppleBundleInfo and AppleTestInfo providers from the test bundle so that
        # clients interacting with the test targets themselves can access the bundle's structure.
        test_bundle_target[AppleBundleInfo],
        test_bundle_target[AppleDsymBundleInfo],
        test_bundle_target[AppleTestInfo],
        test_bundle_target[OutputGroupInfo],
        coverage_common.instrumented_files_info(
            ctx,
            dependency_attributes = ["deps"],
        ),
        testing.ExecutionInfo(execution_requirements),
        testing.TestEnvironment(execution_environment, ctx.attr.env_inherit),
        DefaultInfo(
            executable = executable,
            files = depset(
                [executable, test_bundle],
                transitive = [test_bundle_target[AppleExtraOutputsInfo].files],
            ),
            runfiles = ctx.runfiles(
                files = direct_runfiles,
                transitive_files = depset(transitive = transitive_runfiles),
            ).merge_all(transitive_runfile_objects),
        ),
    ]

apple_test_rule_support = struct(
    apple_test_rule_impl = _apple_test_rule_impl,
)
