# Copyright 2026 Google LLC.
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

"""Model-specific op resolvers for statically linked LiteRT CompiledModel."""

# copybara:uncomment_begin(google-only)
# load("//devtools/build_cleaner/skylark:build_defs.bzl", "register_extension_info")
# copybara:uncomment_end

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "gen_selected_ops")

_OP_RESOLVER_FLAG = "//litert/build_common:op_resolver"

# Header compiled into generated registrations in place of the TFLite headers
# that gen_selected_ops() emits, some of which have restricted visibility.
_REGISTRATION_HEADER = Label("//litert/runtime:selected_op_resolver.h")

def litert_selected_op_resolver(name, models, deps = [], **kwargs):
    """Generates a resolver containing the operators and versions in models.

    Select it by pointing the `op_resolver` flag of
    `//litert/build_common` at the generated target.
    The application continues to use the public CompiledModel C++ API; it does
    not need to include TFLite headers or maintain a handwritten operator list.
    Rebuilding after a model change regenerates the registration automatically.

    Args:
      name: The resolver cc_library target name.
      models: Nonempty list of .tflite model labels (the union is registered).
      deps: Libraries implementing any custom operators in the models.
      **kwargs: Additional cc_library attributes, such as visibility.
    """
    if not models:
        fail("models must contain at least one .tflite model")
    testonly = kwargs.get("testonly", False)
    gen_selected_ops(
        name = name + "_tflite",
        model = models,
        namespace = "litert::internal",
        testonly = testonly,
    )

    # The facade header re-exports every declaration the registration uses, so
    # applications only depend on LiteRT.
    native.genrule(
        name = name + "_registration",
        srcs = [":" + name + "_tflite"],
        outs = [name + "_registration.cc"],
        cmd = "{ echo '#include \"%s/%s\"'; sed -e '/^#include /d' $(SRCS); } > $@" % (
            _REGISTRATION_HEADER.package,
            _REGISTRATION_HEADER.name,
        ),
        testonly = testonly,
    )
    cc_library(
        name = name,
        srcs = [":" + name + "_registration"],
        deps = ["//litert/runtime:selected_op_resolver"] + deps,
        **kwargs
    )

def _op_resolver_transition_impl(_settings, attr):
    return {_OP_RESOLVER_FLAG: str(attr.op_resolver)}

_op_resolver_transition = transition(
    implementation = _op_resolver_transition_impl,
    inputs = [],
    outputs = [_OP_RESOLVER_FLAG],
)

def _selected_op_resolver_test_impl(ctx):
    target = ctx.attr.test[0]
    default_info = target[DefaultInfo]

    new_executable = ctx.actions.declare_file(ctx.attr.name)
    ctx.actions.symlink(
        output = new_executable,
        target_file = default_info.files_to_run.executable,
        is_executable = True,
    )

    providers = [
        DefaultInfo(
            files = depset(direct = [new_executable], transitive = [default_info.files]),
            runfiles = default_info.default_runfiles.merge(ctx.runfiles([new_executable])),
            executable = new_executable,
        ),
    ]
    if RunEnvironmentInfo in target:
        providers.append(target[RunEnvironmentInfo])
    if testing.ExecutionInfo in target:
        # Keeps execution requirements such as "requires-darwin", which route
        # tests built for Apple platforms to macOS executors.
        providers.append(target[testing.ExecutionInfo])
    return providers

_selected_op_resolver_test = rule(
    implementation = _selected_op_resolver_test_impl,
    doc = "Runs `test` built with `op_resolver` as the CompiledModel op resolver.",
    attrs = {
        "test": attr.label(
            mandatory = True,
            executable = True,
            cfg = _op_resolver_transition,
        ),
        "op_resolver": attr.label(mandatory = True),
    },
    test = True,
)

def litert_selected_op_resolver_test(name, op_resolver, **kwargs):
    """A cc_test whose CompiledModel is linked against a specific op resolver.

    The resolver is selected through a configuration transition, so the test
    does not depend on the `op_resolver` flag being passed on the command line
    and can run with default build settings, e.g. in continuous integration.
    The underlying cc_test is named `<name>_basic` and tagged manual.

    Args:
      name: The test target name.
      op_resolver: Label of a litert_selected_op_resolver() target.
      **kwargs: cc_test attributes (srcs, deps, data, size, ...).
    """
    test_kwargs = {}
    for attr in ["args", "flaky", "shard_count", "size", "timeout", "visibility"]:
        if attr in kwargs:
            test_kwargs[attr] = kwargs[attr]
    tags = kwargs.pop("tags", [])
    cc_test(
        name = name + "_basic",
        tags = tags + ["manual", "notap"],
        **kwargs
    )
    _selected_op_resolver_test(
        name = name,
        test = ":" + name + "_basic",
        op_resolver = op_resolver,
        tags = tags,
        **test_kwargs
    )

# copybara:uncomment_begin(google-only)
# register_extension_info(
#     extension = litert_selected_op_resolver_test,
#     label_regex_for_dep = "{extension_name}_basic",
# )
# copybara:uncomment_end
