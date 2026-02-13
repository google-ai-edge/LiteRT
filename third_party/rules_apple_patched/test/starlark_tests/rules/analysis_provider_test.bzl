# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Starlark test macro to create `provider_test`-like rules."""

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
)

visibility("//test/starlark_tests/...")

def _make_provider_test_impl(provider, provider_fn, assertion_fn):
    def _provider_test_impl(ctx):
        env = analysistest.begin(ctx)
        target_under_test = analysistest.target_under_test(env)

        if provider not in target_under_test:
            fail("Provider %s not found in target: %s" % (provider, target_under_test))

        provider_info = target_under_test[provider]
        provider_data = provider_fn(ctx, provider_info)
        assertion_fn(ctx, env, provider_data)
        return analysistest.end(env)

    return _provider_test_impl

def make_provider_test_rule(
        *,
        provider,
        provider_fn = lambda _ctx, p: p,
        assertion_fn,
        attrs,
        config_settings = {}):
    """Returns a new `provider_test`-like rule for a specific provider with custom settings.

    Args:
        provider: Provider to key from target_under_test.
        provider_fn: Callable to get information from provider, this callable should intake
            two parameters:
            - ctx: Rule context.
            - provider: Provider keyed from target_under_test (ie. `target_under_test[provider]`).
        assertion_fn: Callable to perform assertions against the provider info.
            This callable should conform to the following signature `callable(ctx, env, provider)`:
            - ctx: Rule context.
            - env: analysistest environment (ie. `analysistest.begin()`)
        attrs: Dictionary of rule attrs for the analysis phase test.
        config_settings: A dictionary of configuration settings and their values
            that should be applied during tests.

    Returns:
        A rule returned by `analysistest.make` that has the `provider_test` interface with the
        given attrs, and config settings.
    """
    return analysistest.make(
        _make_provider_test_impl(
            provider = provider,
            provider_fn = provider_fn,
            assertion_fn = assertion_fn,
        ),
        attrs = attrs,
        config_settings = config_settings,
    )
