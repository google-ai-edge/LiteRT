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

"""AppIntentsInfo provider implementation for AppIntents support for Apple rules."""

AppIntentsInfo = provider(
    doc = "Private provider to propagate source files required by AppIntents processing.",
    fields = {
        "intent_module_names": """
A List with the module names where App Intents are expected to be found.""",
        "swift_source_files": """
A List with the swift source Files to handle via app intents processing.""",
        "swiftconstvalues_files": """
A List with the swiftconstvalues Files to handle via app intents processing for Xcode 15+.""",
    },
)
