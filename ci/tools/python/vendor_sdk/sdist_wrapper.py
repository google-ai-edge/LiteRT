# Copyright 2025 The AI Edge LiteRT Authors.
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

"""Wrapper to create sdist using setup.py."""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import glob


def run_command(cmd):
    """Run subprocess command and handle errors."""
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running command", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)


def prepare_setup(setup_template, dest_setup, project_name, version, nightly):
    """Generate setup.py from template."""
    content = setup_template.read_text()

    package_name = f"{project_name}_nightly" if nightly else project_name

    content = content.replace("{{ PACKAGE_NAME }}", package_name)
    content = content.replace("{{ PACKAGE_VERSION }}", version)

    dest_setup.write_text(content)


def prepare_init(src_init, dest_init, version):
    """Generate __init__.py with version."""
    content = src_init.read_text()
    content = content.replace("{{ PACKAGE_VERSION }}", version)
    dest_init.write_text(content)


def find_sdist(dist_dir):
    """Find generated tar.gz file."""
    files = glob.glob(str(dist_dir / "*.tar.gz"))

    if not files:
        raise FileNotFoundError("No .tar.gz file generated")

    if len(files) > 1:
        print("Warning: multiple archives found, using first")

    return Path(files[0])


def main():
    parser = argparse.ArgumentParser(description="Create sdist using setup.py")

    parser.add_argument("--project_name", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--setup_py", default="setup.template.py")
    parser.add_argument("--output_sdist_path", required=True)
    parser.add_argument("--nightly_suffix")
    parser.add_argument("--version", required=True)

    args = parser.parse_args()

    original_dir = Path.cwd()
    project_dir = Path(args.dir).resolve()

    build_dir = original_dir / "sdist_build"
    dist_temp = build_dir / "dist_sdist_temp"

    build_dir.mkdir(exist_ok=True)

    try:
        print("Switching to project directory:", project_dir)
        os.chdir(project_dir)

        tmp_setup = build_dir / "setup.py"
        prepare_setup(
            project_dir / args.setup_py,
            tmp_setup,
            args.project_name,
            args.version,
            args.nightly_suffix,
        )

        shutil.copy("MANIFEST.in", build_dir)

        package_dir = build_dir / args.project_name
        package_dir.mkdir(exist_ok=True)

        prepare_init(
            project_dir / args.project_name / "__init__.py",
            package_dir / "__init__.py",
            args.version,
        )

        print("Switching to build directory:", build_dir)
        os.chdir(build_dir)

        dist_temp.mkdir(exist_ok=True)

        py_version = os.environ.get("HERMETIC_PYTHON_VERSION")
        python_exec = f"python{py_version}" if py_version else sys.executable

        cmd = [
            python_exec,
            str(tmp_setup),
            "sdist",
            "--dist-dir",
            str(dist_temp),
        ]

        run_command(cmd)

        archive = find_sdist(dist_temp)

        final_path = original_dir / args.output_sdist_path
        final_path.parent.mkdir(parents=True, exist_ok=True)

        print("Moving archive:", archive, "->", final_path)
        shutil.move(str(archive), str(final_path))

        print("Successfully created:", final_path)

    finally:
        os.chdir(original_dir)
        if build_dir.exists():
            shutil.rmtree(build_dir)


if __name__ == "__main__":
    main()
