# Copyright 2026 The LiteRT Authors. All Rights Reserved.
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
"""Join all specification operators with source evidence, not device claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
COMPILER = HERE.parents[1] / "compiler"


def key(name):
    return name.replace("_", "").lower()


def audit(database, compiler, tfsc):
    sources = {p.name: p.read_text() for p in sorted(compiler.glob("*.cc"))
               if not p.name.endswith(("_test.cc", "_fixture.cc"))}
    headers = [tfsc / "include/tosa/OperatorEnum.hpp",
               tfsc / "include/tosa/GraphOperatorHelpers.hpp"]
    header_text = "\n".join(p.read_text() for p in headers)
    enums = {key(name) for name in re.findall(r"OperatorEnum::(\w+)", header_text)}
    known = {key(op["name"]) for op in database["operators"]}
    references = {}
    for filename, source in sources.items():
        for number, line in enumerate(source.splitlines(), 1):
            names = re.findall(r"Add(\w+)Operator\s*\(|OperatorEnum::(\w+)", line)
            for helper, enum in names:
                name = key(helper or enum)
                if name not in known:
                    raise ValueError(f"Unmapped TFSC operator {helper or enum}")
                references.setdefault(name, []).append(f"compiler/{filename}:{number}")
    return {
        "instruction_set": database["instruction_set"],
        "scope": "Source references are not per-type qualification or device support",
        "source_sha256": {
            **{f"compiler/{name}": hashlib.sha256(text.encode()).hexdigest()
               for name, text in sources.items()},
            **{str(p.relative_to(tfsc)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in headers}},
        "operators": [{
            "name": op["name"],
            "type_combinations": len(op["variants"]),
            "tfsc_helper": "present" if key(op["name"]) in enums else "absent",
            "lowering": "referenced" if key(op["name"]) in references else "not_referenced",
            "references": sorted(set(references.get(key(op["name"]), []))),
            "device_support": "query_required",
        } for op in database["operators"]],
    }


def markdown(report):
    ops = report["operators"]
    used = sum(op["lowering"] == "referenced" for op in ops)
    lines = ["# TOSA lowering audit", "",
             f"{len(ops)} specification operators; {used} referenced by production lowering.",
             "", report["scope"] + ". All device support remains query-dependent.",
             "TFSC helper presence does not establish support for every type row.", "",
             "| TOSA operator | Type rows | TFSC helper | Lowering source |",
             "| --- | ---: | --- | --- |"]
    for op in ops:
        refs = ", ".join(op["references"]) or "Not referenced"
        lines.append(f"| {op['name']} | {op['type_combinations']} | {op['tfsc_helper']} | {refs} |")
    lines.extend(["", "Source SHA-256 fingerprints:", ""])
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in report["source_sha256"].items())
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfsc", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    database = json.loads((HERE / "tosa_001000_1.json").read_text())
    report = audit(database, COMPILER, args.tfsc)
    text = json.dumps(report, indent=2) + "\n" if args.json else markdown(report)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
