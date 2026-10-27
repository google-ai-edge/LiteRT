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
"""Queries specification candidates; never claims driver or lowering support."""

import argparse
import json
from pathlib import Path

DATABASE = Path(__file__).with_name("tosa_001000_1.json")


def requirements_met(requirements, capabilities):
    return any(
        set(clause["all_of"]).issubset(capabilities)
        for clause in requirements["any_of"])


def find_variants(database, op_name, types=None, capabilities=None, enums=None):
    """Filters table rows only; prose, shapes, level and device limits still apply.

  `types` names columns from the registry Type support table. Attribute enum
  requirements are ANDed with row requirements. A missing row is never treated as
  support, and unrecognized enum values/names are errors.
  """
    op = next(
        (o for o in database["operators"] if o["name"] == op_name.upper()),
        None)
    if op is None:
        raise ValueError(f"Unknown mapped operator: {op_name}")
    types = types or {}
    enums = enums or {}
    known_columns = {key for row in op["variants"] for key in row["types"]}
    if set(types) - known_columns:
        raise ValueError(
            f"Type filter must name a table column: {sorted(known_columns)}")
    all_elements = {
        v["element_type"]
        for o in database["operators"]
        for row in o["variants"]
        for v in row["types"].values()
    }
    if set(types.values()) - all_elements:
        raise ValueError(
            f"Unknown element type; use TOSA names: {sorted(all_elements)}")
    if capabilities is not None:
        known_caps = set(database["profiles"]) | set(database["extensions"])
        if set(capabilities) - known_caps:
            raise ValueError(
                "Profile/extension is absent from this SPIR-V mapping")
    enum_constraints = []
    for argument_name, value in enums.items():
        arg = next((a for a in op["arguments"] if a["name"] == argument_name),
                   None)
        if arg is None or arg["tosa_type"] not in database["enums"]:
            raise ValueError(f"Not an enumerated argument: {argument_name}")
        definition = database["enums"][arg["tosa_type"]]
        enum = next((e for e in definition["values"] if e["name"] == value),
                    None)
        if enum is None:
            raise ValueError(f"Unknown {arg['tosa_type']} value: {value}")
        enum_constraints.append(enum["additional_requirements"])
    found = []
    for row in op["variants"]:
        if any(row["types"].get(k, {}).get("element_type") != v
               for k, v in types.items()):
            continue
        if any(k in row["parameters"] and row["parameters"][k] != v
               for k, v in enums.items()):
            continue
        if capabilities is not None and not all(
                requirements_met(req, capabilities)
                for req in [row["requirements"], *enum_constraints]):
            continue
        found.append(row)
    return op, found


def assignment(value):
    parts = value.split("=", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("Expected argument=value")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--op")
    parser.add_argument("--type", action="append", type=assignment, default=[])
    parser.add_argument("--enum", action="append", type=assignment, default=[])
    parser.add_argument("--capability",
                        action="append",
                        help="Repeat for every available profile/extension")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    database = json.loads(args.database.read_text())
    if args.op is None:
        if args.type or args.enum or args.capability:
            parser.error("Filters require --op")
        print(
            json.dumps(
                {
                    "instruction_set": database["instruction_set"],
                    **database["counts"], "status": database["status"]
                },
                indent=2))
        return
    try:
        op, rows = find_variants(database, args.op, dict(args.type),
                                 args.capability, dict(args.enum))
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps({**op, "variants": rows}, indent=2))
        return
    print(
        f"{op['name']}: {len(rows)} specification candidates; not a support decision"
    )
    for row in rows:
        types = ", ".join(
            f"{k}={v['element_type']}" for k, v in row["types"].items())
        params = ", ".join(f"{k}={v}" for k, v in row["parameters"].items())
        print(f"  {row['id']}: {types}{'; ' + params if params else ''}")
        print(f"    requirements: {json.dumps(row['requirements'])}")
    print("Shapes, constants and limits:")
    for arg in op["arguments"]:
        print(
            f"  {arg['name']}: {arg['logical_shape']}, rank={arg['rank_range']}, "
            f"constant={arg['constant_required']}, limits={arg['level_limits']}"
        )
    print(
        f"Context-dependent validation expressions: {len(op['validation_expressions'])}"
    )
    print(f"Registry: {op['sources']['registry']}")


if __name__ == "__main__":
    main()
