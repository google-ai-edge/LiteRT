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
"""Extracts reference data, not a device-support or delegation predicate.

Reads pinned Git objects, without changing either source checkout. The SPIR-V
registry determines the supported instructions and type rows. TOSA 1.0.1 adds
logical shapes, rank/level limits, numerical semantics and validation references.
No TOSA-only operator or extension is implicitly admitted into this mapping.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

REGISTRY_REVISION = "3b5c31fdd42ed166ef174bd7f1b3613672b6cc46"
TOSA_REVISION = "7010bd19cb1e5468f5a60f578b961fc476bd6ae3"
REGISTRY_PATH = "extended/TOSA.001000.1.html"
INSTRUCTION_SET = "TOSA.001000.1"
HERE = Path(__file__).resolve().parent


def normalized(text):
    return " ".join(text.split())


class GitSource:
    """Content and provenance from a committed snapshot (ignores working edits)."""

    def __init__(self, path, revision, repository):
        self.path = Path(path)
        self.repository = repository
        self.revision = self.git("rev-parse", f"{revision}^{{commit}}").strip()
        self.files = {}

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.path), *args], text=True)

    def read(self, path):
        # Hash bytes, not platform-dependent newline translations.
        data = subprocess.check_output(
            ["git", "-C",
             str(self.path), "show", f"{self.revision}:{path}"])
        self.files[path] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")

    def url(self, path):
        return f"{self.repository}/blob/{self.revision}/{path}"

    def metadata(self):
        return {
            "repository": self.repository,
            "revision": self.revision,
            "files_sha256": dict(sorted(self.files.items())),
        }


def requirements(expression, extensions):
    """Comma means OR; 'and' means AND. Extensions also need a base profile."""
    alternatives = []
    for alternative in expression.split(","):
        names = [v.strip() for v in alternative.split(" and ")]
        if not all(re.fullmatch(r"(?:PRO|EXT)-[A-Z0-9]+", v) for v in names):
            raise ValueError(
                f"Unrecognized requirement expression: {expression!r}")
        clauses = [names]
        for name in names:
            if name.startswith("EXT-"):
                bases = extensions[name]["requires_any_profile"]
                clauses = [c + [base] for c in clauses for base in bases]
            elif name not in ("PRO-INT", "PRO-FP"):
                raise ValueError(f"Unknown profile {name}")
        alternatives.extend({"all_of": sorted(set(c))} for c in clauses)
    return {"any_of": alternatives}


def scalar_type(instruction):
    """Keep TOSA semantic width distinct from SPIR-V storage width."""
    table = {
        "OpTypeBool": ("bool_t", None),
        "OpTypeInt 8 0": ("i8_t", 8),
        "OpTypeInt 16 0": ("i16_t", 16),
        "OpTypeInt 32 0": ("i32_t", 32),
        "OpTypeInt 64 0": ("i48_t", 64),
        "OpTypeFloat 16": ("fp16_t", 16),
        "OpTypeFloat 32": ("fp32_t", 32),
        "OpTypeFloat 16 BFloat16KHR": ("bf16_t", 16),
        "OpTypeFloat 8 Float8E4M3EXT": ("fp8e4m3_t", 8),
        "OpTypeFloat 8 Float8E5M2EXT": ("fp8e5m2_t", 8),
    }
    semantic, bits = table[instruction]
    return {
        "element_type": semantic,
        "spirv_scalar": instruction,
        "storage_bits": bits
    }


def type_definitions(code):
    definitions = {}
    for line in code.splitlines():
        name, instruction = line.split(" = ", 1)
        if instruction.startswith("OpTypeTensorARM"):
            match = re.fullmatch(r"OpTypeTensorARM (%\S+) (\S+)(?: (\[.*\]))?",
                                 instruction)
            if not match:
                raise ValueError(f"Unknown tensor definition: {line}")
            scalar, rank, shape = match.groups()
            definitions[name] = {
                **definitions[scalar],
                "rank": rank,
                "shape": shape,
                "registry_type_id": name,
            }
        else:
            definitions[name] = scalar_type(instruction)
    return definitions


def registry_arguments(encoding_table):
    result = {}
    body = encoding_table.find("tr").find("td")
    for paragraph in body.select("div.paragraph"):
        first = paragraph.find("em")
        if first is None:
            continue
        name = first.get_text(strip=True)
        text = paragraph.get_text(" ", strip=True)
        code = paragraph.find_next_sibling("div", class_="listingblock")
        if code is None:
            raise ValueError(f"Missing SPIR-V argument type for {name}")
        result[name] = {
            "constant_required":
                "must come from a constant instruction" in text,
            "mapping_description":
                normalized(text),
            "spirv_type_template":
                code.get_text("\n", strip=True),
        }
    return result


def semantic_arguments(xml_op, mapping_args):
    args = []
    for arg in xml_op.findall("arguments/argument"):
        name = arg.get("name")
        entry = {
            "name":
                name,
            "category":
                arg.get("category"),
            "tosa_type":
                arg.get("type"),
            "element_type_parameter":
                arg.get("tensor-element-type"),
            "logical_shape":
                arg.get("shape"),
            "description":
                normalized(arg.findtext("description")),
            "optional_in_tosa":
                arg.get("optional") == "true",
            "rank_range":
                dict(arg.find("rank").attrib)
                if arg.find("rank") is not None else None,
            "level_limits": [
                dict(limit.attrib) for limit in arg.findall("levellimit")
            ],
        }
        if name in mapping_args:
            entry.update(mapping_args[name])
        elif arg.get("category") == "output":
            # FFT2D/RFFT2D have two output tensors in one OpTypeStruct result.
            entry.update(constant_required=False,
                         spirv_type_template=None,
                         mapping_description="Member of the composite result")
        else:
            raise ValueError(f"Missing registry input/attribute: {name}")
        entry["fixed_element_type"] = None
        for line in (entry["spirv_type_template"] or "").splitlines():
            instruction = line.split(" = ")[-1]
            if instruction.startswith(
                ("OpTypeInt ", "OpTypeFloat ", "OpTypeBool")):
                entry["fixed_element_type"] = scalar_type(instruction)
        args.append(entry)
    if set(mapping_args) - {a["name"] for a in args}:
        raise ValueError("Unmatched registry arguments")
    return args


def semantic_bindings(xml_op, types, parameters):
    """Find TOSA type rows consistent with a registry row; preserve alternatives."""
    args = {a.get("name"): a for a in xml_op.findall("arguments/argument")}
    bound = {}
    for name, tensor in types.items():
        variable = args[name].get("tensor-element-type")
        if variable == "-":
            variable = args[name].get("type")
        value = tensor["element_type"]
        if variable in bound and bound[variable] != value:
            raise ValueError(f"Conflicting type variable {variable}")
        bound[variable] = value
    if "acc_type" in parameters:
        bound["acc_t"] = {
            "INT32": "i32_t",
            "INT48": "i48_t",
            "FP16": "fp16_t",
            "FP32": "fp32_t"
        }[parameters["acc_type"]]
    if "TABLE_SIZE" in parameters:
        bound["TABLE_SIZE"] = parameters["TABLE_SIZE"]
    candidates = []
    for row in xml_op.findall("typesupport"):
        if all(row.get(k, k) == v for k, v in bound.items()):
            candidates.append({
                k: v
                for k, v in row.attrib.items()
                if k not in ("mode", "version_added")
            })
    if not candidates:
        raise ValueError(
            f"{xml_op.findtext('name')}: no TOSA counterpart: {bound}")
    return candidates


def validation_references(code, url):
    """Index assertions without pretending they are context-free predicates.

  The containing algorithm, guards, loops and helper calls still apply. These
  expressions MUST NOT be eval'd or used as an automatic delegation decision.
  """
    rules = []
    for match in re.finditer(r"\b(ERROR_IF|REQUIRE|LEVEL_CHECK)\s*\(", code):
        start = match.end()
        cursor, depth = start, 1
        while depth and cursor < len(code):
            depth += (code[cursor] == "(") - (code[cursor] == ")")
            cursor += 1
        if depth or code[cursor:].lstrip()[:1] != ";":
            raise ValueError("Cannot parse specification assertion")
        line = code.count("\n", 0, match.start()) + 1
        rules.append({
            "kind": match.group(1),
            "expression": normalized(code[start:cursor - 1]),
            "interpretation": "requires_containing_pseudocode_context",
            "source": f"{url}#L{line}",
        })
    return rules


def prose_sections(tosa):
    sections = {}
    paths = tosa.git("ls-tree", "-r", "--name-only", tosa.revision,
                     "chapters").splitlines()
    for path in paths:
        if not path.endswith(".adoc"):
            continue
        content = tosa.read(path)
        matches = list(
            re.finditer(r"^====[ \t]+([A-Z][A-Z0-9_]*)[ \t]*$", content, re.M))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(
                matches) else len(content)
            body = content[match.end():end]
            # Generated tables and algorithm includes are represented separately.
            body = re.sub(r"include::[^\n]*|\[source,c\+\+\]|^----$",
                          "",
                          body,
                          flags=re.M)
            sections[match.group(1)] = {
                "text_asciidoc":
                    body.strip(),
                "source":
                    f"{tosa.url(path)}#L{content.count(chr(10), 0, match.start()) + 1}",
                "interpretation":
                    "normative_prose_not_machine_evaluated",
            }
    return sections


def extract(registry, tosa):
    soup = BeautifulSoup(registry.read(REGISTRY_PATH), "html.parser")
    xml = ET.fromstring(tosa.read("tosa.xml"))
    if dict(xml.find("version").attrib) != {
            "major": "1",
            "minor": "0",
            "patch": "1",
            "draft": "false"
    }:
        raise ValueError("This mapping must be supplemented with TOSA 1.0.1")
    supported = soup.find(id="_profiles_and_extensions").parent
    profile_names = re.findall(r"\bPRO-[A-Z0-9]+", supported.get_text())
    extension_names = re.findall(r"\bEXT-[A-Z0-9]+", supported.get_text())
    all_extensions = {
        e.get("name"): {
            "requires_any_profile": [
                p.text for p in e.findall("profile_supported")
            ],
            "status_in_tosa": e.get("status"),
        } for e in xml.findall("profile_extensions/profile_extension")
    }
    extensions = {k: all_extensions[k] for k in extension_names}
    enums = {}
    for heading in soup.select('h3[id^="enum-"]'):
        rows = heading.parent.find("table").find_all("tr")
        values = []
        for row in rows[1:]:
            cells = [
                c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])
            ]
            if len(cells) != 4:
                raise ValueError("Unknown enumeration table")
            values.append({
                "name":
                    cells[0],
                "value":
                    int(cells[1]),
                "description":
                    cells[2],
                "additional_requirements":
                    requirements(cells[3], extensions) if cells[3] else {
                        "any_of": [{
                            "all_of": []
                        }]
                    }
            })
        enums[heading.get_text(strip=True)] = {
            "values": values,
            "source": f"{registry.url(REGISTRY_PATH)}#{heading['id']}"
        }
    xml_ops = {op.findtext("name"): op for op in xml.findall(".//operator")}
    prose = prose_sections(tosa)
    operators = []
    for heading in soup.find(id="_operators").parent.select("h4"):
        name = heading.get_text(strip=True)
        tables = heading.parent.find_all("table", recursive=False)
        if len(tables) != 2:
            raise ValueError(f"Expected encoding and type tables for {name}")
        encoding = [
            c.get_text(" ", strip=True)
            for c in tables[0].find_all("tr")[-1].find_all("td")
        ]
        if encoding[1] != "12":
            raise ValueError("Expected OpExtInst encoding")
        xml_op = xml_ops[name]
        args = semantic_arguments(xml_op, registry_arguments(tables[0]))
        definitions = type_definitions(
            tables[1].find_previous("pre").get_text())
        rows = tables[1].find_all("tr")
        columns = [c.get_text(" ", strip=True) for c in rows[0].find_all("td")]
        if columns[0] != "Enabling profile/extension":
            raise ValueError("Unknown type table header")
        variants = []
        for index, row in enumerate(rows[1:]):
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if len(cells) != len(columns):
                raise ValueError("Ragged type table")
            types, parameters = {}, {}
            for column, cell in zip(columns[1:], cells[1:]):
                for operand in column.split(", "):
                    if cell.startswith("%"):
                        if "\nOpTypeTensorARM" in cell:
                            inline = cell.replace(
                                "\nOpTypeTensorARM",
                                "\n%inline = OpTypeTensorARM")
                            types[operand] = type_definitions(inline)["%inline"]
                        else:
                            types[operand] = definitions[cell]
                    else:
                        parameters[operand] = cell
            bindings = semantic_bindings(xml_op, types, parameters)
            variants.append({
                "id": f"{name.lower()}_{index:02d}",
                "registry_table_row": index + 1,
                "enabling_expression": cells[0],
                "requirements": requirements(cells[0], extensions),
                "types": types,
                "parameters": parameters,
                "tosa_type_bindings": bindings,
            })
        pseudocode_path = f"pseudocode/operators/{name}.tosac"
        code = tosa.read(pseudocode_path)
        operators.append({
            "name":
                name,
            "opcode":
                int(encoding[5]),
            "category":
                heading.parent.parent.find("h3").get_text(strip=True),
            "spirv_operand_order": [cell.split()[-1] for cell in encoding[6:]],
            "variadic_operands": [
                cell.split()[-1]
                for cell in encoding[6:]
                if cell.count("<id>") > 1
            ],
            "arguments":
                args,
            "variants":
                variants,
            "validation_expressions":
                validation_references(code, tosa.url(pseudocode_path)),
            "semantics":
                prose[name],
            "sources": {
                "registry": f"{registry.url(REGISTRY_PATH)}#{heading['id']}",
                "tosa_arguments": tosa.url("tosa.xml"),
                "tosa_pseudocode": tosa.url(pseudocode_path),
            },
            "coverage": {
                "constraint_evaluator": "not_implemented",
                "litert_lowering": "not_assessed",
                "tfsc_encoding": "not_assessed",
                "device_support": "not_assessed",
            },
        })
    return {
        "$schema":
            "tosa_capabilities.schema.json",
        "schema_version":
            1,
        "instruction_set":
            INSTRUCTION_SET,
        "status":
            "specification_reference_not_delegation_policy",
        "sources": {
            "spirv_registry": registry.metadata(),
            "tosa": tosa.metadata()
        },
        "profiles":
            profile_names,
        "extensions":
            extensions,
        "excluded_tosa_extensions":
            sorted(set(all_extensions) - set(extension_names)),
        "levels": {
            e.get("name"): {
                k.upper(): int(v) for k, v in e.attrib.items() if k != "name"
            } for e in xml.findall("levels/level")
        },
        "enums":
            enums,
        "global_constraints": {
            "tensor_rules":
                "https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_tensor_definitions",
            "helper_semantics":
                tosa.url("chapters/pseudocode.adoc"),
            "shape_contract":
                "fixed_shapes_per_compiled_graph_pipeline",
            "vulkan_requirements": [
                {
                    "vuid": "VUID-RuntimeSpirv-pNext-09919",
                    "rule": "Graph interface tensors must specify shape."
                },
                {
                    "vuid": "VUID-RuntimeSpirv-pNext-09920",
                    "rule": "Tensor constants must specify shape."
                },
                {
                    "vuid":
                        "VUID-RuntimeSpirv-pNext-09921",
                    "rule":
                        "Each tensor graph constant needs exactly one resource description with matching constant ID, rank and dimensions, and compatible format."
                },
                {
                    "vuid":
                        "VUID-RuntimeSpirv-pNext-09923",
                    "rule":
                        "Each tensor interface variable needs exactly one resource description with matching descriptor set/binding, arrayElement zero, matching rank/dimensions and compatible format."
                },
            ],
            "vulkan_source":
                "https://docs.vulkan.org/spec/latest/appendices/spirvenv.html",
            "vulkan_source_checked":
                "2026-10-26",
            "device_checks_required": [
                "queue_family_engine_and_instruction_set",
                "tosa_profiles_extensions_and_level",
                "tensor_limits",
                "tensor_format_tiling_and_usage",
                "capability_quality_flags",
            ],
        },
        "operators":
            operators,
        "counts": {
            "operators": len(operators),
            "type_combinations": sum(len(o["variants"]) for o in operators)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tosa-spec", type=Path, required=True)
    parser.add_argument("--registry-revision", default=REGISTRY_REVISION)
    parser.add_argument("--tosa-revision", default=TOSA_REVISION)
    parser.add_argument("--output",
                        type=Path,
                        default=HERE / "tosa_001000_1.json")
    parser.add_argument("--check",
                        action="store_true",
                        help="Check reproducibility without writing")
    args = parser.parse_args()
    registry = GitSource(args.registry, args.registry_revision,
                         "https://github.com/KhronosGroup/SPIRV-Registry")
    tosa = GitSource(args.tosa_spec, args.tosa_revision,
                     "https://github.com/arm/tosa-specification")
    database = extract(registry, tosa)
    content = json.dumps(database, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit(f"Outdated database: {args.output}")
    else:
        args.output.write_text(content)
    print(json.dumps(database["counts"]))


if __name__ == "__main__":
    main()
