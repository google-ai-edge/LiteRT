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
"""Checks capability data integrity and distinctions important to lowering."""

import copy
import json
from pathlib import Path
import unittest
import tempfile

import audit_lowering

import jsonschema

import capability_query as query
import extract_capabilities as extractor

HERE = Path(__file__).resolve().parent


class CapabilitiesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.database = json.loads((HERE / "tosa_001000_1.json").read_text())
        cls.schema = json.loads(
            (HERE / "tosa_capabilities.schema.json").read_text())
        cls.ops = {o["name"]: o for o in cls.database["operators"]}

    def test_source_audit_distinguishes_helper_reference_and_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "compiler"
            headers = root / "include/tosa"
            compiler.mkdir()
            headers.mkdir(parents=True)
            (compiler / "operation.cc").write_text("g.AddAvgPool2dOperator();")
            (compiler / "operation_test.cc").write_text("g.AddTanhOperator();")
            (headers / "OperatorEnum.hpp").write_text("")
            (headers / "GraphOperatorHelpers.hpp").write_text(
                "OperatorEnum::AvgPool2d; OperatorEnum::Tanh;")
            report = audit_lowering.audit(self.database, compiler, root)
            rows = {row["name"]: row for row in report["operators"]}
            self.assertEqual(len(rows), 66)
            self.assertEqual(rows["AVG_POOL2D"]["lowering"], "referenced")
            self.assertEqual(rows["TANH"]["tfsc_helper"], "present")
            self.assertEqual(rows["TANH"]["lowering"], "not_referenced")
            self.assertEqual(rows["CONV2D"]["tfsc_helper"], "absent")
            self.assertTrue(all(row["device_support"] == "query_required"
                                for row in rows.values()))

    def test_specification_does_not_claim_lowering_is_unimplemented(self):
        self.assertTrue(all(op["coverage"]["litert_lowering"] == "not_assessed"
                            for op in self.ops.values()))

    def rows(self, op, types=None, caps=None, enums=None):
        return query.find_variants(self.database, op, types, caps, enums)[1]

    def test_schema(self):
        jsonschema.Draft202012Validator.check_schema(self.schema)
        jsonschema.Draft202012Validator(self.schema).validate(self.database)

    def test_reference_cannot_silently_claim_device_support(self):
        bad = copy.deepcopy(self.database)
        bad["operators"][0]["coverage"]["device_support"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.schema).validate(bad)

    def test_complete_mapping_snapshot_and_unique_rows(self):
        self.assertEqual(self.database["counts"], {
            "operators": 66,
            "type_combinations": 350
        })
        self.assertEqual(len(self.ops), 66)
        self.assertEqual({o["opcode"] for o in self.ops.values()},
                         set(range(66)))
        ids = [r["id"] for o in self.ops.values() for r in o["variants"]]
        self.assertEqual(len(ids), 350)
        self.assertEqual(len(set(ids)), 350)

    def test_source_pins_and_no_absolute_checkout_requirement(self):
        self.assertEqual(self.database["sources"]["spirv_registry"]["revision"],
                         extractor.REGISTRY_REVISION)
        self.assertEqual(self.database["sources"]["tosa"]["revision"],
                         extractor.TOSA_REVISION)
        self.assertNotIn("/data/bt/", json.dumps(self.database))
        self.assertNotIn("/home/chasun/", json.dumps(self.database))
        for source in self.database["sources"].values():
            for filename, digest in source["files_sha256"].items():
                self.assertFalse(Path(filename).is_absolute())
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_profiles_are_alternatives_for_int32_add(self):
        for profile in ["PRO-INT", "PRO-FP"]:
            self.assertEqual(
                len(self.rows("ADD", {"input1": "i32_t"}, [profile])), 1)
        self.assertFalse(self.rows("ADD", {"input1": "i32_t"}, []))

    def test_extension_also_requires_base_profile(self):
        types = {"input1": "bf16_t"}
        self.assertFalse(self.rows("ADD", types, ["PRO-FP"]))
        self.assertFalse(self.rows("ADD", types, ["EXT-BF16"]))
        self.assertEqual(len(self.rows("ADD", types, ["PRO-FP", "EXT-BF16"])),
                         1)

    def test_cast_requires_both_extensions(self):
        types = {"input": "bf16_t", "output": "fp8e4m3_t"}
        for caps in [["PRO-FP", "EXT-BF16"], ["PRO-FP", "EXT-FP8E4M3"]]:
            self.assertFalse(self.rows("CAST", types, caps))
        self.assertEqual(
            len(self.rows("CAST", types,
                          ["PRO-FP", "EXT-BF16", "EXT-FP8E4M3"])), 1)

    def test_int8_is_not_an_add_type(self):
        self.assertFalse(self.rows("ADD", {"input1": "i8_t"}))

    def test_illegal_matmul_type_combination_not_cartesian_product(self):
        self.assertFalse(self.rows("MATMUL", {"A": "i8_t", "B": "fp32_t"}))
        row, = self.rows("MATMUL", {"A": "i8_t", "B": "i8_t"})
        self.assertEqual(row["types"]["output"]["element_type"], "i32_t")

    def test_int48_semantics_are_not_int64_support(self):
        row, = self.rows("MATMUL", {"A": "i16_t"})
        output = row["types"]["output"]
        self.assertEqual(output["element_type"], "i48_t")
        self.assertEqual(output["storage_bits"], 64)
        self.assertEqual(output["spirv_scalar"], "OpTypeInt 64 0")

    def test_convolution_shapes_bias_and_constants(self):
        args = {a["name"]: a for a in self.ops["CONV2D"]["arguments"]}
        self.assertEqual(args["input"]["logical_shape"], "[N,IH,IW,IC]")
        self.assertEqual(args["weight"]["logical_shape"], "[OC,KH,KW,IC]")
        self.assertEqual(args["output"]["logical_shape"], "[N,OH,OW,OC]")
        self.assertEqual(args["bias"]["logical_shape"], "[BC]")
        self.assertIn("BC == 1", args["bias"]["description"])
        for name in [
                "pad", "stride", "dilation", "input_zp", "weight_zp", "acc_type"
        ]:
            self.assertTrue(args[name]["constant_required"], name)
        self.assertFalse(args["weight"]["constant_required"])
        self.assertIn({
            "value": "dilation_y * KH",
            "limit": "MAX_KERNEL"
        }, args["weight"]["level_limits"])

    def test_accumulator_is_part_of_combination(self):
        rows = self.rows("CONV2D", {"input": "fp16_t", "output": "fp16_t"})
        self.assertEqual({r["parameters"]["acc_type"] for r in rows},
                         {"FP16", "FP32"})
        self.assertEqual(
            len(
                self.rows("CONV2D", {"input": "fp16_t"},
                          enums={"acc_type": "FP32"})), 1)
        self.assertFalse(
            self.rows("CONV2D", {"input": "fp16_t"},
                      enums={"acc_type": "INT32"}))

    def test_matmul_layout_and_zero_points(self):
        args = {a["name"]: a for a in self.ops["MATMUL"]["arguments"]}
        self.assertEqual(
            [args[k]["logical_shape"] for k in ["A", "B", "output"]],
            ["[N,H,C]", "[N,C,W]", "[N,H,W]"])
        self.assertTrue(args["A_zp"]["constant_required"])
        self.assertTrue(args["B_zp"]["constant_required"])

    def test_rounding_mode_adds_extension_requirement(self):
        types = {"input": "i32_t", "output": "i8_t"}
        mode = {"rounding_mode": "DOUBLE_ROUND"}
        self.assertFalse(self.rows("RESCALE", types, ["PRO-INT"], mode))
        self.assertEqual(
            len(
                self.rows("RESCALE", types, ["PRO-INT", "EXT-DOUBLEROUND"],
                          mode)), 1)
        self.assertEqual(
            len(
                self.rows("RESCALE", types, ["PRO-INT"],
                          {"rounding_mode": "SINGLE_ROUND"})), 1)

    def test_table_sizes_depend_on_element_type(self):
        for element, size in [("i8_t", "256"), ("i16_t", "513")]:
            row, = self.rows("TABLE", {"input1": element})
            self.assertEqual(row["parameters"]["TABLE_SIZE"], size)

    def test_fixed_operands_outside_type_table(self):
        for name, argument, element in [("SELECT", "input1", "bool_t"),
                                        ("MUL", "shift", "i8_t"),
                                        ("RESHAPE", "shape", "i32_t")]:
            arg = next(
                a for a in self.ops[name]["arguments"] if a["name"] == argument)
            self.assertEqual(arg["fixed_element_type"]["element_type"], element)
        self.assertEqual(self.ops["CONCAT"]["variadic_operands"], ["input1"])

    def test_inline_gather_index_type_and_composite_fft_output(self):
        for row in self.rows("GATHER"):
            self.assertEqual(row["types"]["indices"]["element_type"], "i32_t")
            self.assertEqual(row["types"]["indices"]["rank"], "2")
        row, = self.rows("FFT2D")
        self.assertEqual(
            set(row["types"]),
            {"input_real", "input_imag", "output_real", "output_imag"})

    def test_excluded_extensions_and_level_bounds(self):
        self.assertEqual(
            set(self.database["excluded_tosa_extensions"]),
            {"EXT-INT4", "EXT-DYNAMIC", "EXT-VARIABLE", "EXT-CONTROLFLOW"})
        self.assertEqual(self.database["levels"]["8K"]["MAX_RANK"], 6)
        self.assertEqual(self.database["levels"]["8K"]["MAX_KERNEL"], 8192)

    def test_shape_equations_preserve_source_and_context(self):
        rules = self.ops["CONV2D"]["validation_expressions"]
        self.assertTrue(
            any("OH != idiv_check" in r["expression"] for r in rules))
        for rule in rules:
            self.assertIn("#L", rule["source"])
            self.assertEqual(rule["interpretation"],
                             "requires_containing_pseudocode_context")
        add = self.ops["ADD"]
        self.assertIn("Rank of input tensors must match",
                      add["semantics"]["text_asciidoc"])
        self.assertTrue(
            any("broadcast_shape" in r["expression"]
                for r in add["validation_expressions"]))

    def test_unknown_queries_fail_closed(self):
        for op, types, caps, enums in [
            ("WHILE_LOOP", {}, None, {}),
            ("ADD", {
                "missing": "fp32_t"
            }, None, {}),
            ("ADD", {
                "input1": "fp64_t"
            }, None, {}),
            ("ADD", {}, ["EXT-DYNAMIC"], {}),
            ("ADD", {}, None, {
                "rounding_mode": "DOUBLE_ROUND"
            }),
            ("RESCALE", {}, None, {
                "rounding_mode": "UNKNOWN"
            }),
        ]:
            with self.subTest(op=op, types=types, caps=caps, enums=enums):
                with self.assertRaises(ValueError):
                    self.rows(op, types, caps, enums)

    def test_extractor_rejects_unknown_requirement_syntax(self):
        with self.assertRaises(ValueError):
            extractor.requirements("PRO-INT or PRO-FP",
                                   self.database["extensions"])
        with self.assertRaises(KeyError):
            extractor.requirements("EXT-DYNAMIC", self.database["extensions"])


if __name__ == "__main__":
    unittest.main()
