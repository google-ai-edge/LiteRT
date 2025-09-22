// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "litert/python/converter.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
#include "pybind11_abseil/statusor_caster.h"  // from @pybind11_abseil

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_litert_converter, m) {
  py::enum_<ConversionConfig::ModelType>(m, "ModelType")
      .value("Jax", ConversionConfig::ModelType::Jax)
      .value("PyTorch", ConversionConfig::ModelType::PyTorch);

  py::class_<ConversionConfig>(m, "ConversionConfig")
      .def(py::init<>())
      .def_property(
          "model_flags_proto_txt_raw",
          // Getter: Return the std::string as py::bytes
          [](const ConversionConfig& c) {
            return py::bytes(c.converter_flags_proto_txt_raw);
          },
          // Setter: Accept only py::bytes
          [](ConversionConfig& c, py::bytes b) {
            c.model_flags_proto_txt_raw = std::string(b);
          })
      .def_property(
          "converter_flags_proto_txt_raw",
          // Getter: Return the std::string as py::bytes
          [](const ConversionConfig& c) {
            return py::bytes(c.converter_flags_proto_txt_raw);
          },
          // Setter: Accept only py::bytes
          [](ConversionConfig& c, py::bytes b) {
            c.converter_flags_proto_txt_raw = std::string(b);
          })
      .def_property(
          "input_contents_txt_raw",
          // Getter: Return the std::string as py::bytes
          [](const ConversionConfig& c) {
            return py::bytes(c.input_contents_txt_raw);
          },
          // Setter: Accept only py::bytes
          [](ConversionConfig& c, py::bytes b) {
            c.input_contents_txt_raw = std::string(b);
          })
      .def_property(
          "debug_info_txt_raw",
          // Getter: Return the std::string as py::bytes
          [](const ConversionConfig& c) {
            return py::bytes(c.debug_info_txt_raw);
          },
          // Setter: Accept only py::bytes
          [](ConversionConfig& c, py::bytes b) {
            c.debug_info_txt_raw = std::string(b);
          })
      .def_readwrite(
          "experimental_enable_composite_direct_lowering",
          &ConversionConfig::experimental_enable_composite_direct_lowering)
      .def_readwrite(
          "experimental_enable_dynamic_update_slice",
          &ConversionConfig::experimental_enable_dynamic_update_slice)
      .def_readwrite("experimental_qdq_conversion_mode",
                     &ConversionConfig::experimental_qdq_conversion_mode)
      .def_readwrite("extended_return", &ConversionConfig::extended_return)
      .def_readwrite("model_origin_framework",
                     &ConversionConfig::model_origin_framework)
      .def_readwrite("original_model_type",
                     &ConversionConfig::original_model_type);

  py::class_<ConverterSignature>(m, "ConverterSignature")
      .def(py::init<>())
      .def_readwrite("signature_name", &ConverterSignature::signature_name)
      .def_readwrite("input_names", &ConverterSignature::input_names)
      .def_readwrite("output_names", &ConverterSignature::output_names)
      .def_readwrite("data", &ConverterSignature::data);

  py::class_<LiteRtConverter>(m, "LiteRtConverter")
      .def(py::init<const ConversionConfig&>())
      .def("addSignature", &LiteRtConverter::addSignature)
      .def("convert", &LiteRtConverter::convert);
}
