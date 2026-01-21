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

#include "litert/python/converter.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/statusor_caster.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf

namespace py = pybind11;

namespace litert {

PYBIND11_MODULE(_pywrap_litert_converter, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  pybind11::google::ImportStatusModule();

  py::class_<ConversionConfig> py_conversion_config(m, "ConversionConfig");
  py::enum_<ConversionConfig::ModelType>(py_conversion_config, "ModelType")
      .value("Unknown", ConversionConfig::ModelType::kUnknown)
      .value("Jax", ConversionConfig::ModelType::kJax)
      .value("PyTorch", ConversionConfig::ModelType::kPyTorch);

  py_conversion_config
      .def(py::init<>())
      .def_readwrite("original_model_type",
                     &ConversionConfig::original_model_type)
      .def_readwrite("converter_flags", &ConversionConfig::converter_flags)
      .def_readwrite("model_flags", &ConversionConfig::model_flags);

  py::class_<Converter>(m, "Converter")
      .def(py::init<const ConversionConfig&>())
      .def("add_signature", &Converter::AddSignature)
      .def("convert", &Converter::Convert);
}

}  // namespace litert
