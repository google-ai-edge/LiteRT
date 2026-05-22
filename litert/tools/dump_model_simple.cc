// Copyright 2026 Google LLC.
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

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"

namespace {

const char* ElementTypeToString(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
      return "f32";
    case litert::ElementType::Int32:
      return "i32";
    case litert::ElementType::Float64:
      return "f64";
    case litert::ElementType::Int64:
      return "i64";
    case litert::ElementType::Float16:
      return "f16";
    case litert::ElementType::Int16:
      return "i16";
    case litert::ElementType::Int8:
      return "i8";
    case litert::ElementType::UInt8:
      return "ui8";
    case litert::ElementType::Int4:
      return "i4";
    case litert::ElementType::Int2:
      return "i2";
    case litert::ElementType::Bool:
      return "bool";
    default:
      return "unknown";
  }
}

void DumpRankedTensorType(const litert::RankedTensorType& type,
                          std::ostream& out) {
  out << "<";
  auto layout = type.Layout();
  auto dims = layout.Dimensions();
  for (auto dim : dims) {
    out << dim << "x";
  }
  out << ElementTypeToString(type.ElementType());
  auto num_elements = layout.NumElements();
  if (num_elements) {
    out << ":" << *num_elements;
  }
  out << ">";
}

void DumpTensor(const litert::SimpleTensor& tensor, std::ostream& out) {
  auto type_id = tensor.TypeId();
  if (type_id == kLiteRtRankedTensorType) {
    auto ranked_type = tensor.RankedTensorType();
    if (ranked_type) {
      DumpRankedTensorType(*ranked_type, out);
    } else {
      out << "<error getting ranked type>";
    }
  } else {
    auto unranked_type = tensor.UnrankedTensorType();
    if (unranked_type) {
      out << "<unranked> "
          << ElementTypeToString(
                 static_cast<litert::ElementType>(unranked_type->element_type));
    } else {
      out << "<error getting unranked type>";
    }
  }
  out << " [" << tensor.Name() << "]";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_file>\n";
    return 1;
  }

  auto model = litert::Model::CreateFromFile(argv[1]);
  if (!model) {
    std::cerr << "Failed to load model from file: " << argv[1] << "\n";
    return 1;
  }

  auto signatures = model->GetSignatures();
  if (!signatures) {
    std::cerr << "Failed to get signatures\n";
    return 1;
  }

  std::cout << "Model Signatures: " << signatures->size() << "\n";
  for (const auto& signature : *signatures) {
    std::cout << "Signature: " << signature.Key() << "\n";

    std::cout << "  Inputs:\n";
    auto input_names = signature.InputNames();
    for (size_t i = 0; i < input_names.size(); ++i) {
      std::cout << "    Input[" << i << "]: ";
      auto tensor = signature.InputTensor(i);
      if (tensor) {
        DumpTensor(*tensor, std::cout);
      } else {
        std::cout << "<error getting tensor>";
      }
      std::cout << "\n";
    }

    std::cout << "  Outputs:\n";
    auto output_names = signature.OutputNames();
    for (size_t i = 0; i < output_names.size(); ++i) {
      std::cout << "    Output[" << i << "]: ";
      auto tensor = signature.OutputTensor(i);
      if (tensor) {
        DumpTensor(*tensor, std::cout);
      } else {
        std::cout << "<error getting tensor>";
      }
      std::cout << "\n";
    }
  }

  return 0;
}
