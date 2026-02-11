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
#include "litert/compiler/mlir/dialects/litert/attributes.h"

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/RandomNumberGenerator.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "litert/compiler/mlir/dialects/litert/callback_resource.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"
#include "litert/compiler/mlir/dialects/litert/lazy_resource_blob.h"
#include "litert/compiler/mlir/dialects/litert/tensor_parser.h"
#include "litert/compiler/mlir/dialects/litert/tensor_printer.h"

#define GET_ATTRDEF_CLASSES
#include "litert/compiler/mlir/dialects/litert/attributes.cc.inc"  // IWYU pragma: keep

namespace litert {

namespace details {

std::string GenerateUUID() {
  uint8_t b[16];

  // Fill the buffer with OS-level random entropy
  if (auto EC = llvm::getRandomBytes(b, 16)) {
    // Fallback or error handling: in most LLVM tools,
    // high-level errors are handled via llvm::report_fatal_error
    ABSL_CHECK(false) << "Failed to generate random bytes: " << EC.message();
  }

  // RFC 4122 Version 4 Requirements:
  // Set the four most significant bits of the 7th byte to 0100 (Version 4)
  b[6] = (b[6] & 0x0F) | 0x40;

  // Set the two most significant bits of the 9th byte to 10 (Variant 1)
  b[8] = (b[8] & 0x3F) | 0x80;

  // Format into the standard 8-4-4-4-12 hex string
  return std::format(
      "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:"
      "02x}{:02x}{:02x}{:02x}{:02x}",
      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
      b[12], b[13], b[14], b[15]);
}

}  // namespace details

void LITERTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "litert/compiler/mlir/dialects/litert/attributes.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// PyDenseElementsAttr
//===----------------------------------------------------------------------===//

mlir::Attribute CallbackResourceElementsAttr::parse(mlir::AsmParser& parser,
                                                    mlir::Type type) {
  parser.emitError(parser.getCurrentLocation())
      << "Parsing PyDenseElementsAttr is not supported.";
  return nullptr;
}

void CallbackResourceElementsAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<__elided__>";
}

CallbackResourceBase* CallbackResourceElementsAttr::GetResource() {
  return getRawHandle().getResource();
}

CallbackResourceElementsAttr CallbackResourceElementsAttr::get(
    mlir::ShapedType type, CallbackResourceElementsHandle handle) {
  return Base::get(type.getContext(), type, handle);
}

CallbackResourceElementsAttr CallbackResourceElementsAttr::get(
    mlir::ShapedType type, std::unique_ptr<CallbackResourceBase> resource) {
  // 1. Get the interface.
  auto& interface =
      CallbackResourceElementsHandle::getManagerInterface(type.getContext());

  // 2. Access the manager.
  CallbackResourceManager& manager = interface.GetCallbackResourceManager();

  // 3. Cast the base Dialect* to the specific Dialect type required by the
  // Handle. This satisfies the 'typename HandleT::Dialect*' requirement in the
  // template.
  using TargetDialect = CallbackResourceElementsHandle::Dialect;
  auto* dialect = static_cast<TargetDialect*>(interface.getDialect());

  // 4. Explicitly call Insert with the handle template argument.
  auto handle = manager.Insert<CallbackResourceElementsHandle>(
      dialect, details::GenerateUUID(), std::move(resource));

  // 5. Build the attribute.
  return Base::get(type.getContext(), type, handle);
}

//===----------------------------------------------------------------------===//
// LazyDenseElementsAttr
//===----------------------------------------------------------------------===//

mlir::Attribute LazyDenseElementsAttr::parse(mlir::AsmParser& parser,
                                             mlir::Type type) {
  if (parser.parseLess().failed()) {
    parser.emitError(parser.getCurrentLocation())
        << "expected '<' after 'lazy_dense'";
    return nullptr;
  }
  auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped_type) {
    parser.emitError(parser.getCurrentLocation())
        << "elements literal must be a shaped type.";
    return nullptr;
  }

  if (!shaped_type.hasStaticShape()) {
    parser.emitError(parser.getCurrentLocation())
        << "elements literal type must have static shape.";
    return nullptr;
  }

  mlir::Type element_type = shaped_type.getElementType();

  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
    if (int_type.isSignlessInteger() && int_type.getWidth() == 32) {
      TensorParser<int32_t> tensor_parser(parser, shaped_type);
      if (tensor_parser.Parse().failed()) {
        return nullptr;
      }
      auto elements = tensor_parser.GetData();
      if (elements.size() == 1 && shaped_type.getNumElements() > 1) {
        std::vector<int32_t> splat_elements(shaped_type.getNumElements(),
                                            elements[0]);
        return LazyDenseElementsAttr::get<int32_t>(shaped_type, splat_elements);
      }

      return LazyDenseElementsAttr::get<int32_t>(shaped_type, elements);
    }
  } else if (mlir::isa<mlir::FloatType>(element_type)) {
    TensorParser<float> tensor_parser(parser, shaped_type);
    if (tensor_parser.Parse().failed()) {
      return nullptr;
    }
    auto elements = tensor_parser.GetData();
    if (elements.size() == 1 && shaped_type.getNumElements() > 1) {
      std::vector<float> splat_elements(shaped_type.getNumElements(),
                                        elements[0]);
      return LazyDenseElementsAttr::get<float>(shaped_type, splat_elements);
    }

    return LazyDenseElementsAttr::get<float>(shaped_type, elements);
  }

  return nullptr;
}

namespace {

static constexpr int kLargeElementsThreshold = 64;

}  // namespace

void LazyDenseElementsAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<";

  absl::Cleanup bracket_closer = [&] { printer << ">"; };

  auto num_elements = getType().getNumElements();
  if (num_elements == 0) return;

  if (num_elements > kLargeElementsThreshold) {
    printer << "__elided__";
    return;
  }

  auto data_handle = GetDataHandle();

  auto element_type = getType().getElementType();

  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
    if (int_type.isSignless() || int_type.isSigned()) {
      switch (int_type.getWidth()) {
        case 32: {
          auto data = data_handle.GetDataAs<int32_t>();
          TensorPrinter<int32_t> tensor_printer(data, getType(), printer);
          tensor_printer.Print();
          break;
        }
        case 64: {
          auto data = data_handle.GetDataAs<int64_t>();
          TensorPrinter<int64_t> tensor_printer(data, getType(), printer);
          tensor_printer.Print();
          break;
        }
      }
    } else {
      switch (int_type.getWidth()) {
        case 32: {
          auto data = data_handle.GetDataAs<uint32_t>();
          TensorPrinter<uint32_t> tensor_printer(data, getType(), printer);
          tensor_printer.Print();
          break;
        }
        case 64: {
          auto data = data_handle.GetDataAs<uint64_t>();
          TensorPrinter<uint64_t> tensor_printer(data, getType(), printer);
          tensor_printer.Print();
          break;
        }
      }
    }
  } else if (auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type)) {
    switch (float_type.getWidth()) {
      case 32: {
        auto data = data_handle.GetDataAs<float>();
        TensorPrinter<float> tensor_printer(data, getType(), printer);
        tensor_printer.Print();
        break;
      }
      case 64: {
        auto data = data_handle.GetDataAs<double>();
        TensorPrinter<double> tensor_printer(data, getType(), printer);
        tensor_printer.Print();
        break;
      }
    }
  }
}

LazyDenseElementsAttr LazyDenseElementsAttr::get(
    mlir::ShapedType type, LazyResourceElementsHandle handle) {
  return Base::get(type.getContext(), type, handle);
}

LazyDenseElementsAttr LazyDenseElementsAttr::get(mlir::ShapedType type,
                                                 LazyResourceBlob blob) {
  auto& manager =
      LazyResourceElementsHandle::getManagerInterface(type.getContext());
  return get(type, manager.Insert(details::GenerateUUID(), std::move(blob)));
}

LazyDenseElementsAttr LazyDenseElementsAttr::get(mlir::ShapedType type,
                                                 llvm::ArrayRef<uint8_t> data,
                                                 size_t alignment) {
  if (auto element_type = type.getElementType();
      element_type.isIntOrIndexOrFloat()) {
    auto bit_width = element_type.getIntOrFloatBitWidth();
    auto num_elements = data.size() / (bit_width / 8);
    CHECK(type.getNumElements() == num_elements)
        << "The number of elements in the ShapedType must match the number of "
           "elements in the provided data buffer (expected "
        << type.getNumElements() << " but got " << num_elements << ").";
  }
  auto blob = LazyResourceBlob::CreateAndCopyData(data, alignment);
  auto& manager =
      LazyResourceElementsHandle::getManagerInterface(type.getContext());
  return get(type, manager.Insert(details::GenerateUUID(), std::move(blob)));
}

ScopedDataHandle LazyDenseElementsAttr::GetDataHandle() const {
  return getRawHandle().GetBlob()->GetDataHandle();
}

const LazyResourceBlob& LazyDenseElementsAttr::GetBlob() const {
  return *getRawHandle().GetBlob();
}

//===----------------------------------------------------------------------===//
// SymDimAttr
//===----------------------------------------------------------------------===//

mlir::Attribute SymDimAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  // TODO(aarfaian): implement
  CHECK(false) << "Not implemented.";
  return nullptr;
}

void SymDimAttr::print(mlir::AsmPrinter& printer) const {
  if (getSize() == mlir::ShapedType::kDynamic) {
    if (!getSymbol()) {
      printer << "?";
    } else {
      printer << getSymbol();
    }
  } else {
    printer << getSize();
  }
}

}  // namespace litert
