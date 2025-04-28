// Copyright 2024 Google LLC.
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

#include "litert/vendors/qualcomm/compiler/legalizations/less_op_legalization.h"

#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "litert/vendors/qualcomm/compiler/legalizations/util.h"
#include "include/QNN/QnnTypes.h"  // from @qairt

namespace litert::qnn {

static constexpr absl::string_view kQnnLessOpTypeName = "ElementWiseLess";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kLessOpFmt = "less_%d";

LiteRtStatus LessOpLegalization::LegalizeOp(const litert::Op& src,
                                            Qnn_OpConfig_t& dest,
                                            GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflLess) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  std::string op_name = absl::StrFormat(kLessOpFmt, op_counter_++);
  LITERT_RETURN_IF_ERROR(SetOpInfo(op_name.c_str(),
                                   kDefaultQnnOpPackageName.data(),
                                   kQnnLessOpTypeName.data(), dest));
  LITERT_RETURN_IF_ERROR(LegalizeSimpleOp(src, dest, graph_mapper));
  LITERT_LOG(LITERT_INFO, "Legalized less op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
