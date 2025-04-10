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

#ifndef ODML_LITERT_LITERT_GOOGLE_INVOKE_QUALCOMM_UTIL_H_
#define ODML_LITERT_LITERT_GOOGLE_INVOKE_QUALCOMM_UTIL_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_dispatch_delegate.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_dispatch_delegate.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/build_stamp.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/tools/outstream.h"
#include "litert/tools/tool_display.h"
#include "tflite/interpreter.h"  // from @org_tensorflow
#include "tflite/model_builder.h"  // from @org_tensorflow

namespace litert::tools {

using ::litert::Environment;
using ::litert::internal::FlatbufferWrapper;
using ::litert::internal::GetMetadata;
using ::litert::testing::TflRuntime;
using ::tflite::FlatBufferModel;
using ::tflite::Interpreter;

static constexpr absl::string_view kToolName = "INVOKE_MODEL";
static constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

inline std::tuple<Environment, TflRuntime::Ptr, ToolDisplay::Ptr,
                  DispatchDelegatePtr>
SetupInvocation(const std::string& model_path, const std::string& err) {
  const std::vector<litert::Environment::Option> environment_options = {
      ::litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));

  auto display =
      std::make_unique<ToolDisplay>(UserStream::MakeFromFlag(err), kToolName);
  auto& disp = *display;
  DumpPreamble(disp);
  auto setup_scope = disp.StartS("Setup");
  disp.Labeled() << absl::StreamFormat("MODEL_PATH: %s\n", model_path);

  // Load model and interpreter.

  auto runtime = TflRuntime::CreateFromFlatBuffer(
      *FlatbufferWrapper::CreateFromTflFile(model_path));
  ABSL_CHECK(runtime) << "Could not setup runtime";
  auto& rt = **runtime;
  auto& interp = rt.Interpreter();

  disp.Labeled() << absl::StreamFormat("Loaded a model of size: %lu\n",
                                       rt.Flatbuffer().Buf().Size());
  disp.Labeled() << absl::StreamFormat(
      "Created interpreter with %lu subgraphs, %lu inputs and %lu outputs\n",
      interp.subgraphs_size(), interp.inputs().size(), interp.outputs().size());

  {
    // Check model is compatible.

    auto tag_scope = disp.StartS("Checking build tag");
    auto build_tag_buf =
        GetMetadata(internal::kLiteRtBuildStampKey, *rt.Flatbuffer().Unpack());
    ABSL_CHECK(build_tag_buf) << "Could not find build tag in metadata\n";
    auto build_stamp = internal::ParseBuildStamp(*build_tag_buf);
    ABSL_CHECK(build_stamp) << "Could not parse build stamp\n";
    auto [man, model] = *build_stamp;
    ABSL_CHECK_EQ(man, "Qualcomm");
    disp.Labeled() << absl::StreamFormat("\n\tSOC_MAN: %s\n\tSOC_MODEL: %s\n",
                                         man, model);
  }

  // Make delegate.
  auto delegate_scope = disp.StartS("Initializing delegate");
  LiteRtEnvironmentOptions env_options;
  LiteRtGetEnvironmentOptions(env->Get(), &env_options);
  auto dispatch_delegate_options =
      CreateDispatchDelegateOptionsPtr(env_options);
  ABSL_CHECK_EQ(
      LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                               rt.Flatbuffer().Buf().Data()),
      kTfLiteOk);
  auto dispatch_delegate = CreateDispatchDelegatePtr(
      env_options, std::move(dispatch_delegate_options));

  return std::make_tuple(std::move(*env), std::move(*runtime),
                         std::move(display), std::move(dispatch_delegate));
}

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_GOOGLE_INVOKE_QUALCOMM_UTIL_H_
