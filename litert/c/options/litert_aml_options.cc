// Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

#include "litert/c/options/litert_aml_options.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h" // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"

// 内部结构体定义
struct LiteRtAmlOptionsT
{
  LiteRtAmlOptionsLogLevel log_level = kLiteRtAmlLogLevelInfo;
  LiteRtAmlOptionsProfiling profiling = kLiteRtAmlProfilingOff;
  LiteRtAmlOptionsPerfMode perf_mode = kLiteRtAmlPerfDefault;
  LiteRtAmlOptionsMemoryPolicy memory_policy = kLiteRtAmlMemoryDefault;
  std::string model_path;
  std::string model_name;
};

// 创建 Options
LiteRtStatus LiteRtAmlOptionsCreate(LiteRtOpaqueOptions *options)
{
  if (options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtAmlOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtAmlOptionsGetIdentifier(), options_data.get(),
      [](void *payload)
      {
        delete reinterpret_cast<LiteRtAmlOptions>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}

// identifier
const char *LiteRtAmlOptionsGetIdentifier() { return "amlogic"; }

// 获取 Options
LiteRtStatus LiteRtAmlOptionsGet(LiteRtOpaqueOptions options,
                                 LiteRtAmlOptions *options_data)
{
  if (options_data == nullptr || options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char *identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtAmlOptionsGetIdentifier())
  {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void *payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtAmlOptions>(payload);

  return kLiteRtStatusOk;
}

// ================= GLOBAL OPTIONS =================

// log_level
LiteRtStatus LiteRtAmlOptionsSetLogLevel(
    LiteRtAmlOptions options, LiteRtAmlOptionsLogLevel log_level)
{
  if (options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->log_level = log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsGetLogLevel(
    LiteRtAmlOptions options, LiteRtAmlOptionsLogLevel *log_level)
{
  if (log_level == nullptr || options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *log_level = options->log_level;
  return kLiteRtStatusOk;
}

// ================= EXECUTION / DISPATCH OPTIONS =================

// PerfMode
LiteRtStatus LiteRtAmlOptionsSetPerfMode(
    LiteRtAmlOptions options, LiteRtAmlOptionsPerfMode mode)
{
  if (options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->perf_mode = mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsGetPerfMode(
    LiteRtAmlOptions options, LiteRtAmlOptionsPerfMode *mode)
{
  if (options == nullptr || mode == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *mode = options->perf_mode;
  return kLiteRtStatusOk;
}

// MemoryPolicy
LiteRtStatus LiteRtAmlOptionsSetMemoryPolicy(
    LiteRtAmlOptions options, LiteRtAmlOptionsMemoryPolicy policy)
{
  if (options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->memory_policy = policy;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsGetMemoryPolicy(
    LiteRtAmlOptions options, LiteRtAmlOptionsMemoryPolicy *policy)
{
  if (options == nullptr || policy == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *policy = options->memory_policy;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsSetModelPath(
    LiteRtAmlOptions options, const char *model_path)
{
  if (options == nullptr || model_path == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_path = model_path;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsGetModelPath(
    LiteRtAmlOptions options, const char **model_path)
{
  if (options == nullptr || model_path == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_path = options->model_path.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsSetModelName(
    LiteRtAmlOptions options, const char *model_name)
{
  if (options == nullptr || model_name == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_name = model_name;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsGetModelName(
    LiteRtAmlOptions options, const char **model_name)
{
  if (options == nullptr || model_name == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_name = options->model_name.c_str();
  return kLiteRtStatusOk;
}

// Profiling
LiteRtStatus LiteRtAmlOptionsSetProfiling(
    LiteRtAmlOptions options, LiteRtAmlOptionsProfiling profiling)
{
  if (options == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->profiling = profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAmlOptionsGetProfiling(
    LiteRtAmlOptions options, LiteRtAmlOptionsProfiling *profiling)
{
  if (options == nullptr || profiling == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *profiling = options->profiling;
  return kLiteRtStatusOk;
}