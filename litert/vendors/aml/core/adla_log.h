/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    adla_log.h
 * @module  aml_compiler_core / aml_runtime
 * @brief   Shared ADLA logging macros (ADLA_LOGI / ADLA_LOGE / ...).
 * @note    Android uses __android_log_print; host falls back to stderr.
 ******************************************************************************/

#ifndef LITERT_VENDORS_AML_CORE_ADLA_LOG_H_
#define LITERT_VENDORS_AML_CORE_ADLA_LOG_H_

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#else
#include <cstdio>

// Host-side fallback so x86/linux builds can compile AML headers.
// ADLA logging will print to stderr with a best-effort format.
#ifndef ANDROID_LOG_VERBOSE
#define ANDROID_LOG_VERBOSE 2
#endif
#ifndef ANDROID_LOG_DEBUG
#define ANDROID_LOG_DEBUG 3
#endif
#ifndef ANDROID_LOG_INFO
#define ANDROID_LOG_INFO 4
#endif
#ifndef ANDROID_LOG_WARN
#define ANDROID_LOG_WARN 5
#endif
#ifndef ANDROID_LOG_ERROR
#define ANDROID_LOG_ERROR 6
#endif
#ifndef ANDROID_LOG_FATAL
#define ANDROID_LOG_FATAL 7
#endif

#ifndef __android_log_print
#define __android_log_print(level, tag, fmt, ...)                          \
  do {                                                                       \
    (void)(level);                                                           \
    std::fprintf(stderr, "[%s] " fmt, (tag), ##__VA_ARGS__);               \
  } while (0)
#endif
#endif

#include <string>

#define ADLA_LOG_TAG "adla_nnrt"

// 日志等级宏定义（与 Android 内部一致）
#ifndef ADLA_LOG_LEVEL_ENUM_DEFINED
#define ADLA_LOG_LEVEL_ENUM_DEFINED
/**
 * @brief Runtime log severity (aligned with Android log levels).
 */
enum ADLA_LogLevel
{
  ADLA_LOG_LEVEL_VERBOSE = ANDROID_LOG_VERBOSE, // 2
  ADLA_LOG_LEVEL_DEBUG = ANDROID_LOG_DEBUG,     // 3
  ADLA_LOG_LEVEL_INFO = ANDROID_LOG_INFO,       // 4
  ADLA_LOG_LEVEL_WARN = ANDROID_LOG_WARN,       // 5
  ADLA_LOG_LEVEL_ERROR = ANDROID_LOG_ERROR,     // 6
  ADLA_LOG_LEVEL_FATAL = ANDROID_LOG_FATAL      // 7
};
#endif

/** @brief Global minimum log level; messages below it are filtered. */
extern int g_adla_log_level;
/** @brief Global TFLite dump verbosity (separate from ADLA log). */
extern int g_tflite_file_dump_level;

/** @brief Core log macro: print when @p level >= g_adla_log_level. */
#define ADLA_LOG(level, fmt, ...)                                            \
  do                                                                         \
  {                                                                          \
    if ((level) >= g_adla_log_level)                                         \
    {                                                                        \
      __android_log_print(level, ADLA_LOG_TAG, "[%s:%d] " fmt "\n",            \
                          __FUNCTION__, __LINE__, ##__VA_ARGS__);              \
    }                                                                        \
  } while (0)

/** @brief Verbose / Debug / Info / Warn / Error convenience wrappers. */
#define ADLA_LOGV(fmt, ...) ADLA_LOG(ANDROID_LOG_VERBOSE, fmt, ##__VA_ARGS__)
#define ADLA_LOGD(fmt, ...) ADLA_LOG(ANDROID_LOG_DEBUG, fmt, ##__VA_ARGS__)
#define ADLA_LOGI(fmt, ...) ADLA_LOG(ANDROID_LOG_INFO, fmt, ##__VA_ARGS__)
#define ADLA_LOGW(fmt, ...) ADLA_LOG(ANDROID_LOG_WARN, fmt, ##__VA_ARGS__)
#define ADLA_LOGE(fmt, ...) ADLA_LOG(ANDROID_LOG_ERROR, fmt, ##__VA_ARGS__)

/**
 * @brief Read an Android system property as int (host stub may return default).
 * @param key Property name.
 * @param default_value Fallback when unset / unavailable.
 * @return Parsed integer or @p default_value.
 */
int GetPropertyAsInt(const char *key, int default_value);

/** @brief Initialize g_adla_log_level from system property / env / defaults. */
void InitAdlaLogLevel();

/**
 * @brief Override log level set via external_options (0 clears override).
 * @param log_level ADLA_LogLevel value (2-7), or 0 to clear and re-read env/property.
 * @note InitAdlaLogLevel() re-applies a non-zero override after reading env.
 */
void SetAdlaLogLevelOverride(int log_level);

/** @brief Current override from SetAdlaLogLevelOverride(); 0 if unset. */
int GetAdlaLogLevelOverride();

/** @brief Initialize g_tflite_file_dump_level from system property / defaults. */
void InitTfliteFileLevel();

#endif // LITERT_VENDORS_AML_CORE_ADLA_LOG_H_
