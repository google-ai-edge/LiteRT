/*******************************************************************************
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 ******************************************************************************/

#include "adla_log.h"

#include <cstdlib>
#include <cstring>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

int g_adla_log_level = ANDROID_LOG_INFO;
int g_tflite_file_dump_level = 0;

namespace {
int g_adla_log_level_override = 0;
}  // namespace

int GetPropertyAsInt(const char *key, int default_value)
{
  if (key == nullptr)
  {
    return default_value;
  }
#ifdef __ANDROID__
  char value[PROP_VALUE_MAX] = {0};
  if (__system_property_get(key, value) > 0)
  {
    return atoi(value);
  }
  return default_value;
#else
  // Linux/host: read environment variables instead of Android properties.
  // Accept both full key (vendor.TFLITE_FILE_DUMP_LEVEL) and short key
  // (TFLITE_FILE_DUMP_LEVEL).
  if (const char *env = std::getenv(key))
  {
    return std::atoi(env);
  }
  if (const char *dot = std::strrchr(key, '.'))
  {
    if (const char *env = std::getenv(dot + 1))
    {
      return std::atoi(env);
    }
  }
  return default_value;
#endif
}

int GetAdlaLogLevelOverride() { return g_adla_log_level_override; }

void SetAdlaLogLevelOverride(int log_level) {
  g_adla_log_level_override = log_level;
  if (log_level != 0) {
    g_adla_log_level = log_level;
  } else {
    g_adla_log_level = GetPropertyAsInt("vendor.AML_DELEGATE_LOG_LEVEL", 3);
  }
}

void InitAdlaLogLevel()
{
  g_adla_log_level = GetPropertyAsInt("vendor.AML_DELEGATE_LOG_LEVEL", 3);
  if (g_adla_log_level_override != 0) {
    g_adla_log_level = g_adla_log_level_override;
  }
}

void InitTfliteFileLevel()
{
  g_tflite_file_dump_level = GetPropertyAsInt("vendor.TFLITE_FILE_DUMP_LEVEL", 0);
}