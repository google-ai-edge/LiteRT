#include "litert/vendors/google_tensor/hooks/hooks_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace litert::google_tensor {

static void StripSpaces(std::string* str) {
  str->erase(std::remove_if(str->begin(), str->end(), ::isspace), str->end());
}

std::string GetVendorHookArgsConfig() {
  const char* hook_args_env = std::getenv("LITERT_VENDOR_HOOK_ARGS");
  if (!hook_args_env) {
    return "";
  }

  std::string raw_env(hook_args_env);
  std::string result;

  // The environment variable can either be a path to a configuration file
  // (e.g., a .textproto file) or the raw configuration string itself.
  // We first attempt to open it as a file.
  std::ifstream file(raw_env);
  if (file.is_open()) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    result = buffer.str();
  } else {
    // If it's not a valid file path, we assume the environment variable
    // contains the literal configuration arguments directly.
    result = raw_env;
  }

  // Strip all whitespaces to simplify downstream string tokenization and
  // parsing.
  StripSpaces(&result);
  return result;
}

}  // namespace litert::google_tensor
