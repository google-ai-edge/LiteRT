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
  std::ifstream file(raw_env);
  if (file.is_open()) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    result = buffer.str();
  } else {
    result = raw_env;
  }

  StripSpaces(&result);
  return result;
}

}  // namespace litert::google_tensor
