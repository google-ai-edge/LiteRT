#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_HOOKS_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_HOOKS_UTILS_H_

#include <string>

namespace litert::google_tensor {

// Retrieves the content of LITERT_VENDOR_HOOK_ARGS.
// If the environment variable points to a readable file, returns the file's
// contents. Otherwise, returns the literal string value of the environment
// variable. Returns an empty string if the environment variable is not set.
// All whitespace characters are stripped from the resulting string before it is
// returned.
std::string GetVendorHookArgsConfig();

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_HOOKS_UTILS_H_
