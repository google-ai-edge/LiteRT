#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/google_tensor_hook.h"

namespace litert::google_tensor {

void LiteRtVendorHook(LiteRtHookType type, const void* /*data*/,
                      size_t /*size*/, void* /*user_data*/) {
  // No-op
}

LiteRtStatus GetHooks(LiteRtDispatchDeviceContext /*device_context*/,
                      LiteRtHook* hook, void** user_data) {
  return kLiteRtStatusOk;
}

}  // namespace litert::google_tensor
