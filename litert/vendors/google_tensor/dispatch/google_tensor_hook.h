#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_GOOGLE_TENSOR_HOOK_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_GOOGLE_TENSOR_HOOK_H_

#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_types.h"
#include "litert/vendors/c/litert_dispatch.h"

namespace litert::google_tensor {

// LiteRtVendorHook is the entry point for vendor-specific profiling and
// instrumentation hooks. These hooks execute during various stages of the
// model execution lifecycle (e.g. compilation start/stop, runtime start/stop,
// and shutdown).
//
// Currently, this is utilized to record power and energy
// metrics (e.g. TPU energy consumption) for Google
// Tensor, but it may be extended to trace and log other latency or performance
// data.
void LiteRtVendorHook(LiteRtHookType type, const void* data, size_t size,
                      void* user_data);

LiteRtStatus GetHooks(LiteRtDispatchDeviceContext device_context,
                      LiteRtHook* hook, void** user_data);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_GOOGLE_TENSOR_HOOK_H_
