#include "litert/vendors/google_tensor/dispatch/google_tensor_hook.h"

#include <cstddef>
#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/hooks/power_trace_hook.h"
#include "litert/vendors/google_tensor/hooks/tpu_tile_hook.h"

namespace litert::google_tensor {

struct GoogleTensorHookContext {
  std::unique_ptr<PowerTraceContext, void (*)(PowerTraceContext*)>
      power_context;
  std::unique_ptr<TpuTileTimeContext, void (*)(TpuTileTimeContext*)>
      tpu_tile_context;

  GoogleTensorHookContext()
      : power_context(CreatePowerTraceContext(), DestroyPowerTraceContext),
        tpu_tile_context(CreateTpuTileTimeContext(),
                         DestroyTpuTileTimeContext) {}
};

// Vendor hook callback function for Google Tensor profiling.
void LiteRtVendorHook(LiteRtHookType type, const void* data, size_t size,
                      void* user_data) {
  auto* context = static_cast<GoogleTensorHookContext*>(user_data);
  if (!context) return;

  LiteRtDispatchInvocationContext icontext = nullptr;
  if (type == kLiteRtHookTypeRuntimeStart ||
      type == kLiteRtHookTypeRuntimeStop) {
    if (size == sizeof(LiteRtDispatchInvocationContext) && data != nullptr) {
      icontext =
          *reinterpret_cast<const LiteRtDispatchInvocationContext*>(data);
    }
  }

  switch (type) {
    case kLiteRtHookTypeRuntimeStart:
      HandlePowerRuntimeStart(context->power_context.get());
      HandleTpuTileTimeRuntimeStart(context->tpu_tile_context.get(), icontext);
      break;

    case kLiteRtHookTypeRuntimeStop:
      HandlePowerRuntimeStop(context->power_context.get());
      HandleTpuTileTimeRuntimeStop(context->tpu_tile_context.get(), icontext);
      break;

    case kLiteRtHookTypeStopAndProcess: {
      std::unique_ptr<GoogleTensorHookContext> owned_context(context);
      HandlePowerStopAndProcess(owned_context->power_context.get());
      HandleTpuTileTimeStopAndProcess(owned_context->tpu_tile_context.get());
      break;
    }

    default:
      break;
  }
}

LiteRtStatus GetHooks(LiteRtDispatchDeviceContext device_context,
                      LiteRtHook* hook, void** user_data) {
  *hook = LiteRtVendorHook;
  auto context = std::make_unique<GoogleTensorHookContext>();
  *user_data = context.get();
  if (device_context) {
    device_context->SetVendorHook(LiteRtVendorHook);
    device_context->SetVendorHookUserData(context.get());
  }
  context.release();
  return kLiteRtStatusOk;
}

}  // namespace litert::google_tensor
