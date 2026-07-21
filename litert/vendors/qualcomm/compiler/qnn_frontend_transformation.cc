#include "litert/vendors/qualcomm/compiler/qnn_frontend_transformation.h"

#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_builder.h"

using litert::Builder;

LiteRtStatus QnnTransformation(LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  LITERT_LOG(LITERT_INFO, "[Google G2G] Running");
  Builder builder(builder_ptr);
  return kLiteRtStatusOk;
}
