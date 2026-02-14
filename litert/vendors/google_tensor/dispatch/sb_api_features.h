#ifndef HARDWARE_GCHIPS_HETERO_RUNTIME_STAR_SHIP_SOUTH_BOUND_SB_API_FEATURES_H_
#define HARDWARE_GCHIPS_HETERO_RUNTIME_STAR_SHIP_SOUTH_BOUND_SB_API_FEATURES_H_

#include <stdlib.h>

#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// A collection of utility functions to assist with querying SouthBound feature
// support for Google Tensor.

// Returns the major version of the Google Tensor SouthBound implementation.
inline int GoogleTensorSouthBoundMajorVersion() {
  static const int major_version =
      atoi(thrGetVendorApiVersion());  // NOLINT(runtime/deprecated_fn)
  return major_version;
}

// Returns the minor version of the Google Tensor SouthBound implementation.
inline int GoogleTensorSouthBoundMinorVersion() {
  static const int minor_version =
      atoi(thrGetVendorApiVersion() + 2);  // NOLINT(runtime/deprecated_fn)
  return minor_version;
}

// Returns the patch version of the Google Tensor SouthBound implementation.
inline int GoogleTensorSouthBoundPatchVersion() {
  static const int patch_version =
      atoi(thrGetVendorApiVersion() + 4);  // NOLINT(runtime/deprecated_fn)
  return patch_version;
}

// A SouthBound "feature" is a collection of 1 or more related SouthBound APIs
// which receive support in a particular semantic version of the Google Tensor
// SouthBound implementation.
enum class GoogleTensorSouthBoundFeature {
  // thrInvocationContextCancel
  kCancellation = 0,
  // thrVendorSetSystemAttributeStr
  kStringSystemAttributes = 1,
  // thrPinSqContainer
  // thrUnpinSqContainer
  kSqPinning = 2,
  // thrGraphAddSqNodeWithInterfaceBindingMode
  // thrGraphConnectNodeInputWithPortName
  // thrGraphConnectNodeOutputWithPortName
  kNamedNodeBinding = 3,
  // thrRegisterFence
  // thrUnregisterFence
  // thrFenceGetDupFd
  // thrInvocationContextPrepareForInvoke2
  // thrInvocationContextAttachInputBufferFence
  // thrInvocationContextDetachInputBufferFence
  // thrInvocationContextGetOutputBufferFence
  kRobustFences = 4,
  // thrGraphConnectNodeInputWithPortIndex
  // thrGraphConnectNodeOutputWithPortIndex
  kIndexedNodeBinding = 5,
  // thrRegisterBuffer* with `type == kThrBufferTypeDmaBuf`
  kDmaBufRegistration = 6,
  // thrGraphAnnotateGraph with `key == kOriginalUid`
  kOriginalUidDispatchDirective = 7,
  // BufferDirectiveAnnotations::kRequestFence
  kRequestFence = 8,
};

// Returns `true` if `feature` is supported by the available Google Tensor
// SouthBound implementation, else `false`.
inline bool GoogleTensorSouthBoundFeatureSupported(
    GoogleTensorSouthBoundFeature feature) {
  switch (feature) {
    case GoogleTensorSouthBoundFeature::kCancellation:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 9;
    case GoogleTensorSouthBoundFeature::kStringSystemAttributes:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 10;
    case GoogleTensorSouthBoundFeature::kSqPinning:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 11;
    case GoogleTensorSouthBoundFeature::kNamedNodeBinding:
    case GoogleTensorSouthBoundFeature::kRobustFences:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 12;
    case GoogleTensorSouthBoundFeature::kIndexedNodeBinding:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 14;
    case GoogleTensorSouthBoundFeature::kDmaBufRegistration:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 15;
    case GoogleTensorSouthBoundFeature::kOriginalUidDispatchDirective:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 16;
    case GoogleTensorSouthBoundFeature::kRequestFence:
      return GoogleTensorSouthBoundMajorVersion() > 0 ||
             GoogleTensorSouthBoundMinorVersion() >= 17;
  }
}

#endif  // HARDWARE_GCHIPS_HETERO_RUNTIME_STAR_SHIP_SOUTH_BOUND_SB_API_FEATURES_H_
