// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_DML_PLATFORM_FUNCTIONS_H_
#define SERVICES_WEBNN_DML_PLATFORM_FUNCTIONS_H_

#include <windows.h>

#define DML_TARGET_VERSION_USE_LATEST
#include "DirectML.h"
#undef DML_TARGET_VERSION_USE_LATEST
#include "d3d12.h"
#include "dxcore.h"

// namespace webnn::dml {

class  PlatformFunctions {
 public:
  PlatformFunctions(const PlatformFunctions&) = delete;
  PlatformFunctions& operator=(const PlatformFunctions&) = delete;

  static PlatformFunctions* GetInstance();

  using D3d12CreateDeviceProc = PFN_D3D12_CREATE_DEVICE;
  D3d12CreateDeviceProc d3d12_create_device_proc() const {
    return d3d12_create_device_proc_;
  }

  using D3d12GetDebugInterfaceProc = PFN_D3D12_GET_DEBUG_INTERFACE;
  D3d12GetDebugInterfaceProc d3d12_get_debug_interface_proc() const {
    return d3d12_get_debug_interface_proc_;
  }

  using DXCoreCreateAdapterFactoryProc =
      decltype(static_cast<STDMETHODIMP (*)(REFIID, void**)>(
          DXCoreCreateAdapterFactory));
  DXCoreCreateAdapterFactoryProc dxcore_create_adapter_factory_proc() const {
    return dxcore_create_adapter_factory_proc_;
  }

  using DmlCreateDevice1Proc = decltype(DMLCreateDevice1)*;
  DmlCreateDevice1Proc dml_create_device1_proc() const {
    return dml_create_device1_proc_;
  }

  bool IsDXCoreSupported() const { return dxcore_library_ != NULL; }

 private:
  PlatformFunctions();
  ~PlatformFunctions();

  bool AllFunctionsLoaded();

  // D3D12
  HMODULE d3d12_library_;
  D3d12CreateDeviceProc d3d12_create_device_proc_;
  D3d12GetDebugInterfaceProc d3d12_get_debug_interface_proc_;

  // DXCore library can be null as it was missing in older Windows 10 versions.
  // It's needed for Microsoft Compute Driver Model (MCDM) devices (NPUs) which
  // are not enumerable via DXGI.
  HMODULE dxcore_library_;
  DXCoreCreateAdapterFactoryProc dxcore_create_adapter_factory_proc_ = nullptr;

  // DirectML
  HMODULE dml_library_;
  DmlCreateDevice1Proc dml_create_device1_proc_;
};

// }  // namespace webnn::dml

#endif  // SERVICES_WEBNN_DML_PLATFORM_FUNCTIONS_H_
