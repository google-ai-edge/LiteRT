// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "litert/runtime/platform_functions.h"

// #include "base/files/file_path.h"
// #include "base/logging.h"
// #include "base/native_library.h"
// #include "base/path_service.h"

#include "litert/c/internal/litert_logging.h"

// namespace webnn::dml {

PlatformFunctions::PlatformFunctions() {
  // D3D12
  d3d12_library_ = ::LoadLibraryExW(L"D3D12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (d3d12_library_ == NULL) {
    LITERT_LOG(LITERT_INFO, "[WebNN] Failed to load D3D12.dll.");
    return;
  }
  d3d12_create_device_proc_ =
      reinterpret_cast<D3d12CreateDeviceProc>(
          GetProcAddress(d3d12_library_, "D3D12CreateDevice"));
  if (!d3d12_create_device_proc_) {
    LITERT_LOG(LITERT_INFO, "[WebNN] Failed to get D3D12CreateDevice function.");
    return;
  }

  // D3d12GetDebugInterfaceProc d3d12_get_debug_interface_proc =
  //     reinterpret_cast<D3d12GetDebugInterfaceProc>(
  //         d3d12_library.GetFunctionPointer("D3D12GetDebugInterface"));
  // if (!d3d12_get_debug_interface_proc) {
  //   LOG(ERROR) << "[WebNN] Failed to get D3D12GetDebugInterface function.";
  //   return;
  // }

  // // First try to Load DirectML.dll from the module folder. It would enable
  // // running unit tests which require DirectML feature level 4.0+ on Windows 10.
  // base::ScopedNativeLibrary dml_library;
  // base::FilePath module_path;
  // if (base::PathService::Get(base::DIR_MODULE, &module_path)) {
  //   dml_library = base::ScopedNativeLibrary(
  //       base::LoadNativeLibrary(module_path.Append(L"directml.dll"), nullptr));
  // }
  // // If it failed to load from module folder, try to load from system folder.
  // if (!dml_library.is_valid()) {
  //   dml_library =
  //       base::ScopedNativeLibrary(base::LoadSystemLibrary(L"directml.dll"));
  // }
  // if (!dml_library.is_valid()) {
  //   LOG(ERROR) << "[WebNN] Failed to load directml.dll.";
  //   return;
  // }
  // // On older versions of Windows, DMLCreateDevice was not publicly documented
  // // and took a different number of arguments than the publicly documented
  // // version of the function supported by later versions of the DLL. We should
  // // use DMLCreateDevice1 which has always been publicly documented and accepts
  // // a well defined number of arguments."
  // DmlCreateDevice1Proc dml_create_device1_proc =
  //     reinterpret_cast<DmlCreateDevice1Proc>(
  //         dml_library.GetFunctionPointer("DMLCreateDevice1"));
  // if (!dml_create_device1_proc) {
  //   LOG(ERROR) << "[WebNN] Failed to get DMLCreateDevice1 function.";
  //   return;
  // }

  // DXCore which is optional.
  dxcore_library_ = ::LoadLibraryExW(L"DXCore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  // PlatformFunctions::DXCoreCreateAdapterFactoryProc
  //     dxcore_create_adapter_factory_proc;
  if (dxcore_library_ == NULL) {
    LITERT_LOG(LITERT_INFO, "[WebNN] Failed to load DXCore.dll.");
  } else {
    dxcore_create_adapter_factory_proc_ =
        reinterpret_cast<DXCoreCreateAdapterFactoryProc>(
            GetProcAddress(dxcore_library_, "DXCoreCreateAdapterFactory"));
    if (!dxcore_create_adapter_factory_proc_) {
      LITERT_LOG(LITERT_INFO, "[WebNN] Failed to get DXCoreCreateAdapterFactory function.");
    }
    LITERT_LOG(LITERT_INFO, "[WebNN] Successed to get DXCoreCreateAdapterFactory function.");
  }
}

PlatformFunctions::~PlatformFunctions() = default;

// static
PlatformFunctions* PlatformFunctions::GetInstance() {
  static PlatformFunctions* instance;
  if (!instance) {
    instance = new PlatformFunctions();
  }
  return instance;
}

// }  // namespace webnn::dml
