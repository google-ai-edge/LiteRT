// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "KextBuilder.h"

#include <IOKit/IOLib.h>

#define super IOService
#define KextBuilder com_google_KextBuilder

OSDefineMetaClassAndStructors(com_google_KextBuilder, IOService);

bool KextBuilder::start(IOService *provider) {
  if (!super::start(provider)) return false;
  registerService();
  IOLog("Loaded, version %s.", OSKextGetCurrentVersionString());
  return true;
}

void KextBuilder::stop(IOService *provider) {
  IOLog("Unloaded.");
  super::stop(provider);
}

#undef super
