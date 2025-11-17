# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ModelUtils core library."""
from . import _op_building_context
from . import dialect_base
from . import pass_base
from . import pybind
from . import utils

OpBuildingContext = _op_building_context.OpBuildingContext

ModulePassBase = pass_base.ModulePassBase
RewritePatternPassBase = pass_base.RewritePatternPassBase

register_mlir_transform = dialect_base.register_mlir_transform
overload_cls_attrs = dialect_base.overload_cls_attrs
mlir_transforms = dialect_base.mlir_transforms
MlirOpBase = dialect_base.MlirOpBase
MlirAttributeBase = dialect_base.MlirAttributeBase
MlirTypeBase = dialect_base.MlirTypeBase
