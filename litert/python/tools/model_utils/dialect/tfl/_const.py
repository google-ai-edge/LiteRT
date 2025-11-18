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
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import arith

ConstOp = arith.ConstantOp
ConstantOp = arith.ConstantOp
PseudoConstOp = arith.ConstantOp


# Overload arith.constant to be tfl.pseudo_const
core.register_mlir_transform("tfl.pseudo_const")(arith.ConstantOp)


def pseudo_const(*args, **kwargs):
  return arith.constant(*args, **kwargs)


def const(*args, **kwargs):
  return arith.constant(*args, **kwargs)


def constant(*args, **kwargs):
  return arith.constant(*args, **kwargs)
