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
"""TFL dialect definitions."""

# pylint: disable=redefined-builtin
# pylint: disable=g-importing-member
from ._abs import *
from ._add import *
from ._arg_max import *
from ._arg_min import *
from ._atan2 import *
from ._average_pool_2d import *
from ._batch_matmul import *
from ._bitcast import *
from ._bitwse_xor import *
from ._broadcast_args import *
from ._broadcast_to import *
from ._cast import *
from ._ceil import *
from ._concatenation import *
from ._const import *
from ._conv_2d import *
from ._cos import *
from ._cumsum import *
from ._custom import *
from ._depthwise_conv_2d import *
from ._dequantize import *
from ._div import *
from ._dynamic_update_slice import *
from ._elu import *
from ._embedding_lookup import *
from ._equal import *
from ._exp import *
from ._expand_dims import *
from ._fill import *
from ._floor import *
from ._floor_div import *
from ._floor_mod import *
from ._fully_connected import *
from ._gather import *
from ._gather_nd import *
from ._gelu import *
from ._greater import *
from ._greater_equal import *
from ._no_value import *
from ._pad_v2 import *
from ._range import *
# pylint: disable=g-multiple-import
# pylint: disable=g-bad-import-order
from ._hard_swish import HardSwishOp, hard_swish
from ._maximum import MaximumOp, maximum
from ._minimum import MinimumOp, minimum
from ._relu import ReluOp, relu
from ._split import SplitOp, split
from ._split_v import SplitVOp, split_v
from ._squeeze import SqueezeOp, squeeze
from ._sqrt import SqrtOp, sqrt
# TODO(cnchan): Update import style with dialect refactor.
from .const_bytes_attr import ConstBytesAttr
from .log import LogOp, log
from .log_softmax import LogSoftmaxOp, log_softmax
from .logistic import LogisticOp, logistic
from .max_pool_2d import MaxPool2DOp, max_pool_2d
from .mean import MeanOp, mean
from .mul import MulOp, mul
from .reshape import ReshapeOp, reshape
from .rsqrt import RsqrtOp, rsqrt
from .select import SelectOp, select
from .select_v2 import SelectV2Op, select_v2
from .shape import ShapeOp, shape
from .slice import SliceOp, slice
from .softmax import SoftmaxOp, softmax
from .split import SplitOp, split
from .sub import SubOp, sub
from .sum import SumOp, sum
from .tanh import TanhOp, tanh
from .tile import TileOp, tile
from .transpose import TransposeOp, transpose
# pylint: enable=g-importing-member
# pylint: enable=g-multiple-import
# pylint: enable=g-bad-import-order
