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
"""TableGen macro definitions for Match DAG."""

from typing import Any, Callable

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import arith
from litert.python.tools.model_utils.dialect import tfl
from litert.python.tools.model_utils.match import _match_pred

__all__ = ["registry"]

registry = {}

match_pred = _match_pred.match_pred
SSAValue = irdl.SSAValue


def register(identifier: str):
  def reg(fn: Callable[[Any], None]):
    registry[identifier] = fn
    return fn

  return reg


def match_owner_op_name(name: str):
  """Returns a MatchPred that matches an op with the given name and owner."""

  def check_fn(value: SSAValue):
    op = value.owner
    return match_pred(isinstance(op, core.MlirOpBase) and op.name == name)

  return check_fn


# fmt: off
# pyformat: disabledef
register("TFL_PseudoConstOp")(match_owner_op_name(tfl.PseudoConstOp.name))
register("TFL_ConstOp")(match_owner_op_name(tfl.ConstOp.name))
register("TFL_ConstantOp")(match_owner_op_name(tfl.ConstantOp.name))
register("Arith_ConstantOp")(match_owner_op_name(arith.ConstantOp.name))

register("TFL_AbsOp")(match_owner_op_name("tfl.abs"))
register("TFL_AddNOp")(match_owner_op_name("tfl.add_n"))
register("TFL_AddOp")(match_owner_op_name("tfl.add"))
register("TFL_ArgMaxOp")(match_owner_op_name("tfl.arg_max"))
register("TFL_ArgMinOp")(match_owner_op_name("tfl.arg_min"))
register("TFL_AssignVariableOp")(match_owner_op_name("tfl.assign_variable"))
register("TFL_Atan2Op")(match_owner_op_name("tfl.atan2"))
register("TFL_AveragePool2DOp")(match_owner_op_name("tfl.average_pool_2d"))
register("TFL_BasicLSTMOp")(match_owner_op_name("tfl.basic_lstm"))
register("TFL_BatchMatMulOp")(match_owner_op_name("tfl.batch_matmul"))
register("TFL_BatchToSpaceNdOp")(match_owner_op_name("tfl.batch_to_space_nd"))
register("TFL_BidirectionalSequenceLSTMOp")(match_owner_op_name("tfl.bidirectional_sequence_lstm"))
register("TFL_BitcastOp")(match_owner_op_name("tfl.bitcast"))
register("TFL_BitwiseXorOp")(match_owner_op_name("tfl.bitwise_xor"))
register("TFL_BroadcastArgsOp")(match_owner_op_name("tfl.broadcast_args"))
register("TFL_BroadcastToOp")(match_owner_op_name("tfl.broadcast_to"))
register("TFL_BucketizeOp")(match_owner_op_name("tfl.bucketize"))
register("TFL_CallOnceOp")(match_owner_op_name("tfl.call_once"))
register("TFL_CastOp")(match_owner_op_name("tfl.cast"))
register("TFL_CeilOp")(match_owner_op_name("tfl.ceil"))
register("TFL_ComplexAbsOp")(match_owner_op_name("tfl.complex_abs"))
register("TFL_ConcatenationOp")(match_owner_op_name("tfl.concatenation"))
register("TFL_ControlNodeOp")(match_owner_op_name("tfl.control_node"))
register("TFL_Conv2DOp")(match_owner_op_name("tfl.conv_2d"))
register("TFL_Conv3DOp")(match_owner_op_name("tfl.conv_3d"))
register("TFL_Conv3DTransposeOp")(match_owner_op_name("tfl.conv_3d_transpose"))
register("TFL_CosOp")(match_owner_op_name("tfl.cos"))
register("TFL_CumsumOp")(match_owner_op_name("tfl.cumsum"))
register("TFL_CustomOp")(match_owner_op_name("tfl.custom"))
register("TFL_CustomTfOp")(match_owner_op_name("tfl.custom_tf"))
register("TFL_DensifyOp")(match_owner_op_name("tfl.densify"))
register("TFL_DepthToSpaceOp")(match_owner_op_name("tfl.depth_to_space"))
register("TFL_DepthwiseConv2DOp")(match_owner_op_name("tfl.depthwise_conv_2d"))
register("TFL_DequantizeOp")(match_owner_op_name("tfl.dequantize"))
register("TFL_DilateOp")(match_owner_op_name("tfl.dilate"))
register("TFL_DivOp")(match_owner_op_name("tfl.div"))
register("TFL_DynamicUpdateSliceOp")(match_owner_op_name("tfl.dynamic_update_slice"))
register("TFL_EluOp")(match_owner_op_name("tfl.elu"))
register("TFL_EmbeddingLookupOp")(match_owner_op_name("tfl.embedding_lookup"))
register("TFL_EqualOp")(match_owner_op_name("tfl.equal"))
register("TFL_ExpOp")(match_owner_op_name("tfl.exp"))
register("TFL_ExpandDimsOp")(match_owner_op_name("tfl.expand_dims"))
register("TFL_ExternalConstOp")(match_owner_op_name("tfl.external_const"))
register("TFL_FakeQuantOp")(match_owner_op_name("tfl.fake_quant"))
register("TFL_FillOp")(match_owner_op_name("tfl.fill"))
register("TFL_FloorDivOp")(match_owner_op_name("tfl.floor_div"))
register("TFL_FloorModOp")(match_owner_op_name("tfl.floor_mod"))
register("TFL_FloorOp")(match_owner_op_name("tfl.floor"))
register("TFL_FullyConnectedOp")(match_owner_op_name("tfl.fully_connected"))
register("TFL_GatherNdOp")(match_owner_op_name("tfl.gather_nd"))
register("TFL_GatherOp")(match_owner_op_name("tfl.gather"))
register("TFL_GeluOp")(match_owner_op_name("tfl.gelu"))
register("TFL_GreaterEqualOp")(match_owner_op_name("tfl.greater_equal"))
register("TFL_GreaterOp")(match_owner_op_name("tfl.greater"))
register("TFL_HardSwishOp")(match_owner_op_name("tfl.hard_swish"))
register("TFL_HashtableFindOp")(match_owner_op_name("tfl.hashtable_find"))
register("TFL_HashtableImportOp")(match_owner_op_name("tfl.hashtable_import"))
register("TFL_HashtableOp")(match_owner_op_name("tfl.hashtable"))
register("TFL_HashtableSizeOp")(match_owner_op_name("tfl.hashtable_size"))
register("TFL_IfOp")(match_owner_op_name("tfl.if"))
register("TFL_ImagOp")(match_owner_op_name("tfl.imag"))
register("TFL_L2NormalizationOp")(match_owner_op_name("tfl.l2_normalization"))
register("TFL_LSTMOp")(match_owner_op_name("tfl.lstm"))
register("TFL_LeakyReluOp")(match_owner_op_name("tfl.leaky_relu"))
register("TFL_LessEqualOp")(match_owner_op_name("tfl.less_equal"))
register("TFL_LessOp")(match_owner_op_name("tfl.less"))
register("TFL_LocalResponseNormalizationOp")(match_owner_op_name("tfl.local_response_normalization"))
register("TFL_LogOp")(match_owner_op_name("tfl.log"))
register("TFL_LogSoftmaxOp")(match_owner_op_name("tfl.log_softmax"))
register("TFL_LogicalAndOp")(match_owner_op_name("tfl.logical_and"))
register("TFL_LogicalNotOp")(match_owner_op_name("tfl.logical_not"))
register("TFL_LogicalOrOp")(match_owner_op_name("tfl.logical_or"))
register("TFL_LogisticOp")(match_owner_op_name("tfl.logistic"))
register("TFL_MatrixDiagOp")(match_owner_op_name("tfl.matrix_diag"))
register("TFL_MatrixSetDiagOp")(match_owner_op_name("tfl.matrix_set_diag"))
register("TFL_MaxPool2DOp")(match_owner_op_name("tfl.max_pool_2d"))
register("TFL_MaximumOp")(match_owner_op_name("tfl.maximum"))
register("TFL_MeanOp")(match_owner_op_name("tfl.mean"))
register("TFL_MinimumOp")(match_owner_op_name("tfl.minimum"))
register("TFL_MirrorPadOp")(match_owner_op_name("tfl.mirror_pad"))
register("TFL_MulOp")(match_owner_op_name("tfl.mul"))
register("TFL_MultinomialOp")(match_owner_op_name("tfl.multinomial"))
register("TFL_NegOp")(match_owner_op_name("tfl.neg"))
register("TFL_NoValueOp")(match_owner_op_name("tfl.no_value"))
register("TFL_NonMaxSuppressionV4Op")(match_owner_op_name("tfl.non_max_suppression_v4"))
register("TFL_NonMaxSuppressionV5Op")(match_owner_op_name("tfl.non_max_suppression_v5"))
register("TFL_NotEqualOp")(match_owner_op_name("tfl.not_equal"))
register("TFL_NumericVerifyOp")(match_owner_op_name("tfl.NumericVerify"))
register("TFL_OneHotOp")(match_owner_op_name("tfl.one_hot"))
register("TFL_PReluOp")(match_owner_op_name("tfl.prelu"))
register("TFL_PackOp")(match_owner_op_name("tfl.pack"))
register("TFL_PadOp")(match_owner_op_name("tfl.pad"))
register("TFL_PadV2Op")(match_owner_op_name("tfl.padv2"))
register("TFL_PolyCallOp")(match_owner_op_name("tfl.poly_call"))
register("TFL_PowOp")(match_owner_op_name("tfl.pow"))
register("TFL_QConstOp")(match_owner_op_name("tfl.pseudo_qconst"))
register("TFL_QuantizeOp")(match_owner_op_name("tfl.quantize"))
register("TFL_RFFT2dOp")(match_owner_op_name("tfl.rfft2d"))
register("TFL_RandomStandardNormalOp")(match_owner_op_name("tfl.random_standard_normal"))
register("TFL_RandomUniformOp")(match_owner_op_name("tfl.random_uniform"))
register("TFL_RangeOp")(match_owner_op_name("tfl.range"))
register("TFL_RankOp")(match_owner_op_name("tfl.rank"))
register("TFL_ReadVariableOp")(match_owner_op_name("tfl.read_variable"))
register("TFL_RealOp")(match_owner_op_name("tfl.real"))
register("TFL_ReduceAllOp")(match_owner_op_name("tfl.reduce_all"))
register("TFL_ReduceAnyOp")(match_owner_op_name("tfl.reduce_any"))
register("TFL_ReduceMaxOp")(match_owner_op_name("tfl.reduce_max"))
register("TFL_ReduceMinOp")(match_owner_op_name("tfl.reduce_min"))
register("TFL_ReduceProdOp")(match_owner_op_name("tfl.reduce_prod"))
register("TFL_Relu0To1Op")(match_owner_op_name("tfl.relu_0_to_1"))
register("TFL_Relu1Op")(match_owner_op_name("tfl.relu_n1_to_1"))
register("TFL_Relu6Op")(match_owner_op_name("tfl.relu6"))
register("TFL_ReluOp")(match_owner_op_name("tfl.relu"))
register("TFL_ReshapeOp")(match_owner_op_name("tfl.reshape"))
register("TFL_ResizeBilinearOp")(match_owner_op_name("tfl.resize_bilinear"))
register("TFL_ResizeNearestNeighborOp")(match_owner_op_name("tfl.resize_nearest_neighbor"))
register("TFL_ReverseSequenceOp")(match_owner_op_name("tfl.reverse_sequence"))
register("TFL_ReverseV2Op")(match_owner_op_name("tfl.reverse_v2"))
register("TFL_RightShiftOp")(match_owner_op_name("tfl.right_shift"))
register("TFL_RoundOp")(match_owner_op_name("tfl.round"))
register("TFL_RsqrtOp")(match_owner_op_name("tfl.rsqrt"))
register("TFL_SVDFOp")(match_owner_op_name("tfl.svdf"))
register("TFL_ScatterNdOp")(match_owner_op_name("tfl.scatter_nd"))
register("TFL_SegmentSumOp")(match_owner_op_name("tfl.segment_sum"))
register("TFL_SelectOp")(match_owner_op_name("tfl.select"))
register("TFL_SelectV2Op")(match_owner_op_name("tfl.select_v2"))
register("TFL_ShapeOp")(match_owner_op_name("tfl.shape"))
register("TFL_SignOp")(match_owner_op_name("tfl.sign"))
register("TFL_SinOp")(match_owner_op_name("tfl.sin"))
register("TFL_SliceOp")(match_owner_op_name("tfl.slice"))
register("TFL_SoftmaxOp")(match_owner_op_name("tfl.softmax"))
register("TFL_SpaceToBatchNdOp")(match_owner_op_name("tfl.space_to_batch_nd"))
register("TFL_SpaceToDepthOp")(match_owner_op_name("tfl.space_to_depth"))
register("TFL_SparseConstOp")(match_owner_op_name("tfl.pseudo_sparse_const"))
register("TFL_SparseQConstOp")(match_owner_op_name("tfl.pseudo_sparse_qconst"))
register("TFL_SparseToDenseOp")(match_owner_op_name("tfl.sparse_to_dense"))
register("TFL_SplitOp")(match_owner_op_name("tfl.split"))
register("TFL_SplitVOp")(match_owner_op_name("tfl.split_v"))
register("TFL_SqrtOp")(match_owner_op_name("tfl.sqrt"))
register("TFL_SquareOp")(match_owner_op_name("tfl.square"))
register("TFL_SquaredDifferenceOp")(match_owner_op_name("tfl.squared_difference"))
register("TFL_SqueezeOp")(match_owner_op_name("tfl.squeeze"))
register("TFL_StridedSliceOp")(match_owner_op_name("tfl.strided_slice"))
register("TFL_SubOp")(match_owner_op_name("tfl.sub"))
register("TFL_SumOp")(match_owner_op_name("tfl.sum"))
register("TFL_TanhOp")(match_owner_op_name("tfl.tanh"))
register("TFL_TileOp")(match_owner_op_name("tfl.tile"))
register("TFL_TopKV2Op")(match_owner_op_name("tfl.topk_v2"))
register("TFL_TransposeConvOp")(match_owner_op_name("tfl.transpose_conv"))
register("TFL_TransposeOp")(match_owner_op_name("tfl.transpose"))
register("TFL_UnidirectionalSequenceLSTMOp")(match_owner_op_name("tfl.unidirectional_sequence_lstm"))
register("TFL_UnidirectionalSequenceRNNOp")(match_owner_op_name("tfl.unidirectional_sequence_rnn"))
register("TFL_UniqueOp")(match_owner_op_name("tfl.unique"))
register("TFL_UnpackOp")(match_owner_op_name("tfl.unpack"))
register("TFL_UnsortedSegmentMaxOp")(match_owner_op_name("tfl.unsorted_segment_max"))
register("TFL_UnsortedSegmentMinOp")(match_owner_op_name("tfl.unsorted_segment_min"))
register("TFL_UnsortedSegmentProdOp")(match_owner_op_name("tfl.unsorted_segment_prod"))
register("TFL_UnsortedSegmentSumOp")(match_owner_op_name("tfl.unsorted_segment_sum"))
register("TFL_VarHandleOp")(match_owner_op_name("tfl.var_handle"))
register("TFL_WhereOp")(match_owner_op_name("tfl.where"))
register("TFL_WhileOp")(match_owner_op_name("tfl.while"))
register("TFL_YieldOp")(match_owner_op_name("tfl.yield"))
register("TFL_ZerosLikeOp")(match_owner_op_name("tfl.zeros_like"))
# fmt: on
# pyformat: enable
