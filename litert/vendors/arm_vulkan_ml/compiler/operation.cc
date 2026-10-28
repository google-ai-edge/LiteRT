// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/extended_ops.h"
#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <limits>
#include <set>
#include <utility>
#include "TosaForSpirvCodegen.hpp"
#include "litert/c/litert_op_options.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d_adapter.h"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::Graph;
using tfsc::tosa::ResId;
using TosaTensor = tfsc::tosa::Tensor;
std::optional<QuantizedTensor> ReadTensor(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type || type->Layout().HasStrides()) return std::nullopt;
  QuantizedTensor result{{}, 1.0, 0};
  result.floating = type->ElementType() == ElementType::Float32;
  if (!result.floating) {
    if (type->ElementType() != ElementType::Int8 ||
        t.QTypeId() != QuantizationTypeId::PerTensor) return std::nullopt;
    auto q = t.PerTensorQuantization();
    if (!std::isfinite(q.scale) || q.scale <= 0 || q.zero_point < -128 || q.zero_point > 127) return std::nullopt;
    result.scale = q.scale; result.zero_point = q.zero_point;
  }
  uint64_t count = 1;
  auto dims = type->Layout().Dimensions();
  if (dims.size() > 6) return std::nullopt;
  for (auto d : dims) {
    if (d <= 0 || d > 8192 || count > uint64_t{INT32_MAX}/d) return std::nullopt;
    count *= d; result.shape.push_back(d);
  }
  return result;
}
uint64_t Count(const std::vector<uint32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}
bool SameQuant(const QuantizedTensor& a, const QuantizedTensor& b) {
  return a.floating == b.floating && a.scale == b.scale && a.zero_point == b.zero_point;
}
std::optional<Activation> ActivationCode(uint32_t code) {
  switch (code) { case 0: return Activation::kNone; case 1: return Activation::kRelu; case 3: return Activation::kRelu6; default: return std::nullopt; }
}
template <typename T>
ResId Constant(Graph& graph, const std::vector<T>& data, const std::vector<uint32_t>& shape,
               std::vector<GraphConstant>& constants) {
  const auto type = tfsc::tosa::GetDataType<T>();
  TosaTensor tensor(type, shape);
  GraphConstant constant{tensor, std::vector<uint8_t>(data.size()*sizeof(T))};
  std::memcpy(constant.bytes.data(), data.data(), constant.bytes.size());
  constants.push_back(std::move(constant));
  return graph.AddGraphConstant(tensor);
}
ResId Rescale(Graph& g, ResId input, const std::vector<uint32_t>& shape,
    DataType input_type, DataType output_type, int32_t izp, int32_t ozp,
    const std::vector<Scale32>& scales, std::vector<GraphConstant>& constants,
    bool per_channel = false) {
  std::vector<int32_t> mult; std::vector<int8_t> shift;
  for (auto scale : scales) { mult.push_back(scale.multiplier); shift.push_back(scale.shift); }
  auto multiplier = Constant(g, mult, {static_cast<uint32_t>(mult.size())}, constants);
  auto shifts = Constant(g, shift, {static_cast<uint32_t>(shift.size())}, constants);
  auto input_zp = g.AddTensorConstant(Attribute(izp, input_type));
  auto output_zp = g.AddTensorConstant(Attribute(ozp, output_type));
  return g.AddRescaleOperator(input,multiplier,shifts,input_zp,output_zp,
      Attribute(true),Attribute(int32_t{1}),Attribute(per_channel),Attribute(false),
      Attribute(false),TosaTensor(output_type,shape));
}
ResId Clamp(Graph& g, ResId input, const OperationPlan& p) {
  return g.AddClampOperator(input, Attribute(static_cast<int8_t>(p.activation.minimum)),
      Attribute(static_cast<int8_t>(p.activation.maximum)), Attribute(int32_t{1}),
      TosaTensor(DataType::int8_t,p.output.shape));
}
ResId Reshape(Graph& g, ResId input, DataType type, const std::vector<uint32_t>& shape) {
  return g.AddReshapeOperator(input, g.AddTensorConstant(Attribute(shape)), TosaTensor(type,shape));
}
bool ChannelScales(const Tensor& t, uint32_t channels, std::vector<double>& result) {
  if (t.QTypeId() == QuantizationTypeId::PerTensor) {
    auto q=t.PerTensorQuantization(); if(q.zero_point) return false;
    result.assign(channels,q.scale); return true;
  }
  if(t.QTypeId()!=QuantizationTypeId::PerChannel) return false;
  auto q=t.PerChannelQuantization();
  if(q.quantized_dimension!=0 || q.num_channels!=channels) return false;
  for(uint32_t i=0;i<channels;++i) { if(q.zero_points[i]) return false; result.push_back(q.scales[i]); }
  return true;
}
}  // namespace

bool AnalyzeFloatConvolution(const Op& op, OperationPlan& p, std::string& reason) {
  auto in=op.Inputs();
  if(in.size()!=3 || p.inputs[0].shape.size()!=4 || p.output.shape.size()!=4) return false;
  auto wt=in[1].RankedTensorType(), bt=in[2].RankedTensorType();
  if(!wt || !bt || wt->ElementType()!=ElementType::Float32 || bt->ElementType()!=ElementType::Float32 ||
      wt->Layout().HasStrides() || bt->Layout().HasStrides() ||
      wt->Layout().Dimensions().size()!=4 || bt->Layout().Dimensions().size()!=1) return false;
  for(auto d:wt->Layout().Dimensions()) {if(d<=0 || d>8192)return false;p.weight_shape.push_back(d);}
  auto weights=in[1].WeightsData<float>(),bias=in[2].WeightsData<float>();
  if(!weights || !bias || weights->size()!=Count(p.weight_shape) ||
      bias->size()!=p.output.shape[3] || bt->Layout().Dimensions()[0]!=p.output.shape[3]) return false;
  p.float_weights.assign(weights->begin(),weights->end());p.float_bias.assign(bias->begin(),bias->end());
  for(float v:p.float_weights)if(!std::isfinite(v))return false;
  for(float v:p.float_bias)if(!std::isfinite(v))return false;
  const bool dw=p.code==kLiteRtOpCodeTflDepthwiseConv2d;
  if(p.inputs[0].shape[0]!=p.output.shape[0])return false;
  if(dw) {
    int32_t multiplier;
    if(LiteRtGetDepthwiseConv2dDepthMultiplierOption(op.Get(),&multiplier)!=kLiteRtStatusOk || multiplier!=1 ||
       p.weight_shape[0]!=1 || p.weight_shape[3]!=p.inputs[0].shape[3] || p.output.shape[3]!=p.inputs[0].shape[3])return false;
  } else if(p.weight_shape[0]!=p.output.shape[3] || p.weight_shape[3]!=p.inputs[0].shape[3])return false;
  auto padding_fn=dw?LiteRtGetDepthwiseConv2dPaddingOption:LiteRtGetConv2dPaddingOption;
  auto activation_fn=dw?LiteRtGetDepthwiseConv2dFusedActivationOption:LiteRtGetConv2dFusedActivationOption;
  auto sh=dw?LiteRtGetDepthwiseConv2dStrideHOption:LiteRtGetConv2dStrideHOption;
  auto sw=dw?LiteRtGetDepthwiseConv2dStrideWOption:LiteRtGetConv2dStrideWOption;
  auto dh=dw?LiteRtGetDepthwiseConv2dDilationHOption:LiteRtGetConv2dDilationHOption;
  auto dw_fn=dw?LiteRtGetDepthwiseConv2dDilationWOption:LiteRtGetConv2dDilationWOption;
  uint32_t padding;p.stride.resize(2);p.dilation.resize(2);p.padding.assign(4,0);
  if(padding_fn(op.Get(),&padding)!=kLiteRtStatusOk || padding>1 ||
     activation_fn(op.Get(),&p.float_activation)!=kLiteRtStatusOk || !ActivationCode(p.float_activation) ||
     sh(op.Get(),&p.stride[0])!=kLiteRtStatusOk || sw(op.Get(),&p.stride[1])!=kLiteRtStatusOk ||
     dh(op.Get(),&p.dilation[0])!=kLiteRtStatusOk || dw_fn(op.Get(),&p.dilation[1])!=kLiteRtStatusOk)return false;
  p.cropped_shape=p.inputs[0].shape;
  for(int axis=0;axis<2;++axis) {
    if(p.stride[axis]<=0 || p.dilation[axis]<=0)return false;
    int64_t n=p.inputs[0].shape[axis+1], k=(static_cast<int64_t>(p.weight_shape[axis+1])-1)*p.dilation[axis]+1;
    int64_t expected=padding==0?(n+p.stride[axis]-1)/p.stride[axis]:(n-k)/p.stride[axis]+1;
    if((padding==1 && n<k) || expected!=p.output.shape[axis+1])return false;
    int64_t total=padding==0?std::max(int64_t{0},(expected-1)*p.stride[axis]+k-n):0;
    if(total>INT32_MAX)return false;
    p.padding[2*axis]=total/2;p.padding[2*axis+1]=total-total/2;
    p.cropped_shape[axis+1]-=(n+total-k)%p.stride[axis];
  }
  return true;
}

std::optional<OperationPlan> AnalyzeOperation(const Op& op, std::string& reason) {
  auto reject = [&](const char* s)->std::optional<OperationPlan> { reason=s; return std::nullopt; };
  OperationPlan p{}; p.code=op.Code();
  auto in=op.Inputs(); auto out=op.Outputs();
  if (IsPortableOperation(p.code) && !out.empty() && out[0].ElementType() == ElementType::Float32) {
    p.portable = AnalyzePortableOperation(op, reason);
    if (!p.portable) return std::nullopt;
    p.runtime_inputs = p.portable->inputs;
    p.inputs.push_back({p.portable->shapes[0],1,0,true});
    return p;
  }
  if(in.empty() || out.size()!=1) return reject("Expected data inputs and exactly one output");
  const int data_index = p.code == kLiteRtOpCodeTflTransposeConv ? 2 : 0;
  if (data_index >= in.size()) return reject("Missing data input");
  auto input=ReadTensor(in[data_index]); auto output=ReadTensor(out[0]);
  if(!input || !output || output->shape.empty()) return reject("Requires static dense int8 tensors with finite per-tensor quantization");
  p.inputs.push_back(*input); p.output=*output; p.runtime_inputs={data_index};
  if (IsExtendedOperation(p.code, input->floating)) {
    if (!AnalyzeExtendedOperation(op, p, reason)) return std::nullopt;
    return p;
  }
  if (input->floating && output->floating &&
      (p.code == kLiteRtOpCodeTflAdd || p.code == kLiteRtOpCodeTflMul)) {
    if (in.size() != 2) return reject("FP32 arithmetic requires two inputs");
    auto rhs = ReadTensor(in[1]);
    if (!rhs || !rhs->floating) return reject("FP32 arithmetic requires matching types");
    const size_t rank = std::max(input->shape.size(), rhs->shape.size());
    auto lhs_shape = input->shape, rhs_shape = rhs->shape;
    lhs_shape.insert(lhs_shape.begin(), rank - lhs_shape.size(), 1);
    rhs_shape.insert(rhs_shape.begin(), rank - rhs_shape.size(), 1);
    std::vector<uint32_t> shape(rank);
    for (size_t i = 0; i < rank; ++i) {
      if (lhs_shape[i] != rhs_shape[i] && lhs_shape[i] != 1 && rhs_shape[i] != 1)
        return reject("FP32 arithmetic broadcast mismatch");
      shape[i] = std::max(lhs_shape[i], rhs_shape[i]);
    }
    if (shape != output->shape) return reject("FP32 arithmetic output shape mismatch");
    auto activation_fn = p.code == kLiteRtOpCodeTflAdd
        ? LiteRtGetAddFusedActivationOption : LiteRtGetMulFusedActivationOption;
    if (activation_fn(op.Get(), &p.float_activation) != kLiteRtStatusOk ||
        !ActivationCode(p.float_activation)) return reject("Unsupported FP32 activation");
    p.inputs.push_back(*rhs);
    p.runtime_inputs.push_back(1);
    reason.clear();
    return p;
  }
  if ((input->floating || output->floating) &&
      p.code != kLiteRtOpCodeTflTranspose && p.code != kLiteRtOpCodeTflPad &&
      p.code != kLiteRtOpCodeTflHardSwish && p.code != kLiteRtOpCodeTflQuantize &&
      p.code != kLiteRtOpCodeTflDequantize &&
      p.code != kLiteRtOpCodeTflLogistic && p.code != kLiteRtOpCodeTflAveragePool2d &&
      p.code != kLiteRtOpCodeTflReshape) {
    if ((p.code == kLiteRtOpCodeTflConv2d || p.code == kLiteRtOpCodeTflDepthwiseConv2d) &&
        input->floating && output->floating && AnalyzeFloatConvolution(op, p, reason)) return p;
    return reject("Unsupported floating-point operation or convolution geometry");
  }
  if(p.code==kLiteRtOpCodeTflConv2d || p.code==kLiteRtOpCodeTflDepthwiseConv2d) {
    p.convolution=AnalyzeConv2d(op,reason);
    if(!p.convolution) return std::nullopt;
    return p;
  }
  auto constant_ints=[&](int index)->bool {
    if(index>=in.size()) return false;
    auto type=in[index].RankedTensorType();
    if(!type || type->ElementType()!=ElementType::Int32 || type->Layout().HasStrides()) return false;
    auto values=in[index].WeightsData<int32_t>(); if(!values) return false;
    p.parameters.assign(values->begin(),values->end()); return true;
  };
  switch(p.code) {
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflRelu6: {
      if(in.size()!=1 || input->shape!=output->shape)return reject("RELU requires matching shapes");
      auto scale=MakeScale32(input->scale/output->scale);
      auto activation=ActivationRange(p.code==kLiteRtOpCodeTflRelu?Activation::kRelu:Activation::kRelu6,output->scale,output->zero_point);
      if(!scale || !activation)return reject("Invalid RELU quantization");
      p.activation = *activation;
      if (SameQuant(*input, *output)) break;
      for(int code=-128;code<=127;++code) {
        auto first=*scale;first.shift=std::min<int8_t>(first.shift,31);
        auto value=ApplyScale32(code-input->zero_point,first);
        if(!value)return reject("RELU scale range invalid");
        int shift=scale->shift-first.shift;int64_t rounded=*value;
        if(shift) {
          int64_t magnitude=(std::abs(rounded)+(int64_t{1}<<(shift-1)))/(int64_t{1}<<shift);
          rounded=rounded<0?-magnitude:magnitude;
        }
        p.table.push_back(std::clamp(rounded+output->zero_point,int64_t{activation->minimum},int64_t{activation->maximum}));
      }
      break;
    }
    case kLiteRtOpCodeTflReshape: {
      if(in.size()!=2 || !SameQuant(*input,*output) || Count(input->shape)!=Count(output->shape) || !constant_ints(1) || p.parameters.size()!=output->shape.size())
        return reject("RESHAPE requires a constant compatible shape and unchanged quantization");
      int inferred=0;
      for(size_t i=0;i<p.parameters.size();++i) {
        if(p.parameters[i]==-1)++inferred;
        else if(p.parameters[i]<=0 || static_cast<uint32_t>(p.parameters[i])!=output->shape[i])return reject("RESHAPE shape disagrees");
      }
      if(inferred>1)return reject("RESHAPE has multiple inferred dimensions");
      break;
    }
    case kLiteRtOpCodeTflSlice: {
      if(in.size()!=3 || !SameQuant(*input,*output) || !constant_ints(1) || p.parameters.size()!=input->shape.size() || output->shape.size()!=input->shape.size())
        return reject("SLICE requires constant starts/sizes and matching quantization");
      p.kernel=p.parameters;
      if(!constant_ints(2) || p.parameters.size()!=input->shape.size())return reject("Invalid SLICE size");
      for(size_t i=0;i<p.parameters.size();++i) {
        int64_t start=p.kernel[i],size=p.parameters[i];
        if(size==-1)size=static_cast<int64_t>(input->shape[i])-start;
        if(start<0 || size<=0 || start+size>input->shape[i] || size!=output->shape[i])return reject("SLICE bounds mismatch");
        p.parameters[i]=size;
      }
      break;
    }
    case kLiteRtOpCodeTflConcatenation: {
      uint32_t fused;
      if(LiteRtGetConcatenationAxisOption(op.Get(),&p.axis)!=kLiteRtStatusOk ||
         LiteRtGetConcatenationFusedActivationOption(op.Get(),&fused)!=kLiteRtStatusOk || fused)return reject("CONCAT requires no fused activation");
      if(p.axis<0)p.axis+=output->shape.size();
      if(p.axis<0 || p.axis>=output->shape.size())return reject("Invalid CONCAT axis");
      p.inputs.clear();p.runtime_inputs.clear();uint64_t total=0;
      for(size_t n=0;n<in.size();++n) {
        auto tensor=ReadTensor(in[n]);
        if(!tensor || !SameQuant(*tensor,*output) || tensor->shape.size()!=output->shape.size())return reject("CONCAT requires matching types/quantization and rank");
        for(size_t i=0;i<output->shape.size();++i)if(i!=p.axis && tensor->shape[i]!=output->shape[i])return reject("CONCAT shape mismatch");
        total+=tensor->shape[p.axis];p.inputs.push_back(*tensor);p.runtime_inputs.push_back(n);
      }
      if(total!=output->shape[p.axis])return reject("CONCAT output extent mismatch");
      break;
    }
    case kLiteRtOpCodeTflDequantize:
    case kLiteRtOpCodeTflQuantize: {
      if (p.code == kLiteRtOpCodeTflQuantize && !input->floating && !output->floating) {
        if (in.size() != 1 || input->shape != output->shape)
          return reject("INT8 QUANTIZE requires identical shapes");
        if (SameQuant(*input, *output)) break;
        auto table = RequantizeTable(input->scale, input->zero_point,
                                    output->scale, output->zero_point);
        if (!table) return reject("INT8 QUANTIZE scale range is not supported");
        p.table = std::move(*table);
        break;
      }
      const bool dequant = p.code == kLiteRtOpCodeTflDequantize;
      if(in.size()!=1 || input->shape!=output->shape || input->floating!=!dequant || output->floating!=dequant)
        return reject("Requires int8/float32 conversion with identical shapes");
      break;
    }
    case kLiteRtOpCodeTflLogistic: {
      if (in.size() != 1 || input->shape != output->shape ||
          input->floating != output->floating)
        return reject("LOGISTIC requires matching shapes and types");
      if (!input->floating) {
        if (output->scale != 1.0 / 256 || output->zero_point != -128)
          return reject("INT8 LOGISTIC requires scale 1/256 and zero point -128");
        // Match LiteRT's reference LUT using float arithmetic, then store in
        // TOSA TABLE's signed input order (-128 through 127).
        for (int code = -128; code <= 127; ++code) {
          float x = static_cast<float>(input->scale) * (code - input->zero_point);
          float y = 1.0f / (1.0f + std::exp(-x));
          int value = static_cast<int>(std::round(y * 256.0f)) - 128;
          p.table.push_back(std::clamp(value, -128, 127));
        }
      }
      break;
    }
    case kLiteRtOpCodeTflHardSwish: {
      if(in.size()!=1 || input->shape!=output->shape || input->floating!=output->floating)
        return reject("HARD_SWISH requires matching shapes and types");
      if(!input->floating) {
        auto table=HardSwishTable(input->scale,input->zero_point,output->scale,output->zero_point);
        if(!table)return reject("HARD_SWISH fixed-point scales are not representable");
        p.table=std::move(*table);
      }
      break;
    }
    case kLiteRtOpCodeTflMul: {
      if(in.size()!=2) return reject("MUL requires two inputs");
      auto rhs=ReadTensor(in[1]);
      if(!rhs || rhs->floating) return reject("MUL requires affine int8 inputs");
      const size_t rank=std::max(input->shape.size(),rhs->shape.size());
      auto lhs_shape=input->shape, rhs_shape=rhs->shape;
      lhs_shape.insert(lhs_shape.begin(),rank-lhs_shape.size(),1);
      rhs_shape.insert(rhs_shape.begin(),rank-rhs_shape.size(),1);
      std::vector<uint32_t> shape(rank);
      for(size_t i=0;i<rank;++i) {
        if(lhs_shape[i]!=rhs_shape[i] && lhs_shape[i]!=1 && rhs_shape[i]!=1) return reject("MUL broadcast mismatch");
        shape[i]=std::max(lhs_shape[i],rhs_shape[i]);
      }
      if(shape!=output->shape) return reject("MUL output shape mismatch");
      uint32_t fused;
      if(LiteRtGetMulFusedActivationOption(op.Get(),&fused)!=kLiteRtStatusOk) return reject("Missing MUL activation");
      auto activation=ActivationCode(fused); if(!activation) return reject("Unsupported MUL activation");
      p.activation=*ActivationRange(*activation,output->scale,output->zero_point);
      auto scale=MakeScale32(input->scale*rhs->scale/output->scale);
      int64_t lo=0,hi=0;
      for(int64_t a:{-128-input->zero_point,127-input->zero_point})
        for(int64_t b:{-128-rhs->zero_point,127-rhs->zero_point}) {lo=std::min(lo,a*b);hi=std::max(hi,a*b);}
      if(!scale || !IsScale32RangeValid(*scale,lo,hi)) return reject("MUL rescale range invalid");
      p.inputs.push_back(*rhs);p.runtime_inputs.push_back(1);p.scales.push_back(*scale);
      p.ranges.push_back({static_cast<int32_t>(lo),static_cast<int32_t>(hi)});
      break;
    }
    case kLiteRtOpCodeTflTranspose: {
      if(in.size()!=2 || !constant_ints(1) || !SameQuant(*input,*output) ||
          p.parameters.size()!=input->shape.size() || output->shape.size()!=input->shape.size())
        return reject("TRANSPOSE requires a constant permutation and matching quantization");
      std::set<int32_t> axes;
      for(size_t i=0;i<p.parameters.size();++i) {
        int axis=p.parameters[i];
        if(axis<0 || axis>=input->shape.size() || !axes.insert(axis).second || output->shape[i]!=input->shape[axis])
          return reject("Invalid TRANSPOSE permutation/output shape");
      }
      break;
    }
    case kLiteRtOpCodeTflPad:
    case kLiteRtOpCodeTflPadv2: {
      if(in.size()!=(p.code==kLiteRtOpCodeTflPadv2?3:2) || !constant_ints(1) || !SameQuant(*input,*output) ||
          p.parameters.size()!=input->shape.size()*2 || output->shape.size()!=input->shape.size())
        return reject("PAD requires constant nonnegative padding and matching quantization");
      p.pad_value=input->zero_point;
      if(p.code==kLiteRtOpCodeTflPadv2) {
        auto fill=ReadTensor(in[2]);auto value=in[2].WeightsData<int8_t>();
        if(!fill || !SameQuant(*fill,*input) || !value || value->size()!=1)return reject("PADV2 requires a constant scalar with matching quantization");
        p.pad_value=(*value)[0];
      }
      for(size_t i=0;i<input->shape.size();++i) {
        if(p.parameters[2*i]<0 || p.parameters[2*i+1]<0 ||
           uint64_t{input->shape[i]}+p.parameters[2*i]+p.parameters[2*i+1]!=output->shape[i])
          return reject("Invalid PAD shape");
      }
      break;
    }
    case kLiteRtOpCodeTflAdd: {
      if(in.size()!=2) return reject("ADD requires two inputs");
      auto rhs=ReadTensor(in[1]);
      if(!rhs || rhs->floating) return reject("ADD requires affine int8 inputs");
      const size_t rank=std::max(input->shape.size(),rhs->shape.size());
      auto lhs_shape=input->shape, rhs_shape=rhs->shape;
      lhs_shape.insert(lhs_shape.begin(),rank-lhs_shape.size(),1);
      rhs_shape.insert(rhs_shape.begin(),rank-rhs_shape.size(),1);
      std::vector<uint32_t> shape(rank);
      for(size_t i=0;i<rank;++i) {
        if(lhs_shape[i]!=rhs_shape[i] && lhs_shape[i]!=1 && rhs_shape[i]!=1) return reject("ADD broadcast mismatch");
        shape[i]=std::max(lhs_shape[i],rhs_shape[i]);
      }
      if(shape!=output->shape) return reject("ADD output shape mismatch");
      uint32_t fused;
      if(LiteRtGetAddFusedActivationOption(op.Get(),&fused)!=kLiteRtStatusOk)
        return reject("Cannot read ADD activation");
      auto activation=ActivationCode(fused);
      if(!activation) return reject("Unsupported ADD activation");
      p.activation=*ActivationRange(*activation,output->scale,output->zero_point);
      p.inputs.push_back(*rhs); p.runtime_inputs.push_back(1);
      const double twice_max=2*std::max(input->scale,rhs->scale);
      for(double scale : {input->scale/twice_max*1048576, rhs->scale/twice_max*1048576,
                         twice_max/(1048576*output->scale)}) {
        auto encoded=MakeScale32(scale); if(!encoded) return reject("ADD scale not representable");
        p.scales.push_back(*encoded);
      }
      int64_t minimum=0,maximum=0;
      for(int i=0;i<2;++i) {
        int32_t lo=-128-p.inputs[i].zero_point, hi=127-p.inputs[i].zero_point;
        if(!IsScale32RangeValid(p.scales[i],lo,hi)) return reject("ADD input rescale range invalid");
        minimum+=*ApplyScale32(lo,p.scales[i]); maximum+=*ApplyScale32(hi,p.scales[i]);
      }
      if(!IsScale32RangeValid(p.scales[2],minimum,maximum)) return reject("ADD accumulator range invalid");
      p.ranges.push_back({static_cast<int32_t>(minimum),static_cast<int32_t>(maximum)});
      break;
    }
    case kLiteRtOpCodeTflMaxPool2d:
    case kLiteRtOpCodeTflAveragePool2d: {
      if(in.size()!=1 || input->shape.size()!=4 || output->shape.size()!=4 ||
          !SameQuant(*input,*output) || input->shape[0]!=output->shape[0] || input->shape[3]!=output->shape[3])
        return reject("Pool requires NHWC, preserved batch/channels and quantization");
      bool avg=p.code==kLiteRtOpCodeTflAveragePool2d;
      auto get_padding=avg?LiteRtGetAveragePool2dPaddingOption:LiteRtGetMaxPool2dPaddingOption;
      auto get_activation=avg?LiteRtGetAveragePool2dFusedActivationOption:LiteRtGetMaxPool2dFusedActivationOption;
      auto get_h=avg?LiteRtGetAveragePool2dFilterHeightOption:LiteRtGetMaxPool2dFilterHeightOption;
      auto get_w=avg?LiteRtGetAveragePool2dFilterWidthOption:LiteRtGetMaxPool2dFilterWidthOption;
      auto get_sh=avg?LiteRtGetAveragePool2dStrideHOption:LiteRtGetMaxPool2dStrideHOption;
      auto get_sw=avg?LiteRtGetAveragePool2dStrideWOption:LiteRtGetMaxPool2dStrideWOption;
      uint32_t padding,fused; p.kernel.resize(2); p.stride.resize(2);
      if(get_padding(op.Get(),&padding)!=kLiteRtStatusOk || padding!=1 ||
         get_activation(op.Get(),&fused)!=kLiteRtStatusOk ||
         get_h(op.Get(),&p.kernel[0])!=kLiteRtStatusOk || get_w(op.Get(),&p.kernel[1])!=kLiteRtStatusOk ||
         get_sh(op.Get(),&p.stride[0])!=kLiteRtStatusOk || get_sw(op.Get(),&p.stride[1])!=kLiteRtStatusOk)
        return reject("Pool currently requires VALID padding");
      auto activation=ActivationCode(fused); if(!activation) return reject("Unsupported pool activation");
      p.float_activation=fused;
      p.activation=*ActivationRange(*activation,output->scale,output->zero_point);
      p.cropped_shape=input->shape;
      for(int axis=0;axis<2;++axis) {
        int64_t d=input->shape[axis+1], k=p.kernel[axis], stride=p.stride[axis];
        if(k<=0 || stride<=0 || d<k || (d-k)/stride+1!=output->shape[axis+1])
          return reject("Invalid pool geometry");
        p.cropped_shape[axis+1]-=(d-k)%stride;
      }
      // Support global pooling and disjoint VALID tiles. Overlapping windows
      // and padding need their own divisor and rounding analysis.
      const bool global = output->shape[1] == 1 && output->shape[2] == 1 &&
          p.kernel[0] == input->shape[1] && p.kernel[1] == input->shape[2];
      const bool tiled = !input->floating && p.stride == p.kernel;
      if (avg && ((!global && !tiled) ||
          uint64_t{static_cast<uint32_t>(p.kernel[0])} * p.kernel[1] * 255 > INT32_MAX))
        return reject("AVG_POOL requires a bounded global or disjoint int8 VALID window");
      break;
    }
    case kLiteRtOpCodeTflSum: {
      bool keep;
      if(in.size()!=2 || !constant_ints(1) || p.parameters.empty() ||
         LiteRtGetSumKeepDimsOption(op.Get(),&keep)!=kLiteRtStatusOk || !keep)
        return reject("SUM requires constant axes and keep_dims=true");
      std::set<int32_t> axes; auto shape=input->shape; uint64_t n=1;
      for(auto& axis:p.parameters) {
        if(axis<0) axis+=shape.size();
        if(axis<0 || axis>=shape.size() || !axes.insert(axis).second) return reject("Invalid SUM axes");
        n*=shape[axis]; shape[axis]=1;
      }
      if(shape!=output->shape || n>uint64_t{INT32_MAX}/255) return reject("SUM shape or accumulation range invalid");
      auto scale=MakeScale32(input->scale/output->scale);
      if(!scale || !IsScale32RangeValid(*scale,(-128-int64_t{input->zero_point})*static_cast<int64_t>(n),
          (127-int64_t{input->zero_point})*static_cast<int64_t>(n))) return reject("SUM rescale range invalid");
      p.scales.push_back(*scale); break;
    }
    case kLiteRtOpCodeTflFullyConnected: {
      if(in.size()!=3) return reject("FC requires constant weights/bias");
      auto wt=in[1].RankedTensorType(); auto bt=in[2].RankedTensorType();
      if(!wt || !bt || wt->ElementType()!=ElementType::Int8 || bt->ElementType()!=ElementType::Int32 ||
         wt->Layout().HasStrides() || bt->Layout().HasStrides() ||
         wt->Layout().Dimensions().size()!=2 || bt->Layout().Dimensions().size()!=1)
        return reject("FC requires dense int8 matrix weights and int32 vector bias");
      auto dims=wt->Layout().Dimensions();
      if(dims[0]<=0 || dims[1]<=0 || bt->Layout().Dimensions()[0]!=dims[0] ||
         Count(input->shape)%dims[1] || output->shape!=std::vector<uint32_t>{static_cast<uint32_t>(Count(input->shape)/dims[1]),static_cast<uint32_t>(dims[0])})
        return reject("FC shapes disagree");
      uint32_t format,fused; bool keep;
      if(LiteRtGetFullyConnectedWeightsFormatOption(op.Get(),&format)!=kLiteRtStatusOk || format ||
         LiteRtGetFullyConnectedKeepNumDimsOption(op.Get(),&keep)!=kLiteRtStatusOk || keep ||
         LiteRtGetFullyConnectedFusedActivationOption(op.Get(),&fused)!=kLiteRtStatusOk)
        return reject("FC requires default weight format and fixed dimensions");
      // asymmetric_quantize_inputs only affects hybrid float-input FC. The
      // fixed int8 path above uses the tensor's explicit scale and zero point.
      auto activation=ActivationCode(fused); if(!activation) return reject("Unsupported FC activation");
      Conv2dSpec s{}; s.input_shape={output->shape[0],1,1,static_cast<uint32_t>(dims[1])};
      s.weight_shape={static_cast<uint32_t>(dims[0]),1,1,static_cast<uint32_t>(dims[1])}; s.output_shape={output->shape[0],1,1,static_cast<uint32_t>(dims[0])};
      s.stride={1,1};s.dilation={1,1};s.activation=*activation;
      s.input_scale=input->scale;s.input_zero_point=input->zero_point;
      s.output_scale=output->scale;s.output_zero_point=output->zero_point;
      auto weights=in[1].WeightsData<int8_t>(); auto bias=in[2].WeightsData<int32_t>();
      if(!weights || !bias || !ChannelScales(in[1],dims[0],s.weight_scales) || !ChannelScales(in[2],dims[0],s.bias_scales))
        return reject("FC constant/quantization data invalid");
      s.weights.assign(weights->begin(),weights->end());s.bias.assign(bias->begin(),bias->end());
      p.convolution=AnalyzeConv2d(std::move(s),reason);
      if(!p.convolution) return std::nullopt;
      p.activation=p.convolution->activation_range; break;
    }
    default: return reject("Operator lowering not implemented");
  }
  reason.clear(); return p;
}

ResId EmitFloatOperation(Graph& g,const std::vector<ResId>& inputs,const OperationPlan& p,
                         std::vector<GraphConstant>& constants) {
  const auto& q=p.inputs[0];auto value=inputs[0];
  TosaTensor ftype(DataType::float32_t,p.output.shape);
  auto scalar=[&](float x) {return Constant(g,std::vector<float>{x},std::vector<uint32_t>(q.shape.size(),1),constants);};
  auto mul=[&](ResId a,ResId b) {return g.AddMulOperator(a,b,g.AddTensorConstant(Attribute(int8_t{0})),ftype);};
  switch(p.code) {
    case kLiteRtOpCodeTflDequantize:
      value=g.AddCastOperator(value,ftype);
      value=g.AddSubOperator(value,scalar(static_cast<float>(q.zero_point)),ftype);
      return mul(value,scalar(static_cast<float>(q.scale)));
    case kLiteRtOpCodeTflQuantize: {
      // Match affine quantization: round the scaled value before adding zp.
      value=mul(value,scalar(static_cast<float>(1.0/p.output.scale)));
      auto positive=g.AddGreaterEqualOperator(value,scalar(0),TosaTensor(DataType::bool_t,p.output.shape));
      auto up=g.AddFloorOperator(g.AddAddOperator(value,scalar(.5f),ftype),ftype);
      auto negative=g.AddSubOperator(scalar(0),value,ftype);
      auto down=g.AddSubOperator(scalar(0),g.AddFloorOperator(g.AddAddOperator(negative,scalar(.5f),ftype),ftype),ftype);
      value=g.AddSelectOperator(positive,up,down,ftype);
      value=g.AddAddOperator(value,scalar(static_cast<float>(p.output.zero_point)),ftype);
      value=g.AddClampOperator(value,Attribute(-128.0f),Attribute(127.0f),Attribute(int32_t{1}),ftype);
      return g.AddCastOperator(value,TosaTensor(DataType::int8_t,p.output.shape));
    }
    case kLiteRtOpCodeTflReshape:return Reshape(g,value,DataType::float32_t,p.output.shape);
    case kLiteRtOpCodeTflTranspose:return g.AddTransposeOperator(value,Attribute(p.parameters),ftype);
    case kLiteRtOpCodeTflPad:return g.AddPadOperator(value,g.AddTensorConstant(Attribute(p.parameters)),g.AddTensorConstant(Attribute(0.0f)),ftype);
    case kLiteRtOpCodeTflLogistic:
      return g.AddSigmoidOperator(value, ftype);
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflMul: {
      auto lhs_shape = p.inputs[0].shape, rhs_shape = p.inputs[1].shape;
      lhs_shape.insert(lhs_shape.begin(), p.output.shape.size() - lhs_shape.size(), 1);
      rhs_shape.insert(rhs_shape.begin(), p.output.shape.size() - rhs_shape.size(), 1);
      auto lhs = Reshape(g, value, DataType::float32_t, lhs_shape);
      auto rhs = Reshape(g, inputs[1], DataType::float32_t, rhs_shape);
      value = p.code == kLiteRtOpCodeTflAdd ? g.AddAddOperator(lhs, rhs, ftype) : mul(lhs, rhs);
      if (p.float_activation)
        value = g.AddClampOperator(value, Attribute(0.0f),
            Attribute(p.float_activation == 3 ? 6.0f : std::numeric_limits<float>::infinity()),
            Attribute(int32_t{1}), ftype);
      return value;
    }
    case kLiteRtOpCodeTflAveragePool2d: {
      auto zero = g.AddTensorConstant(Attribute(int8_t{0}));
      value = g.AddAvgPool2dOperator(value, zero, zero, Attribute(p.kernel),
          Attribute(p.stride), Attribute(std::vector<int32_t>(4, 0)), Attribute(int32_t{3}), ftype);
      if (p.float_activation)
        value = g.AddClampOperator(value, Attribute(0.0f),
            Attribute(p.float_activation == 3 ? 6.0f : std::numeric_limits<float>::infinity()),
            Attribute(int32_t{1}), ftype);
      return value;
    }
    case kLiteRtOpCodeTflHardSwish: {
      auto gate=g.AddAddOperator(value,scalar(3),ftype);
      gate=g.AddClampOperator(gate,Attribute(0.0f),Attribute(6.0f),Attribute(int32_t{1}),ftype);
      return mul(value,mul(gate,scalar(1.0f/6.0f)));
    }
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflDepthwiseConv2d: {
      if(p.cropped_shape!=q.shape)value=g.AddSliceOperator(value,g.AddTensorConstant(Attribute(std::vector<uint32_t>(4,0))),
          g.AddTensorConstant(Attribute(p.cropped_shape)),TosaTensor(DataType::float32_t,p.cropped_shape));
      bool dw=p.code==kLiteRtOpCodeTflDepthwiseConv2d;
      auto shape=p.weight_shape;
      if(dw)shape={shape[1],shape[2],shape[3],1};
      auto weights=Constant(g,p.float_weights,shape,constants);
      auto bias=Constant(g,p.float_bias,{p.output.shape[3]},constants);
      auto zero=g.AddTensorConstant(Attribute(int8_t{0}));
      value=g.AddOperator(dw?tfsc::tosa::OperatorEnum::DepthwiseConv2d:tfsc::tosa::OperatorEnum::Conv2d,
          {value,weights,bias,zero,zero},{ftype},{Attribute(p.padding),Attribute(p.stride),Attribute(p.dilation),Attribute(int32_t{3}),Attribute(false)})[0];
      if(p.float_activation)value=g.AddClampOperator(value,Attribute(0.0f),Attribute(p.float_activation==3?6.0f:std::numeric_limits<float>::infinity()),Attribute(int32_t{1}),ftype);
      return value;
    }
    default:return nullptr;
  }
}

ResId EmitOperation(Graph& g,const std::vector<ResId>& inputs,const OperationPlan& p,
                    std::vector<GraphConstant>& constants) {
  auto input=inputs[0]; const auto& q=p.inputs[0];
  TosaTensor out(p.output.floating ? DataType::float32_t : DataType::int8_t,p.output.shape);
  if (IsExtendedOperation(p.code, q.floating)) return EmitExtendedOperation(g, input, p, constants);
  if(q.floating || p.output.floating) return EmitFloatOperation(g, inputs, p, constants);
  switch(p.code) {
    case kLiteRtOpCodeTflReshape:return Reshape(g,input,DataType::int8_t,p.output.shape);
    case kLiteRtOpCodeTflSlice:return g.AddSliceOperator(input,g.AddTensorConstant(Attribute(p.kernel)),g.AddTensorConstant(Attribute(p.parameters)),out);
    case kLiteRtOpCodeTflConcatenation:return g.AddConcatOperator(inputs,Attribute(p.axis),out);
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflRelu6:
      if (SameQuant(q, p.output)) return Clamp(g, input, p);
      return g.AddTableOperator(input, Constant(g, p.table, {256}, constants), out);
    case kLiteRtOpCodeTflQuantize:
      if (SameQuant(q, p.output)) return input;
      return g.AddTableOperator(input, Constant(g, p.table, {256}, constants), out);
    case kLiteRtOpCodeTflLogistic:
    case kLiteRtOpCodeTflHardSwish:
      return g.AddTableOperator(input,Constant(g,p.table,{256},constants),out);
    case kLiteRtOpCodeTflMul: {
      auto lhs=Rescale(g,input,q.shape,DataType::int8_t,DataType::int32_t,q.zero_point,0,{*MakeScale32(1)},constants);
      auto rhs=Rescale(g,inputs[1],p.inputs[1].shape,DataType::int8_t,DataType::int32_t,p.inputs[1].zero_point,0,{*MakeScale32(1)},constants);
      auto shape=q.shape; shape.insert(shape.begin(),p.output.shape.size()-shape.size(),1);
      lhs=Reshape(g,lhs,DataType::int32_t,shape);
      shape=p.inputs[1].shape;shape.insert(shape.begin(),p.output.shape.size()-shape.size(),1);
      rhs=Reshape(g,rhs,DataType::int32_t,shape);
      auto value=g.AddMulOperator(lhs,rhs,g.AddTensorConstant(Attribute(int8_t{0})),TosaTensor(DataType::int32_t,p.output.shape));
      return Clamp(g,EmitRescale(g,value,p.output.shape,p.output.zero_point,p.scales,p.ranges,constants),p);
    }
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflDepthwiseConv2d: return EmitConv2d(g,input,*p.convolution,constants);
    case kLiteRtOpCodeTflTranspose: return g.AddTransposeOperator(input,Attribute(p.parameters),out);
    case kLiteRtOpCodeTflPadv2:
    case kLiteRtOpCodeTflPad: return g.AddPadOperator(input,g.AddTensorConstant(Attribute(p.parameters)),
        g.AddTensorConstant(Attribute(p.pad_value)),out);
    case kLiteRtOpCodeTflAdd: {
      auto lhs=Rescale(g,input,q.shape,DataType::int8_t,DataType::int32_t,q.zero_point,0,{p.scales[0]},constants);
      auto rhs=Rescale(g,inputs[1],p.inputs[1].shape,DataType::int8_t,DataType::int32_t,p.inputs[1].zero_point,0,{p.scales[1]},constants);
      auto shape=q.shape; shape.insert(shape.begin(),p.output.shape.size()-shape.size(),1);
      lhs=Reshape(g,lhs,DataType::int32_t,shape);
      shape=p.inputs[1].shape;shape.insert(shape.begin(),p.output.shape.size()-shape.size(),1);
      rhs=Reshape(g,rhs,DataType::int32_t,shape);
      auto sum=g.AddAddOperator(lhs,rhs,TosaTensor(DataType::int32_t,p.output.shape));
      return Clamp(g,EmitRescale(g,sum,p.output.shape,p.output.zero_point,{p.scales[2]},p.ranges,constants),p);
    }
    case kLiteRtOpCodeTflAveragePool2d: {
      if (p.cropped_shape != q.shape) {
        input = g.AddSliceOperator(input, g.AddTensorConstant(Attribute(std::vector<uint32_t>(4, 0))),
            g.AddTensorConstant(Attribute(p.cropped_shape)), TosaTensor(DataType::int8_t, p.cropped_shape));
      }
      // Native TOSA rounding is intentionally allowed to differ from LiteRT
      // at quantization boundaries; avoid expanding pooling into integer ops.
      auto result = g.AddAvgPool2dOperator(input,
          g.AddTensorConstant(Attribute(static_cast<int8_t>(q.zero_point))),
          g.AddTensorConstant(Attribute(static_cast<int8_t>(p.output.zero_point))),
          Attribute(p.kernel), Attribute(p.stride), Attribute({0, 0, 0, 0}),
          Attribute(int32_t{1}), out);
      return Clamp(g, result, p);
    }
    case kLiteRtOpCodeTflMaxPool2d: {
      if(p.cropped_shape!=q.shape) input=g.AddSliceOperator(input,
          g.AddTensorConstant(Attribute(std::vector<uint32_t>(4,0))),
          g.AddTensorConstant(Attribute(p.cropped_shape)),TosaTensor(DataType::int8_t,p.cropped_shape));
      auto result = g.AddMaxPool2dOperator(input, Attribute(p.kernel), Attribute(p.stride),
          Attribute({0, 0, 0, 0}), Attribute(int32_t{1}), out);
      return Clamp(g,result,p);
    }
    case kLiteRtOpCodeTflSum: {
      auto value=Rescale(g,input,q.shape,DataType::int8_t,DataType::int32_t,q.zero_point,0,{*MakeScale32(1)},constants);
      auto shape=q.shape;
      for(auto axis:p.parameters) { shape[axis]=1; value=g.AddReduceSumOperator(value,Attribute(axis),TosaTensor(DataType::int32_t,shape)); }
      return Rescale(g,value,p.output.shape,DataType::int32_t,DataType::int8_t,0,p.output.zero_point,p.scales,constants);
    }
    case kLiteRtOpCodeTflFullyConnected: {
      const auto& plan=*p.convolution; const auto& s=plan.spec;
      uint32_t batch=s.input_shape[0], k=s.weight_shape[3], channels=s.weight_shape[0];
      auto a=Reshape(g,input,DataType::int8_t,{1,batch,k});
      std::vector<int8_t> transposed(s.weights.size());
      for(uint32_t c=0;c<channels;++c) for(uint32_t i=0;i<k;++i) transposed[i*channels+c]=s.weights[c*k+i];
      auto b=Constant(g,transposed,{1,k,channels},constants);
      auto value=g.AddMatmulOperator(a,b,g.AddTensorConstant(Attribute(static_cast<int8_t>(q.zero_point))),
          g.AddTensorConstant(Attribute(int8_t{0})),TosaTensor(DataType::int32_t,{1,batch,channels}));
      auto bias=Constant(g,s.bias,{1,1,channels},constants);
      value=g.AddAddOperator(value,bias,TosaTensor(DataType::int32_t,{1,batch,channels}));
      value=EmitRescale(g,value,{1,batch,channels},p.output.zero_point,plan.scales,plan.accumulator_ranges,constants);
      return Clamp(g,Reshape(g,value,DataType::int8_t,p.output.shape),p);
    }
    default: return nullptr;
  }
}
}  // namespace litert::arm_vulkan_ml
