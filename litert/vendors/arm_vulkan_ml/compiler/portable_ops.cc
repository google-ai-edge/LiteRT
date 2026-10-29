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

#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"
#include "litert/core/model/model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include "litert/c/litert_op_options.h"
namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::ResId;
using TT = tfsc::tosa::Tensor;
using Shape = std::vector<uint32_t>;
bool ShapeOf(const Tensor& t, Shape& shape, bool quantized = false) {
  auto type = t.RankedTensorType();
  if (!type || type->ElementType() != (quantized ? ElementType::Int8 : ElementType::Float32) || type->Layout().HasStrides()) return false;
  uint64_t count = 1;
  for (int dim : type->Layout().Dimensions()) {
    if (dim <= 0 || dim > 8192 || count > uint64_t{INT32_MAX}/dim) return false;
    count *= dim; shape.push_back(dim);
  }
  if (shape.empty() && t.HasWeights()) shape.push_back(1);
  return !shape.empty() && shape.size() <= 6;
}
bool Ints(const Tensor& t, std::vector<int32_t>& values) {
  auto type = t.RankedTensorType();
  if (!type || type->ElementType() != ElementType::Int32 || type->Layout().HasStrides()) return false;
  auto data = t.WeightsData<int32_t>();
  if (!data) return false;
  uint64_t count = 1;
  for (int dim : type->Layout().Dimensions()) {
    if (dim <= 0 || count > uint64_t{INT32_MAX}/dim) return false;
    count *= dim;
  }
  if (count != data->size()) return false;
  values.assign(data->begin(), data->end()); return true;
}
uint64_t Count(const Shape& s) { return std::accumulate(s.begin(), s.end(), uint64_t{1}, std::multiplies<uint64_t>()); }
}
bool IsPortableOperation(LiteRtOpCode code) {
  switch (code) {
#define OP(name) case kLiteRtOpCodeTfl##name:
    OP(Relu) OP(Relu6) OP(Slice) OP(Concatenation) OP(Padv2)
    OP(Sub) OP(Div) OP(Maximum) OP(Minimum) OP(Abs) OP(Neg) OP(Square)
    OP(Pow) OP(Sin) OP(Cos) OP(Gelu) OP(ReverseV2) OP(Conv3d)
    OP(Sqrt) OP(Rsqrt) OP(Exp) OP(Log) OP(Floor) OP(Ceil) OP(Tanh)
    OP(Squeeze) OP(ExpandDims) OP(Pack) OP(Unpack) OP(Split) OP(SplitV) OP(Tile)
    OP(Mean) OP(ReduceProd) OP(Softmax) OP(BatchMatmul) OP(StridedSlice) OP(MaxPool2d) OP(AveragePool2d)
      return true;
#undef OP
    default: return false;
  }
}
std::shared_ptr<PortablePlan> AnalyzePortableOperation(const Op& op, std::string& reason) {
  auto reject = [&](const char* message) -> std::shared_ptr<PortablePlan> { reason = message; return nullptr; };
  auto p = std::make_shared<PortablePlan>(); p->code = op.Code();
  p->quantized = !op.Outputs().empty() && op.Outputs()[0].ElementType()==ElementType::Int8;
  if(p->quantized){
    switch(p->code){case kLiteRtOpCodeTflMaximum:case kLiteRtOpCodeTflMinimum:case kLiteRtOpCodeTflSqueeze:case kLiteRtOpCodeTflExpandDims:case kLiteRtOpCodeTflPack:case kLiteRtOpCodeTflUnpack:case kLiteRtOpCodeTflSplit:case kLiteRtOpCodeTflSplitV:case kLiteRtOpCodeTflTile:case kLiteRtOpCodeTflStridedSlice:case kLiteRtOpCodeTflReverseV2:break;default:return reject("Quantized portable operation not enabled");}
    const auto output=op.Outputs()[0];if(output.QTypeId()!=QuantizationTypeId::PerTensor)return reject("Quantized shape op requires per-tensor quantization");auto q=output.PerTensorQuantization();
    if(!std::isfinite(q.scale) || q.scale<=0 || q.zero_point < -128 || q.zero_point>127)return reject("Invalid shape-op quantization");
    for(const auto& t:op.Inputs())if(t.ElementType()==ElementType::Int8){if(t.QTypeId()!=QuantizationTypeId::PerTensor)return reject("Shape-op quantization differs");auto iq=t.PerTensorQuantization();if(iq.scale!=q.scale || iq.zero_point!=q.zero_point)return reject("Shape-op quantization differs");}
    for(const auto& t:op.Outputs()){if(t.ElementType()!=ElementType::Int8 || t.QTypeId()!=QuantizationTypeId::PerTensor)return reject("Shape-op output type differs");auto oq=t.PerTensorQuantization();if(oq.scale!=q.scale || oq.zero_point!=q.zero_point)return reject("Shape-op output quantization differs");}
  }
  auto shape_of=[&](const Tensor& t,Shape& s){return ShapeOf(t,s,p->quantized);};
  const auto in = op.Inputs(); const auto out = op.Outputs();
  const int first = p->code == kLiteRtOpCodeTflSplit ? 1 : 0;
  if (in.size() <= first || out.empty()) return reject("Missing data operands");
  Shape source;
  if (!shape_of(in[first], source)) return reject("Portable lowering requires supported static dense non-scalar data");
  p->inputs = {first}; p->shapes = {source};
  for (const auto& t : out) { Shape s; if (!shape_of(t, s)) return reject("Invalid FP32 output shape"); p->outputs.push_back(s); }
  const int rank = source.size();
  Shape expected = source;
  auto axis = [&](int limit) { if (p->axis < 0) p->axis += limit; return p->axis >= 0 && p->axis < limit; };
  auto ints = [&](int i, std::vector<int32_t>& v) { return i < in.size() && Ints(in[i],v); };
  const auto code = p->code;
  if (code == kLiteRtOpCodeTflConv3d) {
    Shape weight, bias;
    if (in.size()!=3 || rank!=5 || !shape_of(in[1],weight) || weight.size()!=5 ||
        !shape_of(in[2],bias) || bias.size()!=1 || !in[1].HasWeights() || !in[2].HasWeights() ||
        weight[3]!=source[4] || bias[0]!=weight[4]) return reject("CONV3D requires constant DHWIO weights and bias");
    for (int i : {1,2}) { auto data=in[i].WeightsData<float>(); if(!data || data->size()!=Count(i==1?weight:bias))return reject("Invalid CONV3D constants");for(float v:*data)if(!std::isfinite(v))return reject("Nonfinite CONV3D constants"); }
    uint32_t pad;
    p->strides.resize(3);p->sizes.resize(3);p->padding.resize(6);
    if(LiteRtGetConv3dPaddingOption(op.Get(),&pad)!=kLiteRtStatusOk || pad>1 ||
       LiteRtGetConv3dStrideDOption(op.Get(),&p->strides[0])!=kLiteRtStatusOk ||
       LiteRtGetConv3dStrideHOption(op.Get(),&p->strides[1])!=kLiteRtStatusOk ||
       LiteRtGetConv3dStrideWOption(op.Get(),&p->strides[2])!=kLiteRtStatusOk ||
       LiteRtGetConv3dDilationDOption(op.Get(),&p->sizes[0])!=kLiteRtStatusOk ||
       LiteRtGetConv3dDilationHOption(op.Get(),&p->sizes[1])!=kLiteRtStatusOk ||
       LiteRtGetConv3dDilationWOption(op.Get(),&p->sizes[2])!=kLiteRtStatusOk ||
       LiteRtGetConv3dFusedActivationOption(op.Get(),&p->activation)!=kLiteRtStatusOk ||
       (p->activation!=0 && p->activation!=1 && p->activation!=3))return reject("Invalid CONV3D options");
    expected[4]=weight[4];
    for(int a=0;a<3;++a){int64_t n=source[a+1],st=p->strides[a],d=p->sizes[a];if(st<=0 || st>8192 || d<=0 || d>8192)return reject("Invalid CONV3D stride/dilation");int64_t k=(weight[a]-1)*d+1;if(pad==1 && n<k)return reject("Invalid CONV3D geometry");int64_t count=pad==0?(n+st-1)/st:(n-k)/st+1;int64_t total=pad==0?std::max(int64_t{0},(count-1)*st+k-n):0;if(total>INT32_MAX)return reject("CONV3D padding overflow");p->padding[2*a]=total/2;p->padding[2*a+1]=total-total/2;expected[a+1]=static_cast<uint32_t>(count);}
    p->inputs={0,1,2};p->shapes={source,weight,bias};
  } else if(code==kLiteRtOpCodeTflReverseV2) {
    if(in.size()!=2 || !ints(1,p->parameters))return reject("REVERSE requires constant axes");
    std::set<int32_t> seen;
    for(auto& a:p->parameters){if(a<0)a+=rank;if(a<0 || a>=rank || !seen.insert(a).second)return reject("Invalid or duplicate REVERSE axis");}
  } else if(code==kLiteRtOpCodeTflGelu) {
    bool approximate;
    if(in.size()!=1 || LiteRtGetGeluApproximateOption(op.Get(),&approximate)!=kLiteRtStatusOk || approximate)return reject("GELU requires exact erf mode");
  } else if (code == kLiteRtOpCodeTflBatchMatmul) {
    Shape rhs;
    if(in.size()!=2 || !shape_of(in[1],rhs) || rank<2 || rhs.size()<2 ||
       LiteRtGetBatchMatmulAdjXOption(op.Get(),&p->adj_x)!=kLiteRtStatusOk ||
       LiteRtGetBatchMatmulAdjYOption(op.Get(),&p->adj_y)!=kLiteRtStatusOk)return reject("Invalid BATCH_MATMUL operands/options");
    auto lhs=source;
    if(p->adj_x)std::swap(lhs[lhs.size()-1],lhs[lhs.size()-2]);
    if(p->adj_y)std::swap(rhs[rhs.size()-1],rhs[rhs.size()-2]);
    size_t r=std::max(lhs.size(),rhs.size());lhs.insert(lhs.begin(),r-lhs.size(),1);rhs.insert(rhs.begin(),r-rhs.size(),1);
    if(lhs[r-1]!=rhs[r-2])return reject("BATCH_MATMUL contraction mismatch");
    expected.assign(r,1);
    for(size_t i=0;i+2<r;++i){if(lhs[i]!=rhs[i] && lhs[i]!=1 && rhs[i]!=1)return reject("Batch broadcast mismatch");expected[i]=std::max(lhs[i],rhs[i]);}
    expected[r-2]=lhs[r-2];expected[r-1]=rhs[r-1];
    if(Count(Shape(expected.begin(),expected.end()-2))>8192)return reject("Flattened batch exceeds limit");
    p->inputs.push_back(1);p->shapes.push_back({});if(!shape_of(in[1],p->shapes[1]))return reject("Invalid rhs");
    auto expanded=expected;expanded[r-1]=lhs[r-1];
    if(Count(expanded)>64*1024*1024)return reject("Batch materialization exceeds 256 MiB");
    expanded[r-2]=rhs[r-2];expanded[r-1]=rhs[r-1];
    if(Count(expanded)>64*1024*1024)return reject("Batch materialization exceeds 256 MiB");
  } else if(code==kLiteRtOpCodeTflMaxPool2d || code==kLiteRtOpCodeTflAveragePool2d) {
    bool avg=code==kLiteRtOpCodeTflAveragePool2d;uint32_t pad;
    auto gp=avg?LiteRtGetAveragePool2dPaddingOption:LiteRtGetMaxPool2dPaddingOption;
    auto gh=avg?LiteRtGetAveragePool2dFilterHeightOption:LiteRtGetMaxPool2dFilterHeightOption;
    auto gw=avg?LiteRtGetAveragePool2dFilterWidthOption:LiteRtGetMaxPool2dFilterWidthOption;
    auto sh=avg?LiteRtGetAveragePool2dStrideHOption:LiteRtGetMaxPool2dStrideHOption;
    auto sw=avg?LiteRtGetAveragePool2dStrideWOption:LiteRtGetMaxPool2dStrideWOption;
    auto ga=avg?LiteRtGetAveragePool2dFusedActivationOption:LiteRtGetMaxPool2dFusedActivationOption;
    p->kernel.resize(2);p->strides.resize(2);p->padding.resize(4);
    if(in.size()!=1 || rank!=4 || gp(op.Get(),&pad)!=kLiteRtStatusOk || pad>1 ||
       gh(op.Get(),&p->kernel[0])!=kLiteRtStatusOk || gw(op.Get(),&p->kernel[1])!=kLiteRtStatusOk ||
       sh(op.Get(),&p->strides[0])!=kLiteRtStatusOk || sw(op.Get(),&p->strides[1])!=kLiteRtStatusOk ||
       ga(op.Get(),&p->activation)!=kLiteRtStatusOk || (p->activation!=0 && p->activation!=1 && p->activation!=3))return reject("Invalid FP32 pooling options");
    for(int i=0;i<2;++i){int n=source[i+1],k=p->kernel[i],stride=p->strides[i];if(k<=0 || k>8192 || stride<=0 || stride>8192 || (pad==1 && n<k))return reject("Invalid pool geometry");int count=pad==0?(n+stride-1)/stride:(n-k)/stride+1;expected[i+1]=count;int total=pad==0?std::max(0,(count-1)*stride+k-n):0;p->padding[2*i]=total/2;p->padding[2*i+1]=total-total/2;}
  } else if(code==kLiteRtOpCodeTflStridedSlice) {
    int32_t begin_mask,end_mask,ellipsis,new_axis,shrink;bool offset;
    if(in.size()!=4 || !ints(1,p->parameters) || !ints(2,p->sizes) || !ints(3,p->strides) ||
       p->parameters.size()!=rank || p->sizes.size()!=rank || p->strides.size()!=rank ||
       LiteRtGetStridedSliceBeginMaskOption(op.Get(),&begin_mask)!=kLiteRtStatusOk ||
       LiteRtGetStridedSliceEndMaskOption(op.Get(),&end_mask)!=kLiteRtStatusOk ||
       LiteRtGetStridedSliceEllipsisMaskOption(op.Get(),&ellipsis)!=kLiteRtStatusOk ||
       LiteRtGetStridedSliceNewAxisMaskOption(op.Get(),&new_axis)!=kLiteRtStatusOk ||
       LiteRtGetStridedSliceShrinkAxisMaskOption(op.Get(),&shrink)!=kLiteRtStatusOk ||
       LiteRtGetStridedSliceOffsetOption(op.Get(),&offset)!=kLiteRtStatusOk || ellipsis || new_axis || offset)return reject("Unsupported STRIDED_SLICE masks/options");
    expected.clear();
    for(int i=0;i<rank;++i){int n=source[i],b=p->parameters[i],e=p->sizes[i],st=p->strides[i];if(st<=0 || st>8192)return reject("STRIDED_SLICE requires positive strides");if(b<0)b+=n;if(e<0)e+=n;b=(begin_mask&(1<<i))?0:std::clamp(b,0,n);e=(end_mask&(1<<i))?n:std::clamp(e,0,n);if(shrink&(1<<i)){if(b>=n)return reject("Shrink index out of range");e=b+1;st=1;}int size=(e-b+st-1)/st;if(size<=0)return reject("Empty slices unsupported");p->parameters[i]=b;p->sizes[i]=size;p->strides[i]=st;if(!(shrink&(1<<i)))expected.push_back(size);}
  } else if (code == kLiteRtOpCodeTflPow || code == kLiteRtOpCodeTflSub || code == kLiteRtOpCodeTflDiv || code == kLiteRtOpCodeTflMaximum || code == kLiteRtOpCodeTflMinimum) {
    Shape rhs;
    if (in.size()!=2 || !shape_of(in[1],rhs)) return reject("Binary op requires two FP32 operands");
    size_t r=std::max(source.size(),rhs.size()); source.insert(source.begin(),r-source.size(),1); rhs.insert(rhs.begin(),r-rhs.size(),1);
    expected.resize(r);
    for(size_t i=0;i<r;++i) { if(source[i]!=rhs[i] && source[i]!=1 && rhs[i]!=1) return reject("Broadcast mismatch"); expected[i]=std::max(source[i],rhs[i]); }
    p->inputs.push_back(1); p->shapes={source,rhs};
    if (code==kLiteRtOpCodeTflSub || code==kLiteRtOpCodeTflDiv) {
      auto get=code==kLiteRtOpCodeTflSub?LiteRtGetSubFusedActivationOption:LiteRtGetDivFusedActivationOption;
      if(get(op.Get(),&p->activation)!=kLiteRtStatusOk || (p->activation!=0 && p->activation!=1 && p->activation!=3)) return reject("Unsupported fused activation");
    }
  } else if(code==kLiteRtOpCodeTflSlice) {
    if(in.size()!=3 || !ints(1,p->parameters) || !ints(2,p->sizes) || p->parameters.size()!=rank || p->sizes.size()!=rank) return reject("SLICE requires constant rank-sized bounds");
    for(int i=0;i<rank;++i) { if(p->sizes[i]==-1) p->sizes[i]=source[i]-p->parameters[i]; if(p->parameters[i]<0 || p->sizes[i]<=0 || uint64_t{static_cast<uint32_t>(p->parameters[i])}+p->sizes[i]>source[i]) return reject("Invalid SLICE bounds"); expected[i]=p->sizes[i]; }
  } else if(code==kLiteRtOpCodeTflSqueeze || code==kLiteRtOpCodeTflExpandDims) {
    if(code==kLiteRtOpCodeTflExpandDims) {
      if(in.size()!=2 || !ints(1,p->parameters) || p->parameters.size()!=1) return reject("EXPAND_DIMS requires constant axis");
      p->axis=p->parameters[0]; if(!axis(rank+1)) return reject("Invalid axis"); expected.insert(expected.begin()+p->axis,1);
    } else {
      const int32_t* dims=nullptr; int n=0;
      if(in.size()!=1 || LiteRtGetSqueezeDimsOption(op.Get(),&dims,&n)!=kLiteRtStatusOk) return reject("Invalid SQUEEZE options");
      std::set<int> axes;
      for(int i=0;i<n;++i) { int a=dims[i]<0?dims[i]+rank:dims[i]; if(a<0 || a>=rank || source[a]!=1) return reject("Invalid squeeze axis"); axes.insert(a); }
      expected.clear(); for(int i=0;i<rank;++i) if(!(n==0?source[i]==1:axes.count(i))) expected.push_back(source[i]);
    }
  } else if(code==kLiteRtOpCodeTflTile) {
    if(in.size()!=2 || !ints(1,p->parameters) || p->parameters.size()!=rank) return reject("TILE requires constant multiples");
    for(int i=0;i<rank;++i) { if(p->parameters[i]<=0 || uint64_t{source[i]}*p->parameters[i]>8192) return reject("Invalid TILE extent"); expected[i]*=p->parameters[i]; }
  } else if(code==kLiteRtOpCodeTflPack || code==kLiteRtOpCodeTflConcatenation) {
    auto get=code==kLiteRtOpCodeTflPack?LiteRtGetPackAxisOption:LiteRtGetConcatenationAxisOption;
    if(get(op.Get(),&p->axis)!=kLiteRtStatusOk || !axis(rank+(code==kLiteRtOpCodeTflPack))) return reject("Invalid axis");
    p->inputs.clear();p->shapes.clear();
    if(code==kLiteRtOpCodeTflPack) expected.insert(expected.begin()+p->axis,in.size()); else expected[p->axis]=0;
    for(size_t i=0;i<in.size();++i) { Shape s; if(!shape_of(in[i],s) || s.size()!=source.size()) return reject("Invalid concatenated shape"); for(int j=0;j<rank;++j) if((code==kLiteRtOpCodeTflPack || j!=p->axis) && s[j]!=source[j]) return reject("Concatenated shape mismatch"); if(code!=kLiteRtOpCodeTflPack) expected[p->axis]+=s[p->axis]; p->inputs.push_back(i);p->shapes.push_back(s); }
    if(code==kLiteRtOpCodeTflConcatenation && (LiteRtGetConcatenationFusedActivationOption(op.Get(),&p->activation)!=kLiteRtStatusOk || p->activation!=0)) return reject("CONCAT activation unsupported");
  } else if(code==kLiteRtOpCodeTflUnpack || code==kLiteRtOpCodeTflSplit || code==kLiteRtOpCodeTflSplitV) {
    if(code==kLiteRtOpCodeTflUnpack) { if(LiteRtGetUnpackAxisOption(op.Get(),&p->axis)!=kLiteRtStatusOk) return reject("Invalid UNPACK axis"); }
    else { std::vector<int32_t> a; if(!ints(code==kLiteRtOpCodeTflSplit?0:2,a) || a.size()!=1) return reject("SPLIT requires constant axis");p->axis=a[0]; }
    if(!axis(rank)) return reject("Invalid split axis");
    if(code==kLiteRtOpCodeTflSplitV) {
      if(!ints(1,p->sizes) || p->sizes.size()!=out.size()) return reject("SPLIT_V sizes mismatch");
      int missing=-1;int64_t total=0;
      for(size_t i=0;i<p->sizes.size();++i) { if(p->sizes[i]==-1) { if(missing>=0) return reject("Multiple inferred split sizes");missing=i; } else { if(p->sizes[i]<=0)return reject("Invalid split size"); total+=p->sizes[i]; } }
      if(missing>=0){if(total>=source[p->axis])return reject("Invalid inferred split extent");p->sizes[missing]=source[p->axis]-total;}
    } else { int size=code==kLiteRtOpCodeTflUnpack?1:source[p->axis]/out.size(); p->sizes.assign(out.size(),size); }
    int64_t total=0;
    for(size_t i=0;i<out.size();++i) { if(p->sizes[i]<=0)return reject("Invalid split size");total+=p->sizes[i];Shape s=source;s[p->axis]=p->sizes[i];if(code==kLiteRtOpCodeTflUnpack)s.erase(s.begin()+p->axis);if(s!=p->outputs[i])return reject("Split output mismatch"); }
    if(total!=source[p->axis])return reject("Split extents mismatch");reason.clear();return p;
  } else if(code==kLiteRtOpCodeTflMean || code==kLiteRtOpCodeTflReduceProd) {
    bool keep=false;
    const auto* opts=internal::GetTflOptions(*op.Get()).AsReducerOptions();
    if(!opts)return reject("Missing reducer options");keep=opts->keep_dims;
    if(in.size()!=2 || !ints(1,p->parameters) || p->parameters.empty())return reject("Reduction requires constant axes");
    std::set<int32_t> axes;for(auto a:p->parameters){if(a<0)a+=rank;if(a<0 || a>=rank)return reject("Invalid reduction axis");axes.insert(a);}p->parameters.assign(axes.begin(),axes.end());
    expected.clear();p->scalar=1;for(int i=0;i<rank;++i) { if(axes.count(i)){p->scalar*=source[i];if(keep)expected.push_back(1);}else expected.push_back(source[i]); }
  } else if(code==kLiteRtOpCodeTflSoftmax) {
    if(in.size()!=1 || LiteRtGetSoftmaxBetaOption(op.Get(),&p->scalar)!=kLiteRtStatusOk || !std::isfinite(p->scalar) || p->scalar<=0)return reject("SOFTMAX requires finite positive beta");
  } else if(code==kLiteRtOpCodeTflPadv2) {
    if(in.size()!=3 || !ints(1,p->parameters) || p->parameters.size()!=2*rank)return reject("PADV2 requires constant padding");
    auto fill=in[2].WeightsData<float>();if(!fill || fill->size()!=1 || !std::isfinite((*fill)[0]))return reject("PADV2 requires finite constant FP32 fill");p->scalar=(*fill)[0];
    for(int i=0;i<rank;++i){if(p->parameters[2*i]<0 || p->parameters[2*i+1]<0)return reject("Negative padding");uint64_t dim=uint64_t{source[i]}+p->parameters[2*i]+p->parameters[2*i+1];if(dim>8192)return reject("PADV2 extent exceeds limit");expected[i]=dim;}
  } else if(in.size()!=1)return reject("Unary op requires one operand");
  if(out.size()!=1 || expected!=p->outputs[0] || expected.empty())return reject("FP32 output shape mismatch or scalar output");
  reason.clear();return p;
}
std::vector<ResId> EmitPortableOperation(tfsc::tosa::Graph& g, std::vector<ResId> v, const PortablePlan& p) {
  auto type=[&](const Shape& s){return TT(p.quantized?DataType::int8_t:DataType::float32_t,s);};
  auto reshape=[&](ResId x,const Shape& s){return g.AddReshapeOperator(x,g.AddTensorConstant(Attribute(s)),type(s));};
  auto constant=[&](float x,const Shape& s){return reshape(g.AddTensorConstant(Attribute(x)),Shape(s.size(),1));};
  const Shape& out=p.outputs[0];const auto t=type(out);auto x=v[0];
  auto clamp=[&](ResId z,float lo,float hi){return g.AddClampOperator(z,Attribute(lo),Attribute(hi),Attribute(uint32_t{0}),t);};
  auto binary=[&](tfsc::tosa::OperatorEnum op){for(size_t i=0;i<v.size();++i)v[i]=reshape(v[i],p.shapes[i]);return g.AddOperator(op,v,{t})[0];};
  using O=tfsc::tosa::OperatorEnum;
  switch(p.code) {
    case kLiteRtOpCodeTflConv3d: {
      // LiteRT DHWIO -> TOSA ODHWI.
      auto w=p.shapes[1];Shape tw={w[4],w[0],w[1],w[2],w[3]};
      auto weight=g.AddTransposeOperator(v[1],Attribute(std::vector<int32_t>{4,0,1,2,3}),type(tw));
      auto cropped=p.shapes[0];
      for(size_t a=0;a<3;++a){uint64_t k=(static_cast<uint64_t>(w[a])-1)*p.sizes[a]+1;cropped[a+1]-=(cropped[a+1]+p.padding[2*a]+p.padding[2*a+1]-k)%p.strides[a];}
      if(cropped!=p.shapes[0])x=g.AddSliceOperator(x,g.AddTensorConstant(Attribute(std::vector<int32_t>(5,0))),g.AddTensorConstant(Attribute(cropped)),type(cropped));
      auto zero=g.AddTensorConstant(Attribute(0.0f));
      x=g.AddConv3dOperator(x,weight,v[2],zero,zero,Attribute(p.padding),Attribute(p.strides),Attribute(p.sizes),Attribute(int32_t{3}),Attribute(false),t);
      break;
    }
    case kLiteRtOpCodeTflReverseV2:
      for(auto a:p.parameters)x=g.AddReverseOperator(x,Attribute(a),t);
      return {x};
    case kLiteRtOpCodeTflGelu: {
      auto zero=g.AddTensorConstant(Attribute(int8_t{0}));
      auto z=g.AddMulOperator(x,constant(0.7071067811865475f,out),zero,t);
      z=g.AddErfOperator(z,t);z=g.AddAddOperator(z,constant(1.0f,out),t);
      z=g.AddMulOperator(z,constant(0.5f,out),zero,t);
      return {g.AddMulOperator(x,z,zero,t)};
    }
    case kLiteRtOpCodeTflPow:x=binary(O::Pow);break;
    case kLiteRtOpCodeTflBatchMatmul: {
      size_t rank=out.size();auto batch=out;batch.resize(rank-2);uint32_t n=Count(batch);
      for(size_t i=0;i<2;++i){auto shape=p.shapes[i];bool adj=i==0?p.adj_x:p.adj_y;if(adj){std::vector<uint32_t> perm(shape.size());std::iota(perm.begin(),perm.end(),0);std::swap(perm[perm.size()-1],perm[perm.size()-2]);std::swap(shape[shape.size()-1],shape[shape.size()-2]);v[i]=g.AddTransposeOperator(v[i],Attribute(perm),type(shape));}shape.insert(shape.begin(),rank-shape.size(),1);v[i]=reshape(v[i],shape);Shape mult(rank,1),full=shape;for(size_t j=0;j+2<rank;++j){mult[j]=out[j]/shape[j];full[j]=out[j];}if(full!=shape)v[i]=g.AddTileOperator(v[i],g.AddTensorConstant(Attribute(mult)),type(full));v[i]=reshape(v[i],{n,full[rank-2],full[rank-1]});}
      auto zero=g.AddTensorConstant(Attribute(0.0f));
      return {reshape(g.AddMatmulOperator(v[0],v[1],zero,zero,type({n,out[rank-2],out.back()})),out)};
    }
    case kLiteRtOpCodeTflMaxPool2d:case kLiteRtOpCodeTflAveragePool2d: {
      auto cropped=p.shapes[0];
      for(int a=0;a<2;++a)cropped[a+1]-=(cropped[a+1]+p.padding[2*a]+p.padding[2*a+1]-p.kernel[a])%p.strides[a];
      if(cropped!=p.shapes[0])x=g.AddSliceOperator(x,g.AddTensorConstant(Attribute(std::vector<int32_t>(4,0))),g.AddTensorConstant(Attribute(cropped)),type(cropped));
      if(p.code==kLiteRtOpCodeTflMaxPool2d)x=g.AddMaxPool2dOperator(x,Attribute(p.kernel),Attribute(p.strides),Attribute(p.padding),Attribute(uint32_t{0}),t);
      else {auto zero=g.AddTensorConstant(Attribute(0.0f));x=g.AddAvgPool2dOperator(x,zero,zero,Attribute(p.kernel),Attribute(p.strides),Attribute(p.padding),Attribute(int32_t{3}),t);}
      break;
    }
    case kLiteRtOpCodeTflStridedSlice: {
      auto shape=p.shapes[0];
      for(size_t a=0;a<shape.size();++a){std::vector<ResId> slices;auto part=shape;part[a]=1;for(int j=0;j<p.sizes[a];++j){std::vector<int32_t> begin(shape.size(),0);begin[a]=p.parameters[a]+j*p.strides[a];slices.push_back(g.AddSliceOperator(x,g.AddTensorConstant(Attribute(begin)),g.AddTensorConstant(Attribute(part)),type(part)));}shape[a]=p.sizes[a];x=slices.size()==1?slices[0]:g.AddConcatOperator(slices,Attribute(static_cast<int32_t>(a)),type(shape));}
      return {reshape(x,out)};
    }
    case kLiteRtOpCodeTflRelu:return {clamp(x,0,std::numeric_limits<float>::infinity())};
    case kLiteRtOpCodeTflRelu6:return {clamp(x,0,6)};
    case kLiteRtOpCodeTflSub:x=binary(O::Sub);break;
    case kLiteRtOpCodeTflDiv:v[1]=reshape(v[1],p.shapes[1]);v[1]=g.AddReciprocalOperator(v[1],type(p.shapes[1]));v[0]=reshape(v[0],p.shapes[0]);x=g.AddMulOperator(v[0],v[1],g.AddTensorConstant(Attribute(int8_t{0})),t);break;
    case kLiteRtOpCodeTflMaximum:case kLiteRtOpCodeTflMinimum:
      for(size_t i=0;i<v.size();++i)v[i]=reshape(v[i],p.shapes[i]);
      if(p.quantized){for(size_t i=0;i<v.size();++i)v[i]=g.AddCastOperator(v[i],TT(DataType::int32_t,p.shapes[i]));auto z=g.AddOperator(p.code==kLiteRtOpCodeTflMaximum?O::Maximum:O::Minimum,v,{TT(DataType::int32_t,out)},{Attribute(uint32_t{1})})[0];return {g.AddCastOperator(z,t)};}
      return {g.AddOperator(p.code==kLiteRtOpCodeTflMaximum?O::Maximum:O::Minimum,v,{t},{Attribute(uint32_t{0})})[0]};
#define UNARY(name, helper) case kLiteRtOpCodeTfl##name:return {g.Add##helper##Operator(x,t)};
    UNARY(Sin,Sin) UNARY(Cos,Cos) UNARY(Abs,Abs)  UNARY(Rsqrt,Rsqrt) UNARY(Exp,Exp) UNARY(Log,Log)
    UNARY(Floor,Floor) UNARY(Ceil,Ceil) UNARY(Tanh,Tanh)
#undef UNARY
    case kLiteRtOpCodeTflNeg:return {g.AddNegateOperator(x,g.AddTensorConstant(Attribute(0.0f)),g.AddTensorConstant(Attribute(0.0f)),t)};
    case kLiteRtOpCodeTflSqrt:return {g.AddReciprocalOperator(g.AddRsqrtOperator(x,t),t)};
    case kLiteRtOpCodeTflSquare:return {g.AddMulOperator(x,x,g.AddTensorConstant(Attribute(int8_t{0})),t)};
    case kLiteRtOpCodeTflSqueeze:case kLiteRtOpCodeTflExpandDims:return {reshape(x,out)};
    case kLiteRtOpCodeTflTile:return {g.AddTileOperator(x,g.AddTensorConstant(Attribute(p.parameters)),t)};
    case kLiteRtOpCodeTflSlice:return {g.AddSliceOperator(x,g.AddTensorConstant(Attribute(p.parameters)),g.AddTensorConstant(Attribute(p.sizes)),t)};
    case kLiteRtOpCodeTflPadv2:return {g.AddPadOperator(x,g.AddTensorConstant(Attribute(p.parameters)),g.AddTensorConstant(Attribute(p.scalar)),t)};
    case kLiteRtOpCodeTflPack:case kLiteRtOpCodeTflConcatenation:
      if(p.code==kLiteRtOpCodeTflPack)for(size_t i=0;i<v.size();++i){auto s=p.shapes[i];s.insert(s.begin()+p.axis,1);v[i]=reshape(v[i],s);}
      return {g.AddConcatOperator(v,Attribute(p.axis),t)};
    case kLiteRtOpCodeTflUnpack:case kLiteRtOpCodeTflSplit:case kLiteRtOpCodeTflSplitV: {
      std::vector<ResId> result;int start=0;
      for(size_t i=0;i<p.outputs.size();++i){auto s=p.shapes[0];s[p.axis]=p.sizes[i];std::vector<int32_t> begin(s.size(),0);begin[p.axis]=start;start+=p.sizes[i];auto z=g.AddSliceOperator(x,g.AddTensorConstant(Attribute(begin)),g.AddTensorConstant(Attribute(s)),type(s));result.push_back(reshape(z,p.outputs[i]));}return result;
    }
    case kLiteRtOpCodeTflMean:case kLiteRtOpCodeTflReduceProd: {
      auto s=p.shapes[0];for(auto a:p.parameters){s[a]=1;x=p.code==kLiteRtOpCodeTflMean?g.AddReduceSumOperator(x,Attribute(a),type(s)):g.AddReduceProductOperator(x,Attribute(a),type(s));}
      x=reshape(x,out);if(p.code==kLiteRtOpCodeTflMean)x=g.AddMulOperator(x,constant(1/p.scalar,out),g.AddTensorConstant(Attribute(int8_t{0})),t);return {x};
    }
    case kLiteRtOpCodeTflSoftmax: {
      int a=out.size()-1;auto s=out;s[a]=1;
      if(p.scalar!=1)x=g.AddMulOperator(x,constant(p.scalar,out),g.AddTensorConstant(Attribute(int8_t{0})),t);
      auto max=g.AddReduceMaxOperator(x,Attribute(a),Attribute(uint32_t{0}),type(s));
      x=g.AddExpOperator(g.AddSubOperator(x,max,t),t);
      auto sum=g.AddReduceSumOperator(x,Attribute(a),type(s));
      return {g.AddMulOperator(x,g.AddReciprocalOperator(sum,type(s)),g.AddTensorConstant(Attribute(int8_t{0})),t)};
    }
    default:return {};
  }
  if(p.activation==1)x=clamp(x,0,std::numeric_limits<float>::infinity());
  if(p.activation==3)x=clamp(x,0,6);return {x};
}
}
