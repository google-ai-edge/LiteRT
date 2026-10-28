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
bool ShapeOf(const Tensor& t, Shape& shape) {
  auto type = t.RankedTensorType();
  if (!type || type->ElementType() != ElementType::Float32 || type->Layout().HasStrides()) return false;
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
  const auto in = op.Inputs(); const auto out = op.Outputs();
  const int first = p->code == kLiteRtOpCodeTflSplit ? 1 : 0;
  if (in.size() <= first || out.empty()) return reject("Missing data operands");
  Shape source;
  if (!ShapeOf(in[first], source)) return reject("Portable lowering requires static dense non-scalar FP32");
  p->inputs = {first}; p->shapes = {source};
  for (const auto& t : out) { Shape s; if (!ShapeOf(t, s)) return reject("Invalid FP32 output shape"); p->outputs.push_back(s); }
  const int rank = source.size();
  Shape expected = source;
  auto axis = [&](int limit) { if (p->axis < 0) p->axis += limit; return p->axis >= 0 && p->axis < limit; };
  auto ints = [&](int i, std::vector<int32_t>& v) { return i < in.size() && Ints(in[i],v); };
  const auto code = p->code;
  if (code == kLiteRtOpCodeTflBatchMatmul) {
    Shape rhs;
    if(in.size()!=2 || !ShapeOf(in[1],rhs) || rank<2 || rhs.size()<2 ||
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
    p->inputs.push_back(1);p->shapes.push_back({});if(!ShapeOf(in[1],p->shapes[1]))return reject("Invalid rhs");
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
  } else if (code == kLiteRtOpCodeTflSub || code == kLiteRtOpCodeTflDiv || code == kLiteRtOpCodeTflMaximum || code == kLiteRtOpCodeTflMinimum) {
    Shape rhs;
    if (in.size()!=2 || !ShapeOf(in[1],rhs)) return reject("Binary op requires two FP32 operands");
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
    for(size_t i=0;i<in.size();++i) { Shape s; if(!ShapeOf(in[i],s) || s.size()!=source.size()) return reject("Invalid concatenated shape"); for(int j=0;j<rank;++j) if((code==kLiteRtOpCodeTflPack || j!=p->axis) && s[j]!=source[j]) return reject("Concatenated shape mismatch"); if(code!=kLiteRtOpCodeTflPack) expected[p->axis]+=s[p->axis]; p->inputs.push_back(i);p->shapes.push_back(s); }
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
  auto type=[](const Shape& s){return TT(DataType::float32_t,s);};
  auto reshape=[&](ResId x,const Shape& s){return g.AddReshapeOperator(x,g.AddTensorConstant(Attribute(s)),type(s));};
  auto constant=[&](float x,const Shape& s){return reshape(g.AddTensorConstant(Attribute(x)),Shape(s.size(),1));};
  const Shape& out=p.outputs[0];const auto t=type(out);auto x=v[0];
  auto clamp=[&](ResId z,float lo,float hi){return g.AddClampOperator(z,Attribute(lo),Attribute(hi),Attribute(uint32_t{0}),t);};
  auto binary=[&](tfsc::tosa::OperatorEnum op){for(size_t i=0;i<v.size();++i)v[i]=reshape(v[i],p.shapes[i]);return g.AddOperator(op,v,{t})[0];};
  using O=tfsc::tosa::OperatorEnum;
  switch(p.code) {
    case kLiteRtOpCodeTflBatchMatmul: {
      size_t rank=out.size();auto batch=out;batch.resize(rank-2);uint32_t n=Count(batch);
      for(size_t i=0;i<2;++i){auto shape=p.shapes[i];bool adj=i==0?p.adj_x:p.adj_y;if(adj){std::vector<uint32_t> perm(shape.size());std::iota(perm.begin(),perm.end(),0);std::swap(perm[perm.size()-1],perm[perm.size()-2]);std::swap(shape[shape.size()-1],shape[shape.size()-2]);v[i]=g.AddTransposeOperator(v[i],Attribute(perm),type(shape));}shape.insert(shape.begin(),rank-shape.size(),1);v[i]=reshape(v[i],shape);Shape mult(rank,1),full=shape;for(size_t j=0;j+2<rank;++j){mult[j]=out[j]/shape[j];full[j]=out[j];}if(full!=shape)v[i]=g.AddTileOperator(v[i],g.AddTensorConstant(Attribute(mult)),type(full));v[i]=reshape(v[i],{n,full[rank-2],full[rank-1]});}
      auto zero=g.AddTensorConstant(Attribute(int8_t{0}));
      return {reshape(g.AddMatmulOperator(v[0],v[1],zero,zero,type({n,out[rank-2],out.back()})),out)};
    }
    case kLiteRtOpCodeTflMaxPool2d:case kLiteRtOpCodeTflAveragePool2d: {
      auto cropped=p.shapes[0];
      for(int a=0;a<2;++a)cropped[a+1]-=(cropped[a+1]+p.padding[2*a]+p.padding[2*a+1]-p.kernel[a])%p.strides[a];
      if(cropped!=p.shapes[0])x=g.AddSliceOperator(x,g.AddTensorConstant(Attribute(std::vector<int32_t>(4,0))),g.AddTensorConstant(Attribute(cropped)),type(cropped));
      if(p.code==kLiteRtOpCodeTflMaxPool2d)x=g.AddMaxPool2dOperator(x,Attribute(p.kernel),Attribute(p.strides),Attribute(p.padding),Attribute(uint32_t{0}),t);
      else {auto zero=g.AddTensorConstant(Attribute(int8_t{0}));x=g.AddAvgPool2dOperator(x,zero,zero,Attribute(p.kernel),Attribute(p.strides),Attribute(p.padding),Attribute(int32_t{3}),t);}
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
      return {g.AddOperator(p.code==kLiteRtOpCodeTflMaximum?O::Maximum:O::Minimum,v,{t},{Attribute(uint32_t{0})})[0]};
#define UNARY(name, helper) case kLiteRtOpCodeTfl##name:return {g.Add##helper##Operator(x,t)};
    UNARY(Abs,Abs)  UNARY(Rsqrt,Rsqrt) UNARY(Exp,Exp) UNARY(Log,Log)
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
