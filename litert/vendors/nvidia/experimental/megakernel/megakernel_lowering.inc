// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Experimental. Members of TensorRtGraphBuilder
// (compiler/tensorrt_graph_builder.cc), which includes this file inside the
// class: the lowering reads the builder's view of the graph (its in-place
// caches, its MoE block plans, its weight store and holders), and the
// builder runs none of it unless BuildMegakernel() is the entry, which only
// BuildMegakernelEngine (megakernel_lowering.h) calls.

public:
// Build() with the subgraph's whole body as one megakernel program.
Expected<TensorRtBuildResult> BuildMegakernel(
    const Subgraph& subgraph,
    absl::Span<const std::string> read_only_value_cache_inputs,
    TensorRtWeightStoreBuilder* weight_store,
    absl::Span<const std::string> shared_cache_inputs) {
  megakernel_body_ = true;
  return Build(subgraph, read_only_value_cache_inputs, weight_store,
               shared_cache_inputs);
}

private:
// Set by BuildMegakernel: Build() then calls LowerMegakernel in place of
// its op-by-op lowering.
bool megakernel_body_ = false;

// The pieces of the arena of a moe op's plugin (MoeExpertSource) that are
// its experts' weights and scales, where they are in the model file. These
// three are the lowering of the moe op's own (LowerCustomOp), repeated here
// so that the megakernel reads the arenas that op's plugin loads without
// that lowering being touched.
Expected<MoeExpertSource> MegakernelExpertPieces(const Op& op,
                                                 const MoeCustomOp& moe) const {
  MoeExpertSource source;
  source.model_path = weight_store_->source_path();
  source.model_size = weight_store_->source_size();
  source.scale_type = moe.scale_type;
  const int piece_inputs[6] = {3, 5, 7, 4, 6, 8};
  for (int i = 0; i < 6; ++i) {
    const auto bytes = op.Inputs()[piece_inputs[i]].Weights().Bytes();
    LITERT_ASSIGN_OR_RETURN(source.offsets[i],
                            weight_store_->SourceOffset(bytes));
    source.sizes[i] = bytes.size();
  }
  return source;
}

// Adds the router's rows and scales of `routing` to the arena `source`;
// false, and nothing added, if they are not views of the model file.
bool AddMegakernelRouterPieces(const MoeRouting& routing,
                               MoeExpertSource& source) const {
  const Tensor& router = routing.router_weights;
  const Tensor router_scales(router.Context(),
                             router.BlockWiseQuantization().scales);
  const auto router_bytes = router.Weights().Bytes();
  const auto router_scale_bytes = router_scales.Weights().Bytes();
  auto offset = weight_store_->SourceOffset(router_bytes);
  auto scale_offset = weight_store_->SourceOffset(router_scale_bytes);
  if (!offset || !scale_offset) return false;
  source.offsets[6] = *offset;
  source.sizes[6] = router_bytes.size();
  source.offsets[7] = *scale_offset;
  source.sizes[7] = router_scale_bytes.size();
  source.router_scale_type =
      router_scales.ElementType() == litert::ElementType::Float32 ? 0 : 1;
  return true;
}

// Adds the dense feed-forward of `block` to the arena `source`; false, and
// nothing added, if its weights are not views of the model file.
bool AddMegakernelDensePieces(const MoeBlockFusion& block,
                              MoeExpertSource& source) const {
  const Tensor* dense_weights[3] = {&block.dense_gate_w, &block.dense_up_w,
                                    &block.dense_down_w};
  const int weight_pieces[3] = {8, 9, 12};
  const int scale_pieces[3] = {10, 11, 13};
  MoeExpertSource added = source;
  int scale_type = -1;
  for (int i = 0; i < 3; ++i) {
    const Tensor scales(dense_weights[i]->Context(),
                        dense_weights[i]->BlockWiseQuantization().scales);
    const auto bytes = dense_weights[i]->Weights().Bytes();
    const auto scale_bytes = scales.Weights().Bytes();
    const auto offset = weight_store_->SourceOffset(bytes);
    const auto scale_offset = weight_store_->SourceOffset(scale_bytes);
    const int type =
        scales.ElementType() == litert::ElementType::Float32 ? 0 : 1;
    if (!offset || !scale_offset || (scale_type >= 0 && type != scale_type)) {
      return false;
    }
    scale_type = type;
    // The halves of a stacked projection are the gate and the up piece.
    const bool half = block.dense_stacked && i < 2;
    const size_t weight_size = half ? bytes.size() / 2 : bytes.size();
    const size_t scale_size =
        half ? scale_bytes.size() / 2 : scale_bytes.size();
    const size_t skipped = half && i == 1 ? 1 : 0;
    added.offsets[weight_pieces[i]] = *offset + skipped * weight_size;
    added.sizes[weight_pieces[i]] = weight_size;
    added.offsets[scale_pieces[i]] = *scale_offset + skipped * scale_size;
    added.sizes[scale_pieces[i]] = scale_size;
  }
  added.dense_scale_type = scale_type;
  added.dense_hidden = block.dense_hidden;
  source = std::move(added);
  return true;
}

// ===========================================================================
// The whole body of a decode signature as one plugin layer that runs it as
// a program (megakernel_plugin.h): the Gemma 4 mixture-of-experts decode
// step, recognized layer by layer from what the builder's lowering knows
// of it.
//   attention  each layer's odml.cache_update (DecodeCacheUpdateMode) and
//              the odml.sdpa_transposed that reads its caches; in front of
//              them the qkv projection behind the layer's input norm, and
//              odml.qkv_norm_rope (sliding layers) or the head norms and
//              odml.rope (global layers, whose values are their keys
//              normalized without a scale); behind them the o projection,
//              its norm and the residual update
//   block      the layer's feed-forward block as FindMoeBlocks planned it
//              for the MoE plugin: the router, the dense feed-forward, the
//              experts, their norms, the residual update and its scalar
//   tail       the final norm (the activations output), the head and its
//              soft cap (the logits output)
// What else the body computes must only feed these ops (the routing chain
// the MoE plugin replaces, the cache parameters and positions the program
// computes itself). The weights stay in the weight store, the experts'
// arenas are those the MoE plugin loads, and the engine's cache outputs
// alias their inputs through native in-place writes of the rows the
// plugin wrote.
struct MegakernelBuild {
  MegakernelDescription description;
  std::vector<float> norms;
  // The weight store's segments the projections are in, by holder.
  std::vector<uint32_t> segments;
  // The cache inputs in their order among the plugin's inputs.
  std::vector<LiteRtTensor> caches;
  // The mask inputs, by MegakernelLayerSource::mask.
  std::vector<LiteRtTensor> masks;
  std::unordered_map<LiteRtTensor, int32_t> other_inputs;
  Tensor positions;
  std::unordered_set<LiteRtOp> covered;
  // The moe ops by the residual stream their plan routes.
  std::unordered_map<LiteRtTensor, LiteRtOp> moe_by_residual;
};

static Error MegakernelUnsupported(const std::string& why) {
  return Error(kLiteRtStatusErrorUnsupported, "megakernel: " + why);
}

// Appends the FP32 constant `tensor` of `count` values to the norms.
static Expected<int64_t> AddMegakernelVector(MegakernelBuild& build,
                                             const Tensor& tensor,
                                             int64_t count) {
  if (!tensor.HasWeights() || !IsUnquantizedFloat32(tensor) ||
      tensor.Weights().Bytes().size() !=
          static_cast<size_t>(count) * sizeof(float)) {
    return MegakernelUnsupported("a norm or scale is not an FP32 constant of " +
                                 std::to_string(count) + " values (tensor " +
                                 std::to_string(tensor.TensorIndex()) + ")");
  }
  const int64_t offset = static_cast<int64_t>(build.norms.size());
  build.norms.resize(offset + count);
  std::memcpy(build.norms.data() + offset, tensor.Weights().Bytes().data(),
              static_cast<size_t>(count) * sizeof(float));
  return offset;
}

// An odml.rms_norm of `dim` values whose epsilon is the model's: its scale
// among the norms.
Expected<int64_t> AddMegakernelNorm(MegakernelBuild& build, const Op& norm,
                                    int64_t dim) {
  const auto weights = RmsNormWeights(norm, static_cast<int>(dim));
  if (!weights.has_value()) {
    return MegakernelUnsupported("an rms_norm without an FP32 scale of " +
                                 std::to_string(dim));
  }
  if (weights->second != build.description.norm_eps) {
    return MegakernelUnsupported("rms_norms with different epsilons");
  }
  build.covered.insert(norm.Get());
  return AddMegakernelVector(build, norm.Inputs()[1], dim);
}

// The odml.rms_norm that produces `tensor` (after reshapes, which are
// covered), if one does.
std::optional<Op> MegakernelNormOf(MegakernelBuild& build, Tensor tensor) {
  for (int hops = 0; hops < 4; ++hops) {
    const auto producer = tensor.GetDefiningOp();
    if (!producer) return std::nullopt;
    if (producer->Code() == kLiteRtOpCodeTflReshape) {
      build.covered.insert(producer->Get());
      tensor = producer->Inputs()[0];
      continue;
    }
    if (producer->Code() == kLiteRtOpCodeShloComposite &&
        CompositeOpName(*producer) == "odml.rms_norm" &&
        producer->Inputs().size() == 2) {
      return *producer;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// A FULLY_CONNECTED op's blockwise INT4 weights and FP16 scales, put in the
// weight store as the GEMV plugin puts them (the scales after the
// weights), so that the engines of the model share them.
Expected<MegakernelProjectionSource> MegakernelProjection(
    MegakernelBuild& build, const Op& fc) {
  if (fc.Code() != kLiteRtOpCodeTflFullyConnected || fc.Inputs().size() != 2 ||
      !HasNoFusedActivation(fc)) {
    return MegakernelUnsupported(
        "a projection that is not a plain "
        "FULLY_CONNECTED");
  }
  LITERT_ASSIGN_OR_RETURN(auto info, InspectSubbyteGemvWeights(fc.Inputs()[1]));
  if (info.bit_width != 4 || info.scale_block != 32 || !info.scales ||
      info.scales->ElementType() != litert::ElementType::Float16) {
    return MegakernelUnsupported(
        "a projection without blockwise INT4 weights and FP16 scales");
  }
  const auto weights = fc.Inputs()[1].Weights().Bytes();
  const absl::Span<const uint8_t> members[] = {weights,
                                               info.scales->Weights().Bytes()};
  LITERT_ASSIGN_OR_RETURN(auto location, weight_store_->Add(members));
  MegakernelProjectionSource source;
  const auto segment =
      std::find(build.segments.begin(), build.segments.end(), location.segment);
  source.holder = static_cast<int32_t>(segment - build.segments.begin());
  if (segment == build.segments.end()) {
    build.segments.push_back(location.segment);
  }
  source.weights = static_cast<int64_t>(location.offset);
  source.scales = static_cast<int64_t>(location.offset + weights.size());
  source.rows = info.rows;
  source.cols = info.columns;
  build.covered.insert(fc.Get());
  return source;
}

// Whether `tensor` is the step's positions (input_pos, reshaped).
static bool IsMegakernelPositions(const MegakernelBuild& build,
                                  const Tensor& tensor) {
  return StripReshapes(tensor).Get() == build.positions.Get();
}

// The attention of one layer from its cache update; fills the attention
// part of `layer` and returns the residual stream behind the layer's
// attention (h), whose layer input is `x`.
Expected<Tensor> MegakernelAttention(MegakernelBuild& build, const Op& update,
                                     const Tensor& x,
                                     MegakernelLayerSource& layer) {
  const auto layout = DecodeCacheUpdateMode(update);
  if (!layout.has_value() || layout->rows != 1) {
    return MegakernelUnsupported("a cache update that is not one decode row");
  }
  CompositeOptions options;
  if (options.InitFromOp(update.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return MegakernelUnsupported("a cache update without attributes");
  }
  const auto ring = options.attributes_map.value()["is_ring_buffer"];
  layer.ring = !ring.IsNull() && ring.AsBool() ? 1 : 0;
  for (int i = 0; i < 2; ++i) {
    const Tensor cache = update.Inputs()[3 + i];
    const auto in_place = in_place_caches_.find(cache.Get());
    if (in_place == in_place_caches_.end() ||
        in_place->second.op != update.Get() ||
        (i == 1 && !in_place->second.transposed) ||
        !IsFp16StoredCacheType(cache.ElementType()) ||
        raw_input_map_.count(cache.Get()) == 0) {
      return MegakernelUnsupported(
          "a cache that is not an FP16 network input updated in place, "
          "values as rows");
    }
    (i == 0 ? layer.k_cache : layer.v_cache) =
        static_cast<int32_t>(build.caches.size());
    build.caches.push_back(cache.Get());
  }
  layer.cache_len = layout->seq;
  layer.kv_heads = layout->heads;
  layer.head_dim = layout->depth;
  build.covered.insert(update.Get());

  // The attention over the updated caches.
  std::optional<Op> sdpa;
  for (const auto& use : update.Outputs()[0].Uses()) {
    if (use.user.Code() == kLiteRtOpCodeShloComposite &&
        CompositeOpName(use.user) == "odml.sdpa_transposed" &&
        use.user_arg_ind == 1) {
      sdpa = use.user;
    }
  }
  if (!sdpa || update.Outputs()[0].Uses().size() != 1 ||
      update.Outputs()[1].Uses().size() != 1 || sdpa->Inputs().size() < 4 ||
      sdpa->Inputs()[2].Get() != update.Outputs()[1].Get() ||
      !SdpaTransposedGroup(*sdpa).has_value()) {
    return MegakernelUnsupported(
        "a cache update not read by one odml.sdpa_transposed");
  }
  build.covered.insert(sdpa->Get());
  const Tensor mask = sdpa->Inputs()[3];
  if (!mask.IsSubgraphInput() || build.other_inputs.count(mask.Get()) == 0) {
    return MegakernelUnsupported("an attention mask that is not an input");
  }
  const auto known =
      std::find(build.masks.begin(), build.masks.end(), mask.Get());
  layer.mask = static_cast<int32_t>(known - build.masks.begin());
  if (known == build.masks.end()) build.masks.push_back(mask.Get());

  // The queries, keys and values: odml.qkv_norm_rope over the qkv
  // projection (sliding layers), or its slices through the head norms and
  // odml.rope (global layers).
  const Tensor q = sdpa->Inputs()[0];
  const auto q_producer = q.GetDefiningOp();
  if (!q_producer || q_producer->Code() != kLiteRtOpCodeShloComposite) {
    return MegakernelUnsupported("queries not from a composite");
  }
  std::optional<Tensor> qkv;
  if (CompositeOpName(*q_producer) == "odml.qkv_norm_rope") {
    const Op& qnr = *q_producer;
    const auto parsed = ParseQkvNormRope(qnr);
    if (!parsed.has_value() || parsed->params.has_v_norm == 0 ||
        parsed->params.kv_heads != layer.kv_heads ||
        parsed->params.depth != layer.head_dim ||
        parsed->params.epsilon != build.description.norm_eps ||
        update.Inputs()[0].Get() != qnr.Outputs()[1].Get() ||
        update.Inputs()[1].Get() != qnr.Outputs()[2].Get() ||
        !IsMegakernelPositions(build, qnr.Inputs()[1])) {
      return MegakernelUnsupported(
          "an odml.qkv_norm_rope that is not this layer's");
    }
    const auto& p = parsed->params;
    layer.shared_kv = 0;
    layer.q_heads = p.heads;
    layer.rotary_dim = p.rotary;
    layer.rope_min_timescale = p.min_timescale;
    layer.rope_max_timescale = p.max_timescale;
    LITERT_ASSIGN_OR_RETURN(
        layer.q_norm, AddMegakernelVector(build, qnr.Inputs()[2], p.depth));
    LITERT_ASSIGN_OR_RETURN(
        layer.k_norm, AddMegakernelVector(build, qnr.Inputs()[3], p.depth));
    build.covered.insert(qnr.Get());
    qkv = qnr.Inputs()[0];
  } else if (CompositeOpName(*q_producer) == "odml.rope") {
    LITERT_ASSIGN_OR_RETURN(
        qkv, MegakernelGlobalHeads(build, update, *q_producer, layer));
  } else {
    return MegakernelUnsupported("queries from " +
                                 CompositeOpName(*q_producer));
  }
  if (SdpaTransposedGroup(*sdpa).value() * layer.kv_heads != layer.q_heads) {
    return MegakernelUnsupported("an attention of other heads");
  }

  // The qkv projection behind the layer's input norm.
  const auto fc = qkv->GetDefiningOp();
  if (!fc) return MegakernelUnsupported("qkv is not a projection");
  LITERT_ASSIGN_OR_RETURN(layer.qkv, MegakernelProjection(build, *fc));
  const int32_t qkv_rows =
      (layer.q_heads + (layer.shared_kv != 0 ? 1 : 2) * layer.kv_heads) *
      layer.head_dim;
  const auto input_norm = MegakernelNormOf(build, fc->Inputs()[0]);
  if (layer.qkv.rows != qkv_rows ||
      layer.qkv.cols != build.description.model_dim || !input_norm ||
      input_norm->Inputs()[0].Get() != x.Get()) {
    return MegakernelUnsupported(
        "a qkv projection that is not the layer input's, normed");
  }
  LITERT_ASSIGN_OR_RETURN(
      layer.input_norm,
      AddMegakernelNorm(build, *input_norm, build.description.model_dim));

  // The o projection, its norm and the residual update.
  const auto reshape = SingleUser(sdpa->Outputs()[0], 0);
  const auto o = reshape && reshape->Code() == kLiteRtOpCodeTflReshape
                     ? SingleUser(reshape->Outputs()[0], 0)
                     : std::nullopt;
  if (!o) return MegakernelUnsupported("no o projection");
  build.covered.insert(reshape->Get());
  LITERT_ASSIGN_OR_RETURN(layer.o, MegakernelProjection(build, *o));
  const auto post = SingleUser(o->Outputs()[0], 0);
  if (layer.o.rows != build.description.model_dim ||
      layer.o.cols != layer.q_heads * layer.head_dim || !post ||
      post->Code() != kLiteRtOpCodeShloComposite ||
      CompositeOpName(*post) != "odml.rms_norm") {
    return MegakernelUnsupported("an o projection without its norm");
  }
  LITERT_ASSIGN_OR_RETURN(
      layer.post_attention_norm,
      AddMegakernelNorm(build, *post, build.description.model_dim));
  const auto uses = post->Outputs()[0].Uses();
  if (uses.size() != 1 || uses[0].user.Code() != kLiteRtOpCodeTflAdd ||
      !HasNoFusedActivation(uses[0].user) ||
      uses[0].user.Inputs()[1 - uses[0].user_arg_ind].Get() != x.Get()) {
    return MegakernelUnsupported("an attention without its residual update");
  }
  build.covered.insert(uses[0].user.Get());
  return uses[0].user.Outputs()[0];
}

// A global layer's head norms and RoPE from the odml.rope of its queries:
//   q = rope(reshape(norm_q(reshape(slice(qkv)))))
//   k = rope(reshape(norm_k(k4))), k4 = reshape(slice(qkv))
//   v = reshape(norm_v(k4)) with norm_v's scale all ones
// Returns qkv.
Expected<Tensor> MegakernelGlobalHeads(MegakernelBuild& build, const Op& update,
                                       const Op& rope_q,
                                       MegakernelLayerSource& layer) {
  layer.shared_kv = 1;
  // A rope and the head norm in front of it; the norm's input.
  struct Rope {
    float min_timescale = 1.0f;
    float max_timescale = 10000.0f;
    float proportion = 1.0f;
    Tensor heads;  // the norm's input
    int64_t norm = 0;
  };
  auto rope_of = [&](const Op& rope, int32_t heads) -> Expected<Rope> {
    CompositeOptions options;
    if (rope.Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(rope) != "odml.rope" || rope.Inputs().size() != 2 ||
        options.InitFromOp(rope.Get()) != kLiteRtStatusOk ||
        !options.attributes_map.has_value() ||
        !IsMegakernelPositions(build, rope.Inputs()[1])) {
      return MegakernelUnsupported("a rope that is not the step's");
    }
    const auto& map = options.attributes_map.value();
    Rope result;
    if (!map["min_timescale"].IsNull()) {
      result.min_timescale = map["min_timescale"].AsFloat();
    }
    if (!map["max_timescale"].IsNull()) {
      result.max_timescale = map["max_timescale"].AsFloat();
    }
    if (!map["proportion"].IsNull()) {
      result.proportion = map["proportion"].AsFloat();
    }
    const auto norm = MegakernelNormOf(build, rope.Inputs()[0]);
    if (!norm) return MegakernelUnsupported("a rope without its head norm");
    LITERT_ASSIGN_OR_RETURN(result.norm,
                            AddMegakernelNorm(build, *norm, layer.head_dim));
    result.heads = norm->Inputs()[0];
    LITERT_ASSIGN_OR_RETURN(auto type, result.heads.RankedTensorType());
    const auto dims = type.Layout().Dimensions();
    if (dims.size() != 4 || dims[2] != heads || dims[3] != layer.head_dim) {
      return MegakernelUnsupported("a head norm over other heads");
    }
    build.covered.insert(rope.Get());
    return result;
  };
  // The slice of qkv a head norm reads, at `begin` columns.
  auto slice_of = [&](const Tensor& heads, int32_t begin,
                      int32_t width) -> Expected<Tensor> {
    const auto reshape = heads.GetDefiningOp();
    if (!reshape || reshape->Code() != kLiteRtOpCodeTflReshape) {
      return MegakernelUnsupported("head norms not over a slice of qkv");
    }
    const auto slice = reshape->Inputs()[0].GetDefiningOp();
    if (!slice || slice->Code() != kLiteRtOpCodeTflSlice) {
      return MegakernelUnsupported("head norms not over a slice of qkv");
    }
    const auto starts = ReadInt32Constant(slice->Inputs()[1]);
    const auto sizes = ReadInt32Constant(slice->Inputs()[2]);
    if (!starts || !sizes || starts->empty() || sizes->empty() ||
        starts->back() != begin || sizes->back() != width) {
      return MegakernelUnsupported("a slice of qkv at other columns");
    }
    build.covered.insert(reshape->Get());
    build.covered.insert(slice->Get());
    return slice->Inputs()[0];
  };

  const auto rope_k = update.Inputs()[0].GetDefiningOp();
  if (!rope_k) return MegakernelUnsupported("keys not from a rope");
  // q [1, H, 1, D] after the rope: its heads.
  LITERT_ASSIGN_OR_RETURN(auto q_type, rope_q.Outputs()[0].RankedTensorType());
  layer.q_heads = q_type.Layout().Dimensions()[1];
  LITERT_ASSIGN_OR_RETURN(const Rope q, rope_of(rope_q, layer.q_heads));
  LITERT_ASSIGN_OR_RETURN(const Rope k, rope_of(*rope_k, layer.kv_heads));
  if (q.min_timescale != k.min_timescale ||
      q.max_timescale != k.max_timescale || q.proportion != k.proportion) {
    return MegakernelUnsupported("queries and keys rotated differently");
  }
  layer.q_norm = q.norm;
  layer.k_norm = k.norm;
  layer.rope_min_timescale = q.min_timescale;
  layer.rope_max_timescale = q.max_timescale;
  layer.rotary_dim =
      static_cast<int32_t>(std::lround(layer.head_dim * q.proportion));
  layer.rotary_dim -= layer.rotary_dim % 2;
  // The values: the keys' heads normalized without a scale.
  const auto v_norm = MegakernelNormOf(build, update.Inputs()[1]);
  const auto v_scale =
      v_norm ? RmsNormWeights(*v_norm, layer.head_dim) : std::nullopt;
  if (!v_scale || v_norm->Inputs()[0].Get() != k.heads.Get() ||
      v_scale->second != build.description.norm_eps ||
      std::any_of(v_scale->first.begin(), v_scale->first.end(),
                  [](float s) { return s != 1.0f; })) {
    return MegakernelUnsupported(
        "values that are not the keys normalized without a scale");
  }
  build.covered.insert(v_norm->Get());
  const int32_t q_width = layer.q_heads * layer.head_dim;
  LITERT_ASSIGN_OR_RETURN(auto q_source, slice_of(q.heads, 0, q_width));
  LITERT_ASSIGN_OR_RETURN(
      auto k_source,
      slice_of(k.heads, q_width, layer.kv_heads * layer.head_dim));
  if (q_source.Get() != k_source.Get()) {
    return MegakernelUnsupported("queries and keys of two projections");
  }
  return q_source;
}

// The feed-forward block behind `h` as FindMoeBlocks planned it; fills the
// block part of `layer` and returns the layer's output.
Expected<Tensor> MegakernelBlock(MegakernelBuild& build, const Tensor& h,
                                 MegakernelLayerSource& layer) {
  const auto found = build.moe_by_residual.find(h.Get());
  if (found == build.moe_by_residual.end()) {
    return MegakernelUnsupported("an attention not followed by a moe block");
  }
  const Op op(h.Context(), found->second);
  const auto moe = ParseMoeCustomOp(op);
  const auto& plan = moe_plans_.at(found->second);
  if (!moe.has_value() || !plan.routing.has_value() ||
      !plan.block.has_value()) {
    return MegakernelUnsupported("a moe op without its layer block");
  }
  const auto& routing = *plan.routing;
  const auto& block = *plan.block;
  LITERT_ASSIGN_OR_RETURN(layer.arena, MegakernelExpertPieces(op, *moe));
  if (!AddMegakernelRouterPieces(routing, layer.arena) ||
      !AddMegakernelDensePieces(block, layer.arena)) {
    return MegakernelUnsupported(
        "a moe block whose router or dense weights are not in the file");
  }
  if (routing.eps != build.description.norm_eps ||
      moe->model_dim != build.description.model_dim) {
    return MegakernelUnsupported("a moe block of other norms or dims");
  }
  const int64_t dim = build.description.model_dim;
  layer.experts = moe->experts;
  layer.top_k = moe->top_k;
  layer.expert_hidden = moe->hidden_dim;
  layer.expert_gate = moe->gate;
  layer.dense_gate = block.gate_mode;
  layer.router_multiplier = routing.multiplier;
  layer.residual_multiplier = block.multiplier;
  LITERT_ASSIGN_OR_RETURN(
      layer.per_expert_scale,
      AddMegakernelVector(build, op.Inputs()[9], moe->experts));
  LITERT_ASSIGN_OR_RETURN(layer.router_norm,
                          AddMegakernelVector(build, routing.router_norm, dim));
  LITERT_ASSIGN_OR_RETURN(layer.expert_norm,
                          AddMegakernelVector(build, routing.expert_norm, dim));
  LITERT_ASSIGN_OR_RETURN(layer.dense_norm,
                          AddMegakernelVector(build, block.dense_norm, dim));
  LITERT_ASSIGN_OR_RETURN(layer.dense_post,
                          AddMegakernelVector(build, block.dense_post, dim));
  LITERT_ASSIGN_OR_RETURN(layer.moe_post,
                          AddMegakernelVector(build, block.moe_post, dim));
  LITERT_ASSIGN_OR_RETURN(layer.sum_post,
                          AddMegakernelVector(build, block.sum_post, dim));
  build.covered.insert(op.Get());
  for (LiteRtOp absorbed : block.absorbed) build.covered.insert(absorbed);
  return block.output;
}

Expected<void> LowerMegakernel(const Subgraph& subgraph) {
  if (weight_store_ == nullptr) {
    return MegakernelUnsupported("the program needs the weight store");
  }
  if (!Fp16ActivationsEnabled() ||
      ModeFloatType() != nvinfer1::DataType::kBF16) {
    return MegakernelUnsupported("the program needs BF16 activations");
  }
  MegakernelBuild build;
  MegakernelDescription& d = build.description;

  // The inputs in front of the caches, in the signature's order.
  std::vector<nvinfer1::ITensor*> other_inputs;
  std::optional<Tensor> embeddings;
  for (const auto& input : subgraph.Inputs()) {
    if (in_place_caches_.count(input.Get()) != 0) continue;
    const auto raw = raw_input_map_.find(input.Get());
    if (raw == raw_input_map_.end()) {
      return MegakernelUnsupported("an input that is not a network input");
    }
    const std::string name(input.Name());
    const auto index = static_cast<int32_t>(other_inputs.size());
    if (HasSignatureKey(name, "embeddings")) {
      embeddings = input;
      d.embeddings = index;
    } else if (HasSignatureKey(name, "input_pos")) {
      build.positions = input;
      d.positions = index;
    }
    build.other_inputs[input.Get()] = index;
    other_inputs.push_back(raw->second);
    d.input_types.push_back(static_cast<int32_t>(raw->second->getType()));
  }
  if (!embeddings || build.positions.Get() == nullptr ||
      other_inputs[d.embeddings]->getType() != nvinfer1::DataType::kFLOAT ||
      other_inputs[d.positions]->getType() != nvinfer1::DataType::kINT32) {
    return MegakernelUnsupported("no FP32 embeddings or INT32 positions");
  }
  LITERT_ASSIGN_OR_RETURN(auto x_type, embeddings->RankedTensorType());
  d.model_dim = x_type.Layout().Dimensions().back();
  // Every rms_norm of the body has the epsilon of the first.
  for (const auto& op : subgraph.Ops()) {
    if (const auto weights = RmsNormWeights(op, d.model_dim)) {
      d.norm_eps = weights->second;
      break;
    }
  }
  for (const auto& [moe, plan] : moe_plans_) {
    if (plan.routing.has_value()) {
      build.moe_by_residual[StripReshapes(plan.routing->residual).Get()] = moe;
    }
  }

  // The layers, in the order of their cache updates.
  Tensor x = *embeddings;
  for (const auto& op : subgraph.Ops()) {
    if (op.Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(op) != "odml.cache_update") {
      continue;
    }
    MegakernelLayerSource& layer = d.layers.emplace_back();
    LITERT_ASSIGN_OR_RETURN(auto h, MegakernelAttention(build, op, x, layer));
    LITERT_ASSIGN_OR_RETURN(x, MegakernelBlock(build, h, layer));
  }
  if (d.layers.empty()) return MegakernelUnsupported("no layers");

  // The tail: the final norm (the activations), the head, its soft cap.
  std::optional<Op> final_norm;
  for (const auto& use : x.Uses()) {
    if (use.user.Code() == kLiteRtOpCodeShloComposite &&
        CompositeOpName(use.user) == "odml.rms_norm") {
      final_norm = use.user;
    }
  }
  if (!final_norm || x.Uses().size() != 1) {
    return MegakernelUnsupported("the last layer's output is not normed");
  }
  LITERT_ASSIGN_OR_RETURN(d.final_norm,
                          AddMegakernelNorm(build, *final_norm, d.model_dim));
  const Tensor activations = final_norm->Outputs()[0];
  std::optional<Op> head;
  for (const auto& use : activations.Uses()) {
    if (use.user.Code() == kLiteRtOpCodeTflFullyConnected) head = use.user;
  }
  if (!head || activations.Uses().size() != 1) {
    return MegakernelUnsupported("the final norm does not feed the head");
  }
  LITERT_ASSIGN_OR_RETURN(d.head, MegakernelProjection(build, *head));
  d.vocab = d.head.rows;
  const auto scale_in = SingleUser(head->Outputs()[0], 0);
  const auto tanh = scale_in && scale_in->Code() == kLiteRtOpCodeTflMul
                        ? SingleUser(scale_in->Outputs()[0], 0)
                        : std::nullopt;
  const auto scale_out = tanh && tanh->Code() == kLiteRtOpCodeTflTanh
                             ? SingleUser(tanh->Outputs()[0], 0)
                             : std::nullopt;
  const auto c_in =
      scale_in ? ReadFloatScalar(scale_in->Inputs()[1]) : std::nullopt;
  const auto c_out = scale_out && scale_out->Code() == kLiteRtOpCodeTflMul
                         ? ReadFloatScalar(scale_out->Inputs()[1])
                         : std::nullopt;
  if (!c_in || !c_out || !HasNoFusedActivation(*scale_in) ||
      !HasNoFusedActivation(*scale_out)) {
    return MegakernelUnsupported("a head without its soft cap");
  }
  d.soft_cap_in = *c_in;
  d.soft_cap = *c_out;
  build.covered.insert(scale_in->Get());
  build.covered.insert(tanh->Get());
  build.covered.insert(scale_out->Get());
  const Tensor logits = scale_out->Outputs()[0];

  // What else the body computes may only feed what was recognized: the
  // routing chains of the moe blocks, the cache parameters and positions.
  const auto ops = subgraph.Ops();
  std::unordered_set<LiteRtTensor> outputs;
  for (const auto& output : subgraph.Outputs()) outputs.insert(output.Get());
  for (bool changed = true; changed;) {
    changed = false;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      if (build.covered.count(it->Get()) != 0) continue;
      bool feeds_covered = !it->Outputs().empty();
      for (const auto& output : it->Outputs()) {
        if (outputs.count(output.Get()) != 0) feeds_covered = false;
        for (const auto& use : output.Uses()) {
          if (build.covered.count(use.user.Get()) == 0) {
            feeds_covered = false;
          }
        }
      }
      if (feeds_covered) {
        build.covered.insert(it->Get());
        changed = true;
      }
    }
  }
  for (const auto& op : ops) {
    if (build.covered.count(op.Get()) == 0) {
      return MegakernelUnsupported(
          "an op the program does not run: op code " +
          std::to_string(static_cast<int>(op.Code())) +
          (op.Code() == kLiteRtOpCodeShloComposite ? " " + CompositeOpName(op)
                                                   : "") +
          ", output tensor " +
          (op.Outputs().empty()
               ? std::string("none")
               : std::to_string(op.Outputs()[0].TensorIndex())));
    }
  }
  if (build.masks.empty() || build.masks.size() > 2) {
    return MegakernelUnsupported("one or two attention masks");
  }
  for (LiteRtTensor mask : build.masks) {
    d.masks.push_back(build.other_inputs.at(mask));
  }

  // The plugin layer.
  d.norm_values = static_cast<int64_t>(build.norms.size());
  d.holders = static_cast<int32_t>(build.segments.size());
  d.holder_granule = static_cast<int64_t>(weight_store_->granule());
  nvinfer1::Dims norm_dims{};
  norm_dims.nbDims = 1;
  norm_dims.d[0] = d.norm_values;
  LITERT_ASSIGN_OR_RETURN(
      auto* norms, AddFloatConstant(build.norms, norm_dims, "megakernel_norms",
                                    nvinfer1::DataType::kFLOAT));
  std::vector<nvinfer1::ITensor*> inputs = other_inputs;
  for (LiteRtTensor cache : build.caches) {
    inputs.push_back(raw_input_map_.at(cache));
  }
  inputs.push_back(norms);
  const int32_t first_holder = static_cast<int32_t>(inputs.size());
  for (int32_t h = 0; h < d.holders; ++h) {
    LITERT_ASSIGN_OR_RETURN(auto* provisional, ProvisionalWeightHolder());
    inputs.push_back(provisional);
  }
  TrtPtr<nvinfer1::IPluginV3> plugin(CreateMegakernelPlugin(d));
  if (!plugin) {
    return Error(kLiteRtStatusErrorCompilation,
                 "Failed to create the megakernel plugin");
  }
  auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
  if (layer == nullptr ||
      layer->getNbOutputs() != 3 + static_cast<int32_t>(build.caches.size())) {
    return Error(kLiteRtStatusErrorCompilation,
                 "Failed to add the megakernel plugin layer");
  }
  layer->setName(KeepName(UniqueName("megakernel_decode")));
  owned_plugins_.push_back(std::move(plugin));
  for (int32_t h = 0; h < d.holders; ++h) {
    holder_readers_.push_back(
        {layer, build.segments[h], /*scales_too=*/false, first_holder + h});
  }
  // The logits and activations leave as the plugin wrote them, FP32.
  for (const auto& [tensor, index] :
       {std::make_pair(logits, 0), std::make_pair(activations, 1)}) {
    nvinfer1::ITensor* out = layer->getOutput(index);
    out->setName(KeepName(UniqueName("megakernel_output_" +
                                     std::to_string(tensor.TensorIndex()))));
    tensor_map_[tensor.Get()] = out;
    raw_output_map_[tensor.Get()] = out;
  }
  // Each cache's new row, written again where the plugin wrote it by a
  // native in-place cache write, whose output aliases the cache input.
  nvinfer1::ITensor* rows = layer->getOutput(2);
  size_t cache = 0;
  for (const auto& op : ops) {
    if (op.Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(op) != "odml.cache_update") {
      continue;
    }
    const int layer_index = static_cast<int>(cache / 2);
    LITERT_ASSIGN_OR_RETURN(auto* row, SliceTensor1d(rows, layer_index, 1));
    for (int i = 0; i < 2; ++i, ++cache) {
      auto* write = network_->addKVCacheUpdate(
          *raw_input_map_.at(build.caches[cache]),
          *layer->getOutput(3 + static_cast<int32_t>(cache)), *row,
          nvinfer1::KVCacheMode::kLINEAR);
      if (write == nullptr || write->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add a megakernel cache write");
      }
      raw_output_map_[op.Outputs()[i].Get()] = write->getOutput(0);
      LITERT_RETURN_IF_ERROR(
          SetOutputTensor(op.Outputs()[i], write->getOutput(0)));
    }
  }
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX megakernel: the decode body as one "
             "program: %zu layers, model_dim %d, vocab %d, %zu caches, %zu "
             "masks, %lld norm values, %d weight holders",
             d.layers.size(), d.model_dim, d.vocab, build.caches.size(),
             d.masks.size(), static_cast<long long>(d.norm_values), d.holders);
  return {};
}
