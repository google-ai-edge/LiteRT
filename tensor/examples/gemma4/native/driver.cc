/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Native matched-bundle runner with compact INT2 weights and active INT8 KV.
#include "tensor/examples/gemma4/native/driver_support.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include "tensor/examples/gemma4/native/model/helpers/int8_kv_cache.h"
#include "tensor/examples/gemma4/native/matched_bundle_loader.h"
#include "xnnpack/subgraph.h"

ABSL_FLAG(int, num_threads, 2, "Number of threads for the shared XNNPACK pool.");
ABSL_FLAG(bool, weight_cache, true, "Share packed weights across native stages.");
ABSL_FLAG(bool, consistent_arithmetic, false, "Unsupported diagnostic compatibility flag; must remain false.");
ABSL_FLAG(std::string, dump_logits, "", "Use --dump_full_logits instead; must remain empty.");
ABSL_FLAG(bool, dump_intermediates, false, "Use --trace_position instead; must remain false.");

ABSL_FLAG(std::string, bundle_dir, "", "Verified actual-bundle export directory.");

ABSL_FLAG(std::string, cases_file, "", "TSV: case ID, prompt CSV, forced decode-input CSV (or -).");
ABSL_FLAG(std::string, output_dir, "", "New, nonexisting artifact directory.");
ABSL_FLAG(int, warmup_runs, 1, "Discarded warmup repetitions for each case.");
ABSL_FLAG(int, measured_runs, 3, "Measured repetitions for each case.");
ABSL_FLAG(bool, reuse_runtimes, false, "Reuse the case's compiled runtimes; otherwise compile fresh per repetition.");
ABSL_FLAG(bool, dump_full_logits, false, "Write each pass's full final-position vocabulary logits after timing in the first measured repetition only.");

#include "tensor/examples/gemma4/native/active_kv_bank.h"
#include "tensor/examples/gemma4/native/stage_runner.h"
#include "tensor/examples/gemma4/native/memory_snapshot.h"
#include "xnnpack/cache.h"
ABSL_FLAG(bool, share_workspace, false, "Share scratch across serialized stages.");
ABSL_FLAG(bool, preserve_static_int2, false, "Keep original INT2 MLP weights compact with static activation quantization.");
ABSL_FLAG(bool, memory_report, false, "Record diagnostic phase RSS/PSS and allocation counters; timing is ineligible.");
#include "tensor/examples/gemma4/native/active_runtime_audit.h"
ABSL_FLAG(int, prefill_chunk_rows, 128, "Reusable prefill signature rows (128 or 1024).");
ABSL_FLAG(int, cache_capacity, 2048, "Persistent cache capacity; attention uses only valid rows.");
ABSL_FLAG(bool, dump_cache, false, "Dump committed token-major cache after each pass for correctness.");

ABSL_FLAG(int, kv_alignment, 32, "Align attention start/end to this power of two (1..128), masking padding.");
ABSL_FLAG(int, trace_position, -1, "Dump a selected absolute token position at existing stage boundaries.");
ABSL_FLAG(bool, fixed_attention_extent, false, "Diagnostic: restore fixed attention widths while retaining token-major caches.");

namespace litert::tensor::examples::gemma4::native {
namespace {
using BenchClock = std::chrono::steady_clock;
double Milliseconds(BenchClock::time_point start) {
  return std::chrono::duration<double, std::milli>(BenchClock::now() - start).count();
}
long PeakRssKiB() {
  struct rusage usage {};
  return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
}

struct BenchCase {
  std::string id;
  std::vector<int32_t> prompt;
  std::vector<int32_t> forced;
};

absl::StatusOr<std::vector<BenchCase>> ReadCases(const std::string& path) {
  std::ifstream file(path);
  if (!file) return absl::InvalidArgumentError("Cannot open --cases_file");
  std::vector<BenchCase> cases;
  std::set<std::string> seen;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto first = line.find('\t');
    const auto second = first == std::string::npos ? first : line.find('\t', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        line.find('\t', second + 1) != std::string::npos)
      return absl::InvalidArgumentError("Cases require exactly three tab-separated fields");
    BenchCase item;
    item.id = line.substr(0, first);
    if (item.id.empty() || item.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos ||
        !seen.insert(item.id).second)
      return absl::InvalidArgumentError("Unsafe or duplicate case ID");
    LRT_TENSOR_ASSIGN_OR_RETURN(item.prompt, ParseTokenIds(line.substr(first + 1, second - first - 1)));
    const std::string forced = line.substr(second + 1);
    if (!forced.empty() && forced != "-") {
      LRT_TENSOR_ASSIGN_OR_RETURN(item.forced, ParseTokenIds(forced));
    }
    if (item.prompt.empty() || item.prompt.front() != 2 ||
        item.prompt.size() + item.forced.size() >= std::numeric_limits<int>::max())
      return absl::InvalidArgumentError("Prompt must begin with BOS2 and total length must fit int");
    for (const auto& ids : {item.prompt, item.forced})
      for (int32_t token : ids)
        if (token < 0 || token >= 262144)
          return absl::InvalidArgumentError("Token outside E2B vocabulary");
    cases.push_back(std::move(item));
  }
  if (!file.eof() || cases.empty()) return absl::InvalidArgumentError("Empty or invalid cases file");
  return cases;
}

struct SetupTimes { double graph_ms=0, compile_pack_ms=0, total_ms=0; };
using Inputs=Gemma4Inputs<XnnpackMixinTag>;
using Outputs=Gemma4Outputs<XnnpackMixinTag>;
using SR = std::unique_ptr<StageRunner>;
struct AttentionStage {
  XnnTensor q,k,v,mask,output;
  SR runner;
};
struct LayerStage { SR projection, post; AttentionStage attention; };
struct LiveSignature {
  int rows=0, last_layer=0;
  bool decode=false;
  Inputs inputs;
  Outputs outputs;
  ActiveGraphContext cuts;
  SR preprocess, tail;
  std::vector<LayerStage> stages;
};
struct PoolDeleter { void operator()(pthreadpool_t p)const { if(p)pthreadpool_destroy(p); } };
struct WorkspaceDeleter { void operator()(xnn_workspace_t p) const { if(p) xnn_release_workspace(p); } };
struct LiveRuntime {
  // Resources precede their borrowers so runtimes are destroyed first.
  WeightsCache cache;
  std::unique_ptr<pthreadpool,PoolDeleter> pool;
  std::unique_ptr<xnn_workspace,WorkspaceDeleter> workspace;
  std::unique_ptr<LiveSignature> prefill,decode;
  SetupTimes times;
  size_t static_int2_tensors=0, static_int2_operators=0, static_int2_compact_bytes=0;
};

absl::Status RecordMemory(const std::string& phase, const LiveRuntime* rt = nullptr,
    const LoadedTensors* loaded = nullptr, const std::string& case_id = "", int repetition = -1,
    size_t bank_bytes = 0) {
  if (!absl::GetFlag(FLAGS_memory_report)) return absl::OkStatus();
  gemma_memory::Counters counts;
  counts["logical_kv_bytes"] = bank_bytes;
  std::set<const void*> seen_workspaces, seen_owned_buffers, seen_model_buffers;
  if (loaded) for (const auto& entry : loaded->weights_handle) {
    const auto& handle = entry.second;
    auto buffer = handle.GetBuffer();
    if (!buffer.ok()) continue;
    if (!seen_model_buffers.insert(&*buffer).second) continue;
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto bytes, buffer->ByteSize());
    counts["model_source_buffer_bytes"] += bytes;
    counts["model_source_buffer_count"]++;
  }
  auto stage = [&](const SR& runner)->absl::Status {
    if (!runner) return absl::OkStatus();
    LRT_TENSOR_ASSIGN_OR_RETURN(auto info, runner->MemorySnapshot());
    counts["stage_count"]++;
    if (info.workspace && seen_workspaces.insert(info.workspace).second) {
      counts["unique_workspace_count"]++;
      counts["unique_workspace_bytes"] += info.workspace_bytes;
    }
    counts["stage_workspace_bytes_sum_including_shared_aliases"] += info.workspace_bytes;
    counts["graph_constant_capacity_bytes"] += info.graph_constant_capacity_bytes;
    counts["graph_dequantized_capacity_bytes"] += info.graph_dequantized_capacity_bytes;
    counts["graph_fp16_capacity_bytes"] += info.graph_fp16_capacity_bytes;
    for (const auto& b : info.external_buffers) if (b.owns_storage && seen_owned_buffers.insert(b.buffer).second) {
      counts["owned_external_buffer_bytes"] += b.bytes;
      if (b.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) counts["owned_external_output_bytes"] += b.bytes;
    }
    return absl::OkStatus();
  };
  if (rt) {
    counts["static_int2_tensor_count"] = rt->static_int2_tensors;
    counts["static_int2_operator_count"] = rt->static_int2_operators;
    counts["static_int2_compact_bytes"] = rt->static_int2_compact_bytes;
    for (const auto* signature : {rt->prefill.get(), rt->decode.get()}) if (signature) {
      LRT_TENSOR_RETURN_IF_ERROR(stage(signature->preprocess));
      LRT_TENSOR_RETURN_IF_ERROR(stage(signature->tail));
      for (const auto& layer : signature->stages) {
        LRT_TENSOR_RETURN_IF_ERROR(stage(layer.projection));
        LRT_TENSOR_RETURN_IF_ERROR(stage(layer.attention.runner));
        LRT_TENSOR_RETURN_IF_ERROR(stage(layer.post));
      }
    }
    if (rt->cache) {
      const auto* cache = static_cast<const xnn_internal_weights_cache*>(rt->cache->context);
      counts["packed_weight_bytes"] = cache->cache.weights.size;
      counts["packed_weight_capacity_bytes"] = cache->cache.weights.capacity;
      counts["packed_weight_cache_entries"] = cache->cache.num_entries;
      counts["packed_weight_cache_hits"] = cache->cache.hits;
      counts["packed_weight_cache_misses"] = cache->cache.misses;
    }
  }
  if (!gemma_memory::AppendMemorySnapshot(absl::GetFlag(FLAGS_output_dir)+"/memory.jsonl",
      phase, case_id, repetition, gemma_memory::Object(counts)))
    return absl::InternalError("Could not write memory snapshot");
  return absl::OkStatus();
}

template<class Data>
absl::Status BindLive(StageRunner& runner,const TensorHandle& h,const Data& data) {
  if(!runner.graph().Lookup(h).ok())return absl::OkStatus();
  return runner.SetInput(h,data);
}
absl::Status BindFrom(StageRunner& dest,const TensorHandle& in,
    StageRunner& source,const TensorHandle& out) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto data,source.ReadOutput(out));
  return dest.SetInput(in,absl::Span<const std::byte>(data.data(),data.size()));
}
absl::StatusOr<Inputs> MakeLiveInputs(const Config& config,int rows,bool decode,
    const LoadedTensors& loaded,const std::vector<KvOwnerSpec>& specs,int capacity) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto inputs,CreateGemma4Inputs(config,rows,decode?1:0,loaded.weights_handle,false));
  inputs.bundle_dynamic_qd8_head=true;
  auto positions=XnnTensor({.name="matched_rope_positions",.type=Type::kFP32,.shape={1,1,rows,1}});
  inputs.weights["matched.rope.positions"]=positions;
  auto make_rope=[&](int dimension,float base,float proportion){
    const int half=dimension/2;
    std::vector<float> inverse(half,0.0f);
    for(int i=0;i<static_cast<int>(proportion*half);++i)inverse[i]=1.0f/std::pow(base,2.0f*i/dimension);
    auto frequency=XnnTensor({.type=Type::kFP32,.shape={1,1,1,half},.buffer=OwningCpuBuffer::Copy<Type::kFP32>(inverse)});
    auto angles=Mul(positions,frequency);
    auto cosine=Cos(angles),sine=Sin(angles);
    return std::make_pair(Concatenation({cosine,cosine},3),Concatenation({sine,sine},3));
  };
  std::tie(inputs.rope_global_cos,inputs.rope_global_sin)=make_rope(config.global_key_size,config.global_base_frequency,config.global_rope_proportion);
  std::tie(inputs.rope_local_cos,inputs.rope_local_sin)=make_rope(config.head_dim,config.local_base_frequency,config.local_rope_proportion);
  auto sharing=GetKvCacheSharingPatterns(config);
  for(int i=0;i<config.num_layers;++i) {
    auto spec=std::find_if(specs.begin(),specs.end(),[&](auto& s){return s.owner==sharing[i];});
    if(spec==specs.end())return absl::InvalidArgumentError("Missing owner scale");
    inputs.key_caches[i]=MakeInt8KeyCache<XnnpackMixinTag>(absl::StrCat("key_",i),capacity,spec->head_dim,spec->key_scale);
    // In staged mode these handles supply quantization metadata only.
    inputs.value_caches[i]=MakeInt8KeyCache<XnnpackMixinTag>(absl::StrCat("value_",i),capacity,spec->head_dim,spec->value_scale);
  }
  return inputs;
}
absl::StatusOr<AttentionStage> MakeAttentionStage(int rows,int heads,int dim,
    float ks,float vs,const Config& config,pthreadpool_t pool,xnn_weights_cache_t cache, xnn_workspace_t workspace) {
  XnnTensor q({.name="live_q",.type=Type::kFP32,.shape={1,1,heads*rows,dim}});
  auto k=MakeInt8KeyCache<XnnpackMixinTag>("live_k",1,dim,ks);
  auto v=MakeInt8KeyCache<XnnpackMixinTag>("live_v",1,dim,vs);
  XnnTensor mask({.name="live_mask",.type=Type::kFP32,.shape={1,1,heads*rows,1}});
  auto scores=BatchMatMul(q,Cast(k,Type::kFP32),false,true);
  if(config.attn_logits_soft_cap) {
    float cap=*config.attn_logits_soft_cap;
    scores=Mul(Tanh(Mul(scores,1.0f/cap)),cap);
  }
  auto probabilities=Softmax(Add(scores,mask));
  auto output=BatchMatMul(probabilities,Cast(v,Type::kFP32),false,false);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto runner,StageRunner::Create({output},pool,cache,0,workspace));
  LRT_TENSOR_RETURN_IF_ERROR(runner->PrepareRuntime());
  return AttentionStage{q,k,v,mask,output,std::move(runner)};
}
absl::StatusOr<std::unique_ptr<LiveSignature>> BuildSignature(const Config& config,
    int rows,bool decode,const LoadedTensors& loaded,const std::vector<KvOwnerSpec>& specs,
    int capacity,pthreadpool_t pool,xnn_weights_cache_t cache, xnn_workspace_t workspace) {
  auto sig=std::make_unique<LiveSignature>();sig->rows=rows;sig->decode=decode;
  sig->cuts.owners=GetKvCacheSharingPatterns(config);
  sig->last_layer=decode?config.num_layers-1:std::max_element(specs.begin(),specs.end(),[](auto&a,auto&b){return a.owner<b.owner;})->owner;
  LRT_TENSOR_ASSIGN_OR_RETURN(sig->inputs,MakeLiveInputs(config,rows,decode,loaded,specs,capacity));
  struct Scope { Scope(ActiveGraphContext* p){active_graph_context=p;} ~Scope(){active_graph_context=nullptr;} };
  { Scope scope(&sig->cuts);sig->outputs=BuildGemma4Graph(sig->inputs,config); }
  std::vector<TensorHandle> roots{sig->cuts.initial_hidden};
  for(int i=0;i<=sig->last_layer;++i)
    if(decode||i<sig->last_layer)roots.push_back(sig->cuts.ple_outputs[i]);
  LRT_TENSOR_ASSIGN_OR_RETURN(sig->preprocess,StageRunner::Create(roots,pool,cache,0,workspace));
  LRT_TENSOR_RETURN_IF_ERROR(sig->preprocess->PrepareRuntime());
  sig->stages.resize(sig->last_layer+1);
  for(int i=0;i<=sig->last_layer;++i) {
    auto& cut=sig->cuts.layers[i];auto& stage=sig->stages[i];
    bool stop=!decode&&i==sig->last_layer;
    roots.clear();if(!stop)roots.push_back(cut.query);
    if(i==cut.owner){roots.push_back(cut.new_key);roots.push_back(cut.new_value);}
    if(roots.empty())return absl::InternalError("Invalid cache-only final stage");
    LRT_TENSOR_ASSIGN_OR_RETURN(stage.projection,StageRunner::Create(roots,pool,cache,0,workspace));
    LRT_TENSOR_RETURN_IF_ERROR(stage.projection->PrepareRuntime());
    if(stop)break;
    const auto spec=std::find_if(specs.begin(),specs.end(),[&](auto&s){return s.owner==cut.owner;});
    LRT_TENSOR_ASSIGN_OR_RETURN(stage.attention,MakeAttentionStage(rows,config.num_heads,cut.dim,spec->key_scale,spec->value_scale,config,pool,cache,workspace));
    LRT_TENSOR_ASSIGN_OR_RETURN(stage.post,StageRunner::Create({cut.output},pool,cache,0,workspace));
    LRT_TENSOR_RETURN_IF_ERROR(stage.post->PrepareRuntime());
  }
  if(decode) {
    LRT_TENSOR_ASSIGN_OR_RETURN(sig->tail,StageRunner::Create({sig->outputs.logits},pool,cache,0,workspace));
    LRT_TENSOR_RETURN_IF_ERROR(sig->tail->PrepareRuntime());
  }
  int fc=0,bmm=0;
  auto count=[&](const SR& r) { if(!r)return;const auto* sg=static_cast<const XnnpackGraph&>(r->graph()).GetSubgraph();for(size_t i=0;i<sg->num_nodes;++i){fc+=sg->nodes[i].type==xnn_node_type_fully_connected;bmm+=sg->nodes[i].type==xnn_node_type_batch_matrix_multiply;} };
  count(sig->preprocess);count(sig->tail);
  for(auto&s:sig->stages){count(s.projection);count(s.post);count(s.attention.runner);}
  std::cerr<<(decode?"Decode":"Prefill")<<" stages: FC="<<fc<<" BMM="<<bmm<<" rows="<<rows<<"\n";
  if(fc!=(decode?277:129)||bmm!=(decode?70:28))return absl::InternalError("Staged graph work differs from matched bundle");
  return sig;
}
absl::StatusOr<std::unique_ptr<LiveRuntime>> CreateLiveRuntime(const Config& config,
    const LoadedTensors& loaded,const std::vector<KvOwnerSpec>& specs,int threads,int chunk,int capacity) {
  auto start=BenchClock::now();auto rt=std::make_unique<LiveRuntime>();
  xnn_weights_cache_t raw=nullptr;LRT_TENSOR_RETURN_IF_ERROR(xnn_create_weights_cache(&raw));rt->cache.reset(raw);
  if (absl::GetFlag(FLAGS_share_workspace)) { xnn_workspace_t scratch=nullptr; LRT_TENSOR_RETURN_IF_ERROR(xnn_create_workspace(&scratch)); rt->workspace.reset(scratch); }
  if(threads>1){rt->pool.reset(pthreadpool_create(threads));if(!rt->pool)return absl::ResourceExhaustedError("Threadpool allocation failed");}
  LRT_TENSOR_ASSIGN_OR_RETURN(rt->prefill,BuildSignature(config,chunk,false,loaded,specs,capacity,rt->pool.get(),raw,rt->workspace.get()));
  LRT_TENSOR_ASSIGN_OR_RETURN(rt->decode,BuildSignature(config,1,true,loaded,specs,capacity,rt->pool.get(),raw,rt->workspace.get()));
  const auto compact = GetStaticInt2WeightAudit(loaded.weights_handle);
  rt->static_int2_tensors=compact.tensor_count;
  rt->static_int2_compact_bytes=compact.compact_bytes;
  auto audit_static = [&](const SR& runner)->absl::Status {
    if (!runner) return absl::OkStatus();
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto count, CountStaticInt2RuntimeOperators(runner->runtime()));
    rt->static_int2_operators += count;
    return absl::OkStatus();
  };
  for (const auto* signature : {rt->prefill.get(),rt->decode.get()}) {
    LRT_TENSOR_RETURN_IF_ERROR(audit_static(signature->preprocess));
    LRT_TENSOR_RETURN_IF_ERROR(audit_static(signature->tail));
    for (const auto& layer : signature->stages) {
      LRT_TENSOR_RETURN_IF_ERROR(audit_static(layer.projection));
      LRT_TENSOR_RETURN_IF_ERROR(audit_static(layer.attention.runner));
      LRT_TENSOR_RETURN_IF_ERROR(audit_static(layer.post));
    }
  }
  const size_t expected_qc2=absl::GetFlag(FLAGS_preserve_static_int2)?60:0;
  if (rt->static_int2_tensors!=expected_qc2 || rt->static_int2_operators!=expected_qc2)
    return absl::InternalError("Expected all 60 original INT2 MLP tensors/operators or none");
  std::cerr << "Static INT2 MLP tensors=" << rt->static_int2_tensors << " operators=" << rt->static_int2_operators << " bytes=" << rt->static_int2_compact_bytes << "\n";
  LRT_TENSOR_RETURN_IF_ERROR(xnn_finalize_weights_cache(raw,xnn_weights_cache_finalization_kind_hard));
  rt->times.total_ms=rt->times.compile_pack_ms=Milliseconds(start);
  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_pack", rt.get(), &loaded));
  return rt;
}
absl::Status TraceStage(StageRunner& runner,const TensorHandle& handle,const std::string& tag,
    int rows,int row,int heads=1) {
  if(row<0||row>=rows)return absl::OkStatus();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto data,runner.ReadOutput(handle));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto index,runner.graph().Lookup(handle));
  const auto& info=runner.graph().values()[index].info;
  size_t elements=1;for(auto d:info.shape)elements*=d;
  size_t element_bytes=handle.GetType()==Type::kI8?1:4;
  size_t dim=elements/(rows*heads);
  const auto file=absl::GetFlag(FLAGS_output_dir)+"/trace."+std::to_string(absl::GetFlag(FLAGS_trace_position))+"."+tag;
  std::ofstream out(file,std::ios::binary);
  for(int h=0;h<heads;++h)out.write(reinterpret_cast<const char*>(data.data())+(size_t(h)*rows+row)*dim*element_bytes,dim*element_bytes);
  return out?absl::OkStatus():absl::InternalError("Trace write failed");
}
struct StageTimes {double forward=0,attention=0,append=0,projection=0,post=0;int chunks=0;};
absl::Status RunLiveTokens(LiveSignature& sig,const Config& config,const LoadedTensors& loaded,
    ActiveKvBank& bank,absl::Span<const int32_t> tokens,StageTimes& times) {
  const int rows=sig.rows,start=bank.length(),count=tokens.size(),end=start+count;
  if(count<1||count>rows||(sig.decode&&count!=1))return absl::InvalidArgumentError("Invalid active signature token count");
  LRT_TENSOR_RETURN_IF_ERROR(bank.BeginAppend(count));
  struct Transaction {ActiveKvBank& bank;~Transaction(){if(bank.append_in_flight())bank.Abort();}} transaction{bank};
  std::vector<int32_t> padded(rows,0);std::copy(tokens.begin(),tokens.end(),padded.begin());
  std::vector<float> embeddings(size_t(rows)*config.embed_dim);
  std::vector<std::vector<float>> ple(config.num_layers,std::vector<float>(size_t(rows)*config.per_layer_input_dim));
  LRT_TENSOR_RETURN_IF_ERROR(loaded.token_embedding->Lookup(padded,absl::MakeSpan(embeddings)));
  LRT_TENSOR_RETURN_IF_ERROR(loaded.emb_per_layer_table->LookupPerLayer(padded,config.num_layers,config.per_layer_input_dim,absl::MakeSpan(ple)));
  LRT_TENSOR_RETURN_IF_ERROR(sig.preprocess->SetInput(sig.inputs.embedded_input,embeddings));
  for(int i=0;i<config.num_layers;++i)LRT_TENSOR_RETURN_IF_ERROR(BindLive(*sig.preprocess,sig.inputs.per_layer_token_embeddings[i],ple[i]));
  auto at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(sig.preprocess->Run());times.forward+=Milliseconds(at);
  const int trace_row=absl::GetFlag(FLAGS_trace_position)-start;
  std::vector<float> positions(rows);for(int r=0;r<rows;++r)positions[r]=start+r;
  for(int i=0;i<=sig.last_layer;++i) {
    auto& cut=sig.cuts.layers[i];auto& stage=sig.stages[i];
    auto& prev=i?*sig.stages[i-1].post:*sig.preprocess;
    const auto& hidden=i?sig.cuts.layers[i-1].output:sig.cuts.initial_hidden;
    LRT_TENSOR_RETURN_IF_ERROR(BindFrom(*stage.projection,cut.hidden_input,prev,hidden));
    LRT_TENSOR_RETURN_IF_ERROR(BindLive(*stage.projection,sig.inputs.weights.at("matched.rope.positions"),positions));
    at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(stage.projection->Run());
    double elapsed=Milliseconds(at);times.forward+=elapsed;times.projection+=elapsed;
    if(trace_row>=0&&trace_row<count) {
      if(sig.decode||i<sig.last_layer)LRT_TENSOR_RETURN_IF_ERROR(TraceStage(*stage.projection,cut.query,absl::StrCat("layer",i,".q.f32"),rows,trace_row,config.num_heads));
      if(i==cut.owner) {
        LRT_TENSOR_RETURN_IF_ERROR(TraceStage(*stage.projection,cut.new_key,absl::StrCat("layer",i,".k.i8"),rows,trace_row));
        LRT_TENSOR_RETURN_IF_ERROR(TraceStage(*stage.projection,cut.new_value,absl::StrCat("layer",i,".v.i8"),rows,trace_row));
      }
    }
    if(i==cut.owner) {
      at=BenchClock::now();
      LRT_TENSOR_ASSIGN_OR_RETURN(auto k,stage.projection->ReadOutputAs<int8_t>(cut.new_key));
      LRT_TENSOR_ASSIGN_OR_RETURN(auto v,stage.projection->ReadOutputAs<int8_t>(cut.new_value));
      LRT_TENSOR_RETURN_IF_ERROR(bank.Append(i,{k.data(),size_t(count)*cut.dim},{v.data(),size_t(count)*cut.dim}));
      times.append+=Milliseconds(at);
    }
    if(!sig.decode&&i==sig.last_layer)break;
    auto& attention=stage.attention;
    int begin=cut.global?0:std::max(0,start-config.sliding_window_size+1);
    const int alignment=absl::GetFlag(FLAGS_kv_alignment);
    begin=begin/alignment*alignment;
    int padded_end=(end+alignment-1)/alignment*alignment;
    int extent=padded_end-begin;
    if(absl::GetFlag(FLAGS_fixed_attention_extent))extent=cut.global?bank.capacity():config.sliding_window_size+rows-1;
    LRT_TENSOR_RETURN_IF_ERROR(attention.runner->ReshapeInput(attention.k,Shape{1,1,extent,cut.dim}));
    LRT_TENSOR_RETURN_IF_ERROR(attention.runner->ReshapeInput(attention.v,Shape{1,1,extent,cut.dim}));
    LRT_TENSOR_RETURN_IF_ERROR(attention.runner->ReshapeInput(attention.mask,Shape{1,1,config.num_heads*rows,extent}));
    LRT_TENSOR_RETURN_IF_ERROR(BindFrom(*attention.runner,attention.q,*stage.projection,cut.query));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto kview,bank.PaddedKeys(cut.owner,begin,end,padded_end));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto vview,bank.PaddedValues(cut.owner,begin,end,padded_end));
    std::vector<int8_t> fixed_k,fixed_v;
    if(absl::GetFlag(FLAGS_fixed_attention_extent)) {
      fixed_k.resize(size_t(extent)*cut.dim);fixed_v.resize(fixed_k.size());
      std::copy(kview.begin(),kview.end(),fixed_k.begin());std::copy(vview.begin(),vview.end(),fixed_v.begin());
      kview=absl::MakeConstSpan(fixed_k);vview=absl::MakeConstSpan(fixed_v);
    }
    LRT_TENSOR_RETURN_IF_ERROR(attention.runner->SetInput(attention.k,kview));
    LRT_TENSOR_RETURN_IF_ERROR(attention.runner->SetInput(attention.v,vview));
    std::vector<float> mask(size_t(config.num_heads)*rows*extent,std::numeric_limits<float>::lowest());
    for(int h=0;h<config.num_heads;++h)for(int r=0;r<count;++r) {
      int lower=cut.global?0:std::max(0,start+r-config.sliding_window_size+1);
      for(int p=std::max(lower,begin);p<=start+r;++p)mask[(size_t(h)*rows+r)*extent+p-begin]=0;
    }
    LRT_TENSOR_RETURN_IF_ERROR(attention.runner->SetInput(attention.mask,mask));
    at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(attention.runner->Run());
    elapsed=Milliseconds(at);times.forward+=elapsed;times.attention+=elapsed;
    if(absl::GetFlag(FLAGS_dump_full_logits))LRT_TENSOR_RETURN_IF_ERROR(AuditActiveAttention(attention.runner->runtime(),config.num_heads*rows,extent,cut.dim));
    if(trace_row>=0&&trace_row<count)LRT_TENSOR_RETURN_IF_ERROR(TraceStage(*attention.runner,attention.output,absl::StrCat("layer",i,".attention.f32"),rows,trace_row,config.num_heads));
    LRT_TENSOR_RETURN_IF_ERROR(BindFrom(*stage.post,cut.hidden_input,prev,hidden));
    LRT_TENSOR_RETURN_IF_ERROR(BindFrom(*stage.post,cut.ple_input,*sig.preprocess,sig.cuts.ple_outputs[i]));
    LRT_TENSOR_RETURN_IF_ERROR(BindFrom(*stage.post,cut.context_input,*attention.runner,attention.output));
    at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(stage.post->Run());
    elapsed=Milliseconds(at);times.forward+=elapsed;times.post+=elapsed;
    if(trace_row>=0&&trace_row<count)LRT_TENSOR_RETURN_IF_ERROR(TraceStage(*stage.post,cut.output,absl::StrCat("layer",i,".hidden.f32"),rows,trace_row));
  }
  if(sig.decode) {
    LRT_TENSOR_RETURN_IF_ERROR(BindFrom(*sig.tail,sig.cuts.final_hidden,*sig.stages.back().post,sig.cuts.layers.back().output));
    at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(sig.tail->Run());times.forward+=Milliseconds(at);
  }
  LRT_TENSOR_RETURN_IF_ERROR(bank.Commit());++times.chunks;
  return absl::OkStatus();
}
struct PassResult {int index,input_token,context_length,argmax;double elapsed_ms;StageTimes times;std::string logits_file;};
absl::Status WriteLiveRun(const BenchCase& item,int run_index,bool warmup,int threads,bool reused,
    bool compiled,const LiveRuntime& rt,double load_ms,const std::vector<PassResult>& passes,const std::string& dir) {
  std::string file=dir+"/"+item.id+(warmup?".warmup_":".run_")+absl::StrFormat("%03d.json",run_index);
  std::ofstream out(file);out<<std::setprecision(17);
  out<<"{\"schema_version\":1,\"runner\":\"xnnpack_tensor_live_int8\",\"case_id\":\""<<item.id
     <<"\",\"prompt_token_ids\":["<<absl::StrJoin(item.prompt,",")<<"],\"forced_decode_token_ids\":["<<absl::StrJoin(item.forced,",")
     <<"],\"vocab_size\":262144,\"logits_dtype\":\"float32-little-endian\",\"run_index\":"<<run_index<<",\"warmup\":"<<(warmup?"true":"false")<<",\"num_threads\":"<<threads
     <<",\"reuse_runtimes\":"<<(reused?"true":"false")<<",\"compiled_this_run\":"<<(compiled?"true":"false")
     <<",\"setup\":{\"total_ms\":"<<rt.times.total_ms<<"},\"model_load_prepare_ms\":"<<load_ms
     <<",\"prefill_rows\":"<<rt.prefill->rows<<",\"kv_capacity\":"<<absl::GetFlag(FLAGS_cache_capacity)
     <<",\"kv_alignment\":"<<absl::GetFlag(FLAGS_kv_alignment)<<",\"fixed_attention_extent\":"<<(absl::GetFlag(FLAGS_fixed_attention_extent)?"true":"false")
     <<",\"static_int2_tensor_count\":"<<rt.static_int2_tensors<<",\"static_int2_operator_count\":"<<rt.static_int2_operators
     <<",\"static_int2_compact_bytes\":"<<rt.static_int2_compact_bytes
     <<",\"share_workspace\":"<<(absl::GetFlag(FLAGS_share_workspace)?"true":"false")
     <<",\"preserve_static_int2\":"<<(absl::GetFlag(FLAGS_preserve_static_int2)?"true":"false")
     <<",\"memory_report\":"<<(absl::GetFlag(FLAGS_memory_report)?"true":"false")<<",\"timings_valid_for_benchmark\":"<<(absl::GetFlag(FLAGS_memory_report)?"false":"true")
     <<",\"kv_dtype\":\"int8\",\"kv_layout\":\"owner_position_channel\",\"peak_process_rss_kib\":"<<PeakRssKiB()<<",\"passes\":[";
  for(size_t i=0;i<passes.size();++i){const auto&p=passes[i];if(i)out<<",";
    out<<"{\"kind\":\""<<(i?"decode":"prefill")<<"\",\"decode_index\":"<<p.index<<",\"input_ids\":[";
    if(i)out<<p.input_token;else out<<absl::StrJoin(item.prompt,",");
    out<<"],\"context_length_after\":"<<p.context_length<<",\"logits_position\":"<<p.context_length-1
       <<",\"argmax_id\":"<<p.argmax<<",\"elapsed_ms\":"<<p.elapsed_ms<<",\"forward_ms\":"<<p.times.forward
       <<",\"cache_handoff_or_update_ms\":"<<p.times.append<<",\"attention_run_ms\":"<<p.times.attention
       <<",\"projection_run_ms\":"<<p.times.projection<<",\"post_attention_run_ms\":"<<p.times.post
       <<",\"invocations\":"<<p.times.chunks<<",\"logits_file\":";
    if(p.logits_file.empty())out<<"null";else out<<"\""<<p.logits_file<<"\"";out<<"}";
  }
  out<<"]}\n";return out?absl::OkStatus():absl::InternalError("Write run failed");
}
absl::Status RunLiveCase(LiveRuntime& rt,const Config& config,const LoadedTensors& loaded,
    const std::vector<KvOwnerSpec>& specs,const BenchCase& item,int index,bool warmup,int threads,
    bool reused,bool compiled,bool dump,double load_ms,const std::string& dir) {
  auto start=BenchClock::now();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bank,ActiveKvBank::Create(specs,absl::GetFlag(FLAGS_cache_capacity)));
  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_kv_allocation", &rt, &loaded, item.id, index, size_t(absl::GetFlag(FLAGS_cache_capacity))*9216));
  StageTimes times;
  for(size_t offset=0;offset+1<item.prompt.size();) {
    int count=std::min<size_t>(rt.prefill->rows,item.prompt.size()-1-offset);
    LRT_TENSOR_RETURN_IF_ERROR(RunLiveTokens(*rt.prefill,config,loaded,bank,{item.prompt.data()+offset,size_t(count)},times));offset+=count;
  }
  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_prefill_prefix", &rt, &loaded, item.id, index, size_t(absl::GetFlag(FLAGS_cache_capacity))*9216));
  std::vector<PassResult> passes;
  for(size_t i=0;i<=item.forced.size();++i) {
    if(i){start=BenchClock::now();times={};}
    int32_t token=i?item.forced[i-1]:item.prompt.back();
    LRT_TENSOR_RETURN_IF_ERROR(RunLiveTokens(*rt.decode,config,loaded,bank,{&token,1},times));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto data,rt.decode->tail->ReadOutputAs<float>(rt.decode->outputs.logits));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto pred,SelectTokenAndDump({data.data(),data.size()},"",i));
    PassResult pass{int(i),i?token:-1,bank.length(),pred,Milliseconds(start),times,""};
    if(dump) {
      pass.logits_file=item.id+(i?absl::StrFormat(".decode_%04d.f32",i):".prefill.f32");
      LRT_TENSOR_RETURN_IF_ERROR(WriteRawFloats(dir+"/"+pass.logits_file,{data.data(),data.size()}));
      if(absl::GetFlag(FLAGS_dump_cache))for(auto&s:specs) {
        LRT_TENSOR_ASSIGN_OR_RETURN(auto k,bank.Keys(s.owner,0,bank.length()));
        LRT_TENSOR_ASSIGN_OR_RETURN(auto v,bank.Values(s.owner,0,bank.length()));
        for(auto kv:{std::make_pair("k",k),std::make_pair("v",v)}) {
          std::ofstream f(dir+"/"+pass.logits_file+absl::StrFormat(".owner%d.",s.owner)+kv.first+".i8",std::ios::binary);
          f.write(reinterpret_cast<const char*>(kv.second.data()),kv.second.size());if(!f)return absl::InternalError("Cache dump failed");
        }
      }
    }
    passes.push_back(std::move(pass));
    if (i == 0 || i == 1 || i == item.forced.size())
      LRT_TENSOR_RETURN_IF_ERROR(RecordMemory(i == 0 ? "after_first_logits" : (i == 1 ? "after_first_forced_decode" : "after_last_decode"), &rt, &loaded, item.id, index, size_t(absl::GetFlag(FLAGS_cache_capacity))*9216));
  }
  return WriteLiveRun(item,index,warmup,threads,reused,compiled,rt,load_ms,passes,dir);
}
absl::Status LiveMain() {
  int threads=absl::GetFlag(FLAGS_num_threads),chunk=absl::GetFlag(FLAGS_prefill_chunk_rows),capacity=absl::GetFlag(FLAGS_cache_capacity);
  int warmups=absl::GetFlag(FLAGS_warmup_runs),runs=absl::GetFlag(FLAGS_measured_runs);
  bool reuse=absl::GetFlag(FLAGS_reuse_runtimes);
  int alignment=absl::GetFlag(FLAGS_kv_alignment);
  if(absl::GetFlag(FLAGS_fixed_attention_extent)&&alignment!=1)return absl::InvalidArgumentError("Fixed-width diagnostic requires alignment1");
  if(alignment<1||alignment>128||(alignment&(alignment-1))||capacity%alignment)return absl::InvalidArgumentError("Alignment must be power of two 1..128 dividing capacity");
  if(threads<1||warmups<0||runs<1||(chunk!=128&&chunk!=1024)||capacity<1||capacity>8448)return absl::InvalidArgumentError("Invalid live-runner configuration");
  if(absl::GetFlag(FLAGS_consistent_arithmetic)||!absl::GetFlag(FLAGS_weight_cache)||absl::GetFlag(FLAGS_dump_intermediates)||!absl::GetFlag(FLAGS_dump_logits).empty())return absl::InvalidArgumentError("Use default arithmetic/cache and live-runner trace flags");
  LRT_TENSOR_ASSIGN_OR_RETURN(auto cases,ReadCases(absl::GetFlag(FLAGS_cases_file)));
  for(auto&c:cases)if(c.prompt.size()+c.forced.size()>capacity)return absl::InvalidArgumentError("Case exceeds KV capacity");
  const std::string dir=absl::GetFlag(FLAGS_output_dir);std::error_code error;
  if(dir.empty()||!std::filesystem::create_directory(dir,error))return absl::InvalidArgumentError("Output directory must be new: "+error.message());
  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("before_load"));
  auto start=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(xnn_initialize(nullptr));
  const auto config=Config::From(ModelVariant::kE2B);
  SetPreserveStaticInt2Weights(absl::GetFlag(FLAGS_preserve_static_int2));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto loaded,LoadPublishedBundle(absl::GetFlag(FLAGS_bundle_dir),config));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto specs,LoadPublishedKvSpecs(absl::GetFlag(FLAGS_bundle_dir),config));
  double load_ms=Milliseconds(start);
  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_load", nullptr, &loaded));
  std::unique_ptr<LiveRuntime> rt;
  for(const auto& c:cases)for(int i=0;i<warmups+runs;++i) {
    bool compiled=!rt;
    if(!rt){LRT_TENSOR_ASSIGN_OR_RETURN(rt,CreateLiveRuntime(config,loaded,specs,threads,chunk,capacity));}
    bool warmup=i<warmups;int index=warmup?i:i-warmups;
    LRT_TENSOR_RETURN_IF_ERROR(RunLiveCase(*rt,config,loaded,specs,c,index,warmup,threads,reuse,compiled,
      absl::GetFlag(FLAGS_dump_full_logits)&&!warmup&&index==0,load_ms,dir));
    std::cerr<<"Completed "<<c.id<<" iteration "<<i<<"\n";
    LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_session", rt.get(), &loaded, c.id, index));
    if(!reuse)rt.reset();
  }
  rt.reset();
  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_runtime_destroy", nullptr, &loaded));
  std::ofstream f(dir+"/run.json");f<<"{\"status\":\"completed\",\"cases\":"<<cases.size()<<",\"peak_process_rss_kib\":"<<PeakRssKiB()<<"}\n";
  return f?absl::OkStatus():absl::InternalError("Cannot write summary");
}
} }
int main(int argc,char**argv) {
  absl::SetFlag(&FLAGS_num_threads,2);absl::ParseCommandLine(argc,argv);absl::InitializeLog();
  auto status=litert::tensor::examples::gemma4::native::LiveMain();if(!status.ok()){std::cerr<<status<<"\n";return 1;}return 0;
}
