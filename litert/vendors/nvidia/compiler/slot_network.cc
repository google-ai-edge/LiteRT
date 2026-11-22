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

#include "litert/vendors/nvidia/compiler/slot_network.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/nvidia/compiler/slot_plugins.h"
#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "NvInfer.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace litert::nvidia {
namespace {

Error Fail(const std::string& what) {
  return Error(kLiteRtStatusErrorCompilation,
               "Batched decode network: " + what);
}

struct PluginDeleter {
  void operator()(nvinfer1::IPluginV3* plugin) const { delete plugin; }
};

class SlotNetworkBuilder {
 public:
  SlotNetworkBuilder(nvinfer1::INetworkDefinition& step,
                     nvinfer1::INetworkDefinition& out, int32_t slots,
                     std::vector<std::unique_ptr<nvinfer1::IPluginV3>>* plugins)
      : step_(step), out_(out), slots_(slots), plugins_(plugins) {}

  Expected<int32_t> Build() {
    for (int32_t i = 0; i < step_.getNbInputs(); ++i) {
      nvinfer1::ITensor* input = step_.getInput(i);
      for (int32_t s = 0; s < slots_; ++s) {
        nvinfer1::ITensor* copy =
            out_.addInput(SlotTensorName(input->getName(), s).c_str(),
                          input->getType(), input->getDimensions());
        if (copy == nullptr) return Fail("an input");
        map_[Key(input, s)] = copy;
      }
    }
    // In the order of their dependencies: a layer's input may have been
    // rewired to a tensor of a layer added after it.
    LITERT_ASSIGN_OR_RETURN(const std::vector<int32_t> order, LayerOrder());
    for (int32_t l : order) {
      LITERT_RETURN_IF_ERROR(CloneLayer(*step_.getLayer(l)));
    }
    for (int32_t i = 0; i < step_.getNbOutputs(); ++i) {
      nvinfer1::ITensor* output = step_.getOutput(i);
      for (int32_t s = 0; s < slots_; ++s) {
        LITERT_ASSIGN_OR_RETURN(nvinfer1::ITensor * copy, Map(output, s));
        if (shared_.count(copy) != 0) {
          // A constant output (never in a decode step) would be one tensor
          // for every slot.
          return Fail("an output that is a constant");
        }
        copy->setName(SlotTensorName(output->getName(), s).c_str());
        out_.markOutput(*copy);
      }
    }
    return merged_;
  }

 private:
  static uint64_t Key(const nvinfer1::ITensor* tensor, int32_t slot) {
    return (reinterpret_cast<uint64_t>(tensor) << 4) |
           static_cast<uint64_t>(slot);
  }

  Expected<nvinfer1::ITensor*> Map(const nvinfer1::ITensor* tensor,
                                   int32_t slot) {
    if (auto it = shared_tensor_.find(tensor); it != shared_tensor_.end()) {
      return it->second;
    }
    auto it = map_.find(Key(tensor, slot));
    if (it == map_.end()) {
      return Fail(std::string("tensor ") + tensor->getName() +
                  " is made by no layer before it is used");
    }
    return it->second;
  }

  // The step's layers, each after the layers that make its inputs.
  Expected<std::vector<int32_t>> LayerOrder() {
    const int32_t count = step_.getNbLayers();
    std::unordered_map<const nvinfer1::ITensor*, int32_t> producer;
    for (int32_t l = 0; l < count; ++l) {
      nvinfer1::ILayer* layer = step_.getLayer(l);
      for (int32_t o = 0; o < layer->getNbOutputs(); ++o) {
        producer[layer->getOutput(o)] = l;
      }
    }
    std::vector<int32_t> order;
    order.reserve(count);
    std::vector<uint8_t> state(count, 0);  // 1 visiting, 2 done
    for (int32_t root = 0; root < count; ++root) {
      if (state[root] != 0) continue;
      // Iterative depth-first walk: (layer, next input to look at).
      std::vector<std::pair<int32_t, int32_t>> stack{{root, 0}};
      state[root] = 1;
      while (!stack.empty()) {
        auto& [l, next] = stack.back();
        nvinfer1::ILayer* layer = step_.getLayer(l);
        if (next < layer->getNbInputs()) {
          nvinfer1::ITensor* input = layer->getInput(next++);
          if (input == nullptr) continue;
          auto it = producer.find(input);
          if (it == producer.end()) continue;  // a network input
          if (state[it->second] == 1) return Fail("a cycle of layers");
          if (state[it->second] == 0) {
            state[it->second] = 1;
            stack.push_back({it->second, 0});
          }
          continue;
        }
        state[l] = 2;
        order.push_back(l);
        stack.pop_back();
      }
    }
    return order;
  }

  // A layer's mapped inputs for one slot: all of them, nulls kept.
  Expected<std::vector<nvinfer1::ITensor*>> Inputs(nvinfer1::ILayer& layer,
                                                   int32_t slot) {
    std::vector<nvinfer1::ITensor*> inputs(layer.getNbInputs(), nullptr);
    for (int32_t i = 0; i < layer.getNbInputs(); ++i) {
      if (layer.getInput(i) == nullptr) continue;
      LITERT_ASSIGN_OR_RETURN(inputs[i], Map(layer.getInput(i), slot));
    }
    return inputs;
  }

  // The layer's optional inputs past `first` and its settings, onto `copy`.
  void Finish(nvinfer1::ILayer& layer, nvinfer1::ILayer& copy,
              const std::vector<nvinfer1::ITensor*>& inputs, int32_t first,
              int32_t slot) {
    for (int32_t i = first; i < static_cast<int32_t>(inputs.size()); ++i) {
      if (inputs[i] != nullptr) copy.setInput(i, *inputs[i]);
    }
    copy.setName((std::string(layer.getName()) + "#slot" + std::to_string(slot))
                     .c_str());
  }

  void Record(nvinfer1::ILayer& layer, nvinfer1::ILayer& copy, int32_t slot) {
    for (int32_t o = 0; o < layer.getNbOutputs(); ++o) {
      map_[Key(layer.getOutput(o), slot)] = copy.getOutput(o);
    }
  }

  Expected<void> CloneLayer(nvinfer1::ILayer& layer) {
    if (layer.getType() == nvinfer1::LayerType::kCONSTANT) {
      // One constant for every slot, under its own name: a weight the plan
      // leaves out (a holder of the weight store, a refittable weight) is
      // found by that name.
      auto& constant = static_cast<nvinfer1::IConstantLayer&>(layer);
      // A refittable weight of the step is a placeholder the plan leaves out
      // (a holder of the weight store): it was added without values, and
      // what the step's layer returns for them after its build is no host
      // memory (TensorRT-RTX read hundreds of MB from there and faulted).
      const bool placeholder =
          step_.areWeightsMarkedRefittable(layer.getName());
      nvinfer1::Weights weights = constant.getWeights();
      if (placeholder) weights.values = nullptr;
      nvinfer1::IConstantLayer* copy =
          out_.addConstant(constant.getDimensions(), weights);
      if (copy == nullptr) return Fail("a constant");
      copy->setName(layer.getName());
      // The step names a constant's weights as its layer when it names them
      // at all (the weight holders); the name outlives this network's build.
      if (placeholder) {
        if (!out_.setWeightsName(copy->getWeights(), layer.getName()) ||
            !out_.markWeightsRefittable(layer.getName())) {
          return Fail(std::string("weights ") + layer.getName());
        }
      }
      shared_tensor_[layer.getOutput(0)] = copy->getOutput(0);
      shared_.insert({copy->getOutput(0), true});
      return {};
    }
    if (layer.getType() == nvinfer1::LayerType::kPLUGIN_V3) {
      LITERT_ASSIGN_OR_RETURN(bool twinned, AddSlotsTwin(layer));
      if (twinned) return {};
      LITERT_ASSIGN_OR_RETURN(bool merged, MergeProjection(layer));
      if (merged) return {};
    }
    if (layer.getType() == nvinfer1::LayerType::kKVCACHE_UPDATE) {
      LITERT_ASSIGN_OR_RETURN(bool written, AddCacheWrites(layer));
      if (written) return {};
    }
    if (OnlySharedInputs(layer)) {
      // Computed from the weights alone (a dequantization of a constant):
      // once for every slot.
      LITERT_RETURN_IF_ERROR(CloneForSlot(layer, 0));
      for (int32_t o = 0; o < layer.getNbOutputs(); ++o) {
        nvinfer1::ITensor* copy = map_[Key(layer.getOutput(o), 0)];
        shared_tensor_[layer.getOutput(o)] = copy;
        shared_.insert({copy, true});
      }
      return {};
    }
    for (int32_t s = 0; s < slots_; ++s) {
      LITERT_RETURN_IF_ERROR(CloneForSlot(layer, s));
    }
    return {};
  }

  // Whether every input of a layer other than a plugin or a cache update is
  // one every slot shares.
  bool OnlySharedInputs(nvinfer1::ILayer& layer) const {
    if (layer.getType() == nvinfer1::LayerType::kPLUGIN_V3 ||
        layer.getType() == nvinfer1::LayerType::kKVCACHE_UPDATE) {
      return false;
    }
    for (int32_t i = 0; i < layer.getNbInputs(); ++i) {
      const nvinfer1::ITensor* input = layer.getInput(i);
      if (input != nullptr && shared_tensor_.count(input) == 0) return false;
    }
    return true;
  }

  // A blockwise INT4 GEMV whose weights every slot shares: one plugin over
  // the slots' activation rows, concatenated on the rows axis.
  // A plugin layer with a state per slot (decode attention, QkvNormRope, an
  // INT8 cache write) as one launch for every slot (slot_plugins.h's twin):
  // its inputs slot-major, slot s's output o its output s * m + o.
  Expected<bool> AddSlotsTwin(nvinfer1::ILayer& layer) {
    auto& plugin_layer = static_cast<nvinfer1::IPluginV3Layer&>(layer);
    std::unique_ptr<nvinfer1::IPluginV3, PluginDeleter> twin(
        CreateSlotsTwin(plugin_layer, slots_));
    if (twin == nullptr) return false;
    const int32_t n = layer.getNbInputs();
    const int32_t m = layer.getNbOutputs();
    std::vector<nvinfer1::ITensor*> inputs(static_cast<size_t>(n) * slots_);
    for (int32_t s = 0; s < slots_; ++s) {
      for (int32_t i = 0; i < n; ++i) {
        if (layer.getInput(i) == nullptr) return false;
        LITERT_ASSIGN_OR_RETURN(inputs[s * n + i], Map(layer.getInput(i), s));
      }
    }
    nvinfer1::IPluginV3Layer* added = tensorrt_rtx_1_5_0_99::AddPluginV3(
        out_, inputs.data(), static_cast<int32_t>(inputs.size()), *twin);
    Keep(std::move(twin));
    if (added == nullptr || added->getNbOutputs() != m * slots_) {
      return Fail(std::string("the slots twin of ") + layer.getName());
    }
    added->setName((std::string(layer.getName()) + "#slots").c_str());
    for (int32_t s = 0; s < slots_; ++s) {
      for (int32_t o = 0; o < m; ++o) {
        map_[Key(layer.getOutput(o), s)] = added->getOutput(s * m + o);
      }
    }
    ++twinned_;
    return true;
  }

  // A cache update of an FP16 cache the network takes and gives as itself (a
  // network input updated into a network output) as one write launch for
  // every slot (CreateCacheWriteSlotsPlugin), in place where the dispatch
  // binds the output to the input's buffer.
  Expected<bool> AddCacheWrites(nvinfer1::ILayer& layer) {
    if (layer.getNbInputs() != 3 || layer.getNbOutputs() != 1) return false;
    nvinfer1::ITensor* cache = layer.getInput(0);
    nvinfer1::ITensor* rows = layer.getInput(1);
    nvinfer1::ITensor* index = layer.getInput(2);
    nvinfer1::ITensor* out = layer.getOutput(0);
    if (cache == nullptr || rows == nullptr || index == nullptr ||
        !cache->isNetworkInput() || !out->isNetworkOutput() ||
        cache->getType() != nvinfer1::DataType::kHALF ||
        (rows->getType() != nvinfer1::DataType::kHALF &&
         rows->getType() != nvinfer1::DataType::kBF16 &&
         rows->getType() != nvinfer1::DataType::kFLOAT) ||
        index->getType() != nvinfer1::DataType::kINT32) {
      return false;
    }
    const nvinfer1::Dims c = cache->getDimensions();
    const nvinfer1::Dims r = rows->getDimensions();
    const nvinfer1::Dims x = index->getDimensions();
    if (c.nbDims != 4 || r.nbDims != 4 || c.d[0] != 1 || r.d[0] != 1 ||
        r.d[1] != c.d[1] || r.d[3] != c.d[3] || r.d[2] > c.d[2] ||
        x.nbDims != 1 || x.d[0] != 1) {
      return false;
    }
    std::unique_ptr<nvinfer1::IPluginV3, PluginDeleter> plugin(
        CreateCacheWriteSlotsPlugin(slots_, /*columns=*/false));
    if (plugin == nullptr) return false;
    std::vector<nvinfer1::ITensor*> inputs(3 * static_cast<size_t>(slots_));
    for (int32_t s = 0; s < slots_; ++s) {
      LITERT_ASSIGN_OR_RETURN(inputs[3 * s], Map(cache, s));
      LITERT_ASSIGN_OR_RETURN(inputs[3 * s + 1], Map(rows, s));
      LITERT_ASSIGN_OR_RETURN(inputs[3 * s + 2], Map(index, s));
    }
    nvinfer1::IPluginV3Layer* added = tensorrt_rtx_1_5_0_99::AddPluginV3(
        out_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    Keep(std::move(plugin));
    if (added == nullptr || added->getNbOutputs() != slots_) {
      return Fail(std::string("the slots' cache writes of ") + layer.getName());
    }
    added->setName((std::string(layer.getName()) + "#slots").c_str());
    for (int32_t s = 0; s < slots_; ++s) {
      map_[Key(out, s)] = added->getOutput(s);
    }
    ++cache_writes_;
    return true;
  }

  // A row-wise plugin whose only per-slot input is its activation (input 0)
  // runs once over the rows of several slots, its weights read once for
  // them; its output is sliced back per slot.
  //  - One row per slot (a decode step): every slot in one launch of the
  //    plugin's slot-rows form (each row's bits those of a row alone).
  //  - A slot's rows (an MTP verify step's T rows): whole verifies as units,
  //    as many slots per launch as fit the GEMV's row limit, through the very
  //    plugin of the single verify: its kernel at T rows or at a multiple of
  //    T gives each row the same bits (the per-channel kernel keeps a row's
  //    sums in order at any count; on the tensor cores each row is its own
  //    MMA column). The MoE block's verify path is not merged.
  Expected<bool> MergeProjection(nvinfer1::ILayer& layer) {
    auto& plugin_layer = static_cast<nvinfer1::IPluginV3Layer&>(layer);
    if (layer.getNbOutputs() != 1 || layer.getNbInputs() < 2 ||
        layer.getInput(0) == nullptr) {
      return false;
    }
    for (int32_t i = 1; i < layer.getNbInputs(); ++i) {
      if (layer.getInput(i) == nullptr ||
          shared_tensor_.count(layer.getInput(i)) == 0) {
        return false;  // a per-slot operand: not a shared projection
      }
    }
    const nvinfer1::Dims dims = layer.getInput(0)->getDimensions();
    const nvinfer1::Dims out_dims = layer.getOutput(0)->getDimensions();
    if (dims.nbDims < 2 || out_dims.nbDims != dims.nbDims) return false;
    const int32_t axis = dims.nbDims - 2;
    for (int32_t d = 0; d < axis; ++d) {
      if (dims.d[d] != 1) return false;  // the rows on one axis only
    }
    const int64_t rows = dims.d[axis];
    if (rows < 1 || out_dims.d[axis] != rows) return false;
    std::unique_ptr<nvinfer1::IPluginV3, PluginDeleter> twin(
        SubbyteGemvSlotRowsPlugin(plugin_layer.getPlugin()));
    int32_t per_launch = slots_;
    if (rows == 1) {
      if (twin == nullptr) {
        twin.reset(CreateSlotRowsPlugin(plugin_layer.getPlugin()));
      }
    } else {
      // A GEMV (it has a slot-rows form) runs the verify's own plugin.
      if (twin == nullptr) return false;
      twin.reset(plugin_layer.getPlugin().clone());
      per_launch = static_cast<int32_t>(kLiteRtNvidiaSubbyteGemvMaxRows / rows);
    }
    if (twin == nullptr || per_launch < 2) return false;
    for (int32_t first = 0; first < slots_; first += per_launch) {
      const int32_t count = std::min(per_launch, slots_ - first);
      std::vector<nvinfer1::ITensor*> rows_of_slots(count);
      for (int32_t s = 0; s < count; ++s) {
        LITERT_ASSIGN_OR_RETURN(rows_of_slots[s],
                                Map(layer.getInput(0), first + s));
      }
      std::vector<nvinfer1::ITensor*> inputs(layer.getNbInputs());
      std::string suffix = "#rows" + std::to_string(first);
      if (count == 1) {
        inputs[0] = rows_of_slots[0];
      } else {
        nvinfer1::IConcatenationLayer* concat =
            out_.addConcatenation(rows_of_slots.data(), count);
        if (concat == nullptr) return Fail("the slots' projection rows");
        concat->setAxis(axis);
        concat->setName((std::string(layer.getName()) + suffix).c_str());
        inputs[0] = concat->getOutput(0);
      }
      for (int32_t i = 1; i < layer.getNbInputs(); ++i) {
        inputs[i] = shared_tensor_[layer.getInput(i)];
      }
      // Every launch its own plugin object: the last one is the template's.
      std::unique_ptr<nvinfer1::IPluginV3, PluginDeleter> plugin(
          first + per_launch >= slots_ ? twin.release() : twin->clone());
      if (plugin == nullptr) return Fail("a plugin of slot rows");
      nvinfer1::IPluginV3Layer* merged = tensorrt_rtx_1_5_0_99::AddPluginV3(
          out_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
      Keep(std::move(plugin));
      if (merged == nullptr) return Fail("a slot-rows projection");
      merged->setName(
          (std::string(layer.getName()) + "#slots" + std::to_string(first))
              .c_str());
      for (int32_t s = 0; s < count; ++s) {
        nvinfer1::Dims start{};
        start.nbDims = out_dims.nbDims;
        nvinfer1::Dims stride{};
        stride.nbDims = out_dims.nbDims;
        for (int32_t d = 0; d < out_dims.nbDims; ++d) stride.d[d] = 1;
        start.d[axis] = s * rows;
        nvinfer1::ISliceLayer* slice =
            out_.addSlice(*merged->getOutput(0), start, out_dims, stride);
        if (slice == nullptr) return Fail("a slot's projection rows");
        slice->setName(
            (std::string(layer.getName()) + "#slot" + std::to_string(first + s))
                .c_str());
        map_[Key(layer.getOutput(0), first + s)] = slice->getOutput(0);
      }
    }
    ++merged_;
    return true;
  }

  Expected<void> CloneForSlot(nvinfer1::ILayer& layer, int32_t slot) {
    LITERT_ASSIGN_OR_RETURN(auto in, Inputs(layer, slot));
    const auto need = [&](int32_t i) -> Expected<nvinfer1::ITensor*> {
      if (i >= static_cast<int32_t>(in.size()) || in[i] == nullptr) {
        return Fail(std::string("a missing input of ") + layer.getName());
      }
      return in[i];
    };
    nvinfer1::ILayer* copy = nullptr;
    int32_t first_optional = static_cast<int32_t>(in.size());
    using LT = nvinfer1::LayerType;
    switch (layer.getType()) {
      case LT::kELEMENTWISE: {
        auto& l = static_cast<nvinfer1::IElementWiseLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        copy = out_.addElementWise(*a, *b, l.getOperation());
        break;
      }
      case LT::kSHUFFLE: {
        auto& l = static_cast<nvinfer1::IShuffleLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        auto* c = out_.addShuffle(*a);
        if (c != nullptr) {
          c->setFirstTranspose(l.getFirstTranspose());
          if (in.size() < 2 || in[1] == nullptr) {
            c->setReshapeDimensions(l.getReshapeDimensions());
          }
          c->setSecondTranspose(l.getSecondTranspose());
          c->setZeroIsPlaceholder(l.getZeroIsPlaceholder());
        }
        copy = c;
        first_optional = 1;
        break;
      }
      case LT::kACTIVATION: {
        auto& l = static_cast<nvinfer1::IActivationLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        auto* c = out_.addActivation(*a, l.getActivationType());
        if (c != nullptr) {
          c->setAlpha(l.getAlpha());
          c->setBeta(l.getBeta());
        }
        copy = c;
        break;
      }
      case LT::kSLICE: {
        auto& l = static_cast<nvinfer1::ISliceLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        auto* c = out_.addSlice(*a, l.getStart(), l.getSize(), l.getStride());
        if (c != nullptr) {
          c->setMode(l.getMode());
          if (l.getAxes().nbDims > 0) c->setAxes(l.getAxes());
        }
        copy = c;
        first_optional = 1;
        break;
      }
      case LT::kCONCATENATION: {
        auto& l = static_cast<nvinfer1::IConcatenationLayer&>(layer);
        for (auto* t : in) {
          if (t == nullptr) return Fail("a concatenation input");
        }
        auto* c =
            out_.addConcatenation(in.data(), static_cast<int32_t>(in.size()));
        if (c != nullptr) c->setAxis(l.getAxis());
        copy = c;
        break;
      }
      case LT::kUNARY: {
        auto& l = static_cast<nvinfer1::IUnaryLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        copy = out_.addUnary(*a, l.getOperation());
        break;
      }
      case LT::kREDUCE: {
        auto& l = static_cast<nvinfer1::IReduceLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        copy = out_.addReduce(*a, l.getOperation(), l.getReduceAxes(),
                              l.getKeepDimensions());
        break;
      }
      case LT::kMATRIX_MULTIPLY: {
        auto& l = static_cast<nvinfer1::IMatrixMultiplyLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        copy = out_.addMatrixMultiply(*a, l.getOperation(0), *b,
                                      l.getOperation(1));
        break;
      }
      case LT::kKVCACHE_UPDATE: {
        auto& l = static_cast<nvinfer1::IKVCacheUpdateLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        LITERT_ASSIGN_OR_RETURN(auto* c, need(2));
        copy = out_.addKVCacheUpdate(*a, *b, *c, l.getCacheMode());
        break;
      }
      case LT::kGATHER: {
        auto& l = static_cast<nvinfer1::IGatherLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        auto* c = out_.addGatherV2(*a, *b, l.getMode());
        if (c != nullptr) {
          c->setGatherAxis(l.getGatherAxis());
          c->setNbElementWiseDims(l.getNbElementWiseDims());
        }
        copy = c;
        break;
      }
      case LT::kSELECT: {
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        LITERT_ASSIGN_OR_RETURN(auto* c, need(2));
        copy = out_.addSelect(*a, *b, *c);
        break;
      }
      case LT::kSCATTER: {
        auto& l = static_cast<nvinfer1::IScatterLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        LITERT_ASSIGN_OR_RETURN(auto* c, need(2));
        auto* s = out_.addScatter(*a, *b, *c, l.getMode());
        if (s != nullptr) s->setAxis(l.getAxis());
        copy = s;
        break;
      }
      case LT::kCAST: {
        auto& l = static_cast<nvinfer1::ICastLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        copy = out_.addCast(*a, l.getToType());
        break;
      }
      case LT::kSOFTMAX: {
        auto& l = static_cast<nvinfer1::ISoftMaxLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        auto* c = out_.addSoftMax(*a);
        if (c != nullptr) c->setAxes(l.getAxes());
        copy = c;
        break;
      }
      case LT::kDEQUANTIZE: {
        auto& l = static_cast<nvinfer1::IDequantizeLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        auto* c = out_.addDequantize(*a, *b, l.getToType());
        if (c != nullptr) {
          c->setAxis(l.getAxis());
          const nvinfer1::Dims block = l.getBlockShape();
          if (block.nbDims > 0 && !c->setBlockShape(block)) {
            return Fail("a dequantization's block shape");
          }
        }
        copy = c;
        first_optional = 2;
        break;
      }
      case LT::kQUANTIZE: {
        auto& l = static_cast<nvinfer1::IQuantizeLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        auto* c = out_.addQuantize(*a, *b, l.getToType());
        if (c != nullptr) {
          c->setAxis(l.getAxis());
          const nvinfer1::Dims block = l.getBlockShape();
          if (block.nbDims > 0 && !c->setBlockShape(block)) {
            return Fail("a quantization's block shape");
          }
        }
        copy = c;
        first_optional = 2;
        break;
      }
      case LT::kIDENTITY: {
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        copy = out_.addIdentity(*a);
        break;
      }
      case LT::kONE_HOT: {
        auto& l = static_cast<nvinfer1::IOneHotLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        LITERT_ASSIGN_OR_RETURN(auto* b, need(1));
        LITERT_ASSIGN_OR_RETURN(auto* c, need(2));
        copy = out_.addOneHot(*a, *b, *c, l.getAxis());
        break;
      }
      case LT::kTOPK: {
        auto& l = static_cast<nvinfer1::ITopKLayer&>(layer);
        LITERT_ASSIGN_OR_RETURN(auto* a, need(0));
        copy = out_.addTopK(*a, l.getOperation(), l.getK(), l.getReduceAxes(),
                            l.getIndicesType());
        first_optional = 1;
        break;
      }
      case LT::kFILL: {
        auto& l = static_cast<nvinfer1::IFillLayer&>(layer);
        auto* c =
            out_.addFill(l.getDimensions(), l.getOperation(), l.getToType());
        if (c != nullptr) {
          if (l.isAlphaBetaInt64()) {
            c->setAlphaInt64(l.getAlphaInt64());
            c->setBetaInt64(l.getBetaInt64());
          } else {
            c->setAlpha(l.getAlpha());
            c->setBeta(l.getBeta());
          }
        }
        copy = c;
        first_optional = 0;
        break;
      }
      case LT::kPLUGIN_V3: {
        auto& l = static_cast<nvinfer1::IPluginV3Layer&>(layer);
        std::unique_ptr<nvinfer1::IPluginV3, PluginDeleter> plugin(
            l.getPlugin().clone());
        if (plugin == nullptr) return Fail("a plugin clone");
        for (auto* t : in) {
          if (t == nullptr) return Fail("a plugin input");
        }
        copy = tensorrt_rtx_1_5_0_99::AddPluginV3(
            out_, in.data(), static_cast<int32_t>(in.size()), *plugin);
        Keep(std::move(plugin));
        break;
      }
      default:
        return Fail(std::string("layer ") + layer.getName() + " of type " +
                    std::to_string(static_cast<int>(layer.getType())));
    }
    if (copy == nullptr) {
      return Fail(std::string("a copy of ") + layer.getName());
    }
    Finish(layer, *copy, in, first_optional, slot);
    Record(layer, *copy, slot);
    return {};
  }

  nvinfer1::INetworkDefinition& step_;
  nvinfer1::INetworkDefinition& out_;
  const int32_t slots_;
  // TensorRT-RTX reads a plugin layer's plugin until the network is built:
  // the caller keeps them alive until then.
  void Keep(std::unique_ptr<nvinfer1::IPluginV3, PluginDeleter> plugin) {
    plugins_->push_back(std::unique_ptr<nvinfer1::IPluginV3>(plugin.release()));
  }

  std::vector<std::unique_ptr<nvinfer1::IPluginV3>>* plugins_;
  int32_t merged_ = 0;
  int32_t twinned_ = 0;
  int32_t cache_writes_ = 0;

 public:
  int32_t twinned() const { return twinned_; }
  int32_t cache_writes() const { return cache_writes_; }

 private:
  std::unordered_map<uint64_t, nvinfer1::ITensor*> map_;
  std::unordered_map<const nvinfer1::ITensor*, nvinfer1::ITensor*>
      shared_tensor_;
  std::unordered_map<const nvinfer1::ITensor*, bool> shared_;
};

}  // namespace

int32_t DecodeSlotsRequested() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_SLOTS");
  if (value == nullptr || value[0] == '\0') return 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 2) return 0;
  return parsed > 8 ? 8 : static_cast<int32_t>(parsed);
}

Expected<nvinfer1::INetworkDefinition*> BuildSlotDecodeNetwork(
    nvinfer1::IBuilder& builder, nvinfer1::INetworkDefinition& step,
    int32_t slots, int32_t* merged,
    std::vector<std::unique_ptr<nvinfer1::IPluginV3>>* plugins) {
  if (slots < 2) return Fail("fewer than two slots");
  if (plugins == nullptr) return Fail("no place for the plugins");
  std::unique_ptr<nvinfer1::INetworkDefinition> out(
      builder.createNetworkV2(step.getFlags()));
  if (out == nullptr) return Fail("a network");
  SlotNetworkBuilder slot_builder(step, *out, slots, plugins);
  LITERT_ASSIGN_OR_RETURN(const int32_t count, slot_builder.Build());
  if (merged != nullptr) *merged = count;
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX batched decode network of %d slots: %d "
             "row-wise plugins over the slots' rows, %d slot twins, %d cache "
             "writes in one launch",
             slots, count, slot_builder.twinned(), slot_builder.cache_writes());
  return out.release();
}

}  // namespace litert::nvidia
