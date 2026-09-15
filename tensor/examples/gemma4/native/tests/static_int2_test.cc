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

// Focused correctness tests; no timings, benchmarks, model files or devices.
#include "tensor/examples/gemma4/native/stage_runner.h"
#include "tensor/examples/gemma4/native/model/helpers/mobile_fully_connected.h"
#include <array>
#include <cstring>
#include <iostream>
#include <random>

using namespace litert::tensor;
using namespace litert::tensor::examples::gemma4::native;
using XT = Tensor<XnnpackMixinTag>;
#define REQUIRE(c) do { ++checks; if (!(c)) return absl::InternalError("Check failed: " #c); } while(false)
static size_t checks = 0, vectors = 0, values = 0;
static std::shared_ptr<PerChannelAffineQuantization> Q(std::vector<float> scales) {
  return std::make_shared<PerChannelAffineQuantization>(std::move(scales), std::vector<int64_t>{0}, 0);
}
static absl::Status Test(int columns, int channels, float in_scale, float out_scale) {
  const size_t elements = static_cast<size_t>(columns) * channels;
  std::vector<uint8_t> widened(elements / 2 + XNN_EXTRA_BYTES, 0);
  std::mt19937 random(columns + channels);
  std::vector<int> codes(elements);
  for (size_t i = 0; i < elements; ++i) {
    codes[i] = static_cast<int>(random() % 4) - 2;
    widened[i / 2] |= static_cast<uint8_t>((codes[i] & 15) << (4 * (i % 2)));
  }
  StaticInt2Provenance provenance{"tiny.weight", "INT2", std::string(64, '0'),
                                 10, 0, 1, {channels, columns}, elements / 4};
  LRT_TENSOR_ASSIGN_OR_RETURN(auto compact, StaticInt2WeightBuffer::FromWidenedI4(
      widened.data(), elements / 2, provenance));
  REQUIRE(compact->ByteSize().value() == elements / 4);
  REQUIRE(compact->Lock().size() == elements / 4);
  REQUIRE(compact->LockMutable().size() == 0);
  for (size_t i = 0; i < elements; ++i) {
    int code = (compact->data()[i / 4] >> (2 * (i % 4))) & 3;
    if (code >= 2) code -= 4;
    REQUIRE(code == codes[i]);
  }
  auto bad = provenance; bad.source_dtype = "INT4";
  REQUIRE(!StaticInt2WeightBuffer::FromWidenedI4(widened.data(), elements / 2, bad).ok());
  bad = provenance; bad.source_bytes += 1;
  REQUIRE(!StaticInt2WeightBuffer::FromWidenedI4(widened.data(), elements / 2, bad).ok());
  std::vector<uint8_t> invalid = widened; invalid[0] = 2;
  REQUIRE(!StaticInt2WeightBuffer::FromWidenedI4(invalid.data(), elements / 2, provenance).ok());
  invalid[0] = 13;
  REQUIRE(!StaticInt2WeightBuffer::FromWidenedI4(invalid.data(), elements / 2, provenance).ok());
  std::vector<float> scales(channels);
  for (int i = 0; i < channels; ++i) scales[i] = 0.01f + float(i % 17) / 250.0f;
  XT w2({.name="tiny.weight", .type=Type::kI2, .shape={channels,columns},
         .buffer=compact, .quantization=Q(scales)});
  XT w4({.name="tiny.weight", .type=Type::kI4, .shape={channels,columns},
         .buffer=std::make_shared<SpanCpuBuffer>(widened), .quantization=Q(scales)});
  XT input({.name="input", .type=Type::kFP32, .shape={1,1,columns}});
  absl::flat_hash_map<std::string, XT> weights;
  weights.emplace("tiny.input_scale", XT({.type=Type::kFP32,.shape={1},.buffer=in_scale}));
  weights.emplace("tiny.output_scale", XT({.type=Type::kFP32,.shape={1},.buffer=out_scale}));
  const auto y2 = MobileFullyConnected(input, w2, &weights);
  const auto y4 = MobileFullyConnected(input, w4, &weights);
  REQUIRE(y2.GetStatus().ok()); REQUIRE(y4.GetStatus().ok());
  REQUIRE(!MobileFullyConnected(input, w2).GetStatus().ok());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto runner2, StageRunner::Create({y2},nullptr,nullptr));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto runner4, StageRunner::Create({y4},nullptr,nullptr));
  LRT_TENSOR_RETURN_IF_ERROR(runner2->PrepareRuntime());
  LRT_TENSOR_RETURN_IF_ERROR(runner4->PrepareRuntime());
  REQUIRE(CountStaticInt2RuntimeOperators(runner2->runtime()).value() == 1);
  REQUIRE(CountStaticInt2RuntimeOperators(runner4->runtime()).value() == 0);
  REQUIRE(compact->lowering_count == 1);
  for (int rows : {1, 2, 7, 128, 1}) {
    std::array<int32_t,3> shape{1,rows,columns};
    LRT_TENSOR_RETURN_IF_ERROR(runner2->ReshapeInput(input,shape));
    LRT_TENSOR_RETURN_IF_ERROR(runner4->ReshapeInput(input,shape));
    std::vector<float> data(static_cast<size_t>(rows) * columns);
    for (size_t i = 0; i < data.size(); ++i) {
      const float half = (static_cast<int>(i % 257) - 128 + 0.5f) * in_scale;
      switch (i % 5) {
        case 0: data[i] = half; break;
        case 1: data[i] = std::nextafter(half, -std::numeric_limits<float>::infinity()); break;
        case 2: data[i] = std::nextafter(half, std::numeric_limits<float>::infinity()); break;
        case 3: data[i] = (static_cast<int>(random() % 8192) - 4096) * in_scale / 19.0f; break;
        default: data[i] = 0; break;
      }
    }
    LRT_TENSOR_RETURN_IF_ERROR(runner2->SetInputAsCopy(input,data));
    LRT_TENSOR_RETURN_IF_ERROR(runner4->SetInputAsCopy(input,data));
    LRT_TENSOR_RETURN_IF_ERROR(runner2->Run());
    LRT_TENSOR_RETURN_IF_ERROR(runner4->Run());
    LRT_TENSOR_ASSIGN_OR_RETURN(auto actual,runner2->ReadOutputAs<float>(y2));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto expected,runner4->ReadOutputAs<float>(y4));
    REQUIRE(actual.size() == expected.size());
    if (std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)) != 0) {
      size_t mismatches = 0;
      for (size_t i=0;i<actual.size();++i) if(actual.data()[i]!=expected.data()[i]) {
        if (mismatches++<4) std::cerr << "mismatch columns=" << columns << " channels=" << channels << " rows=" << rows << " i=" << i << " QC2=" << actual.data()[i] << " QC4=" << expected.data()[i] << '\n';
      }
      return absl::InternalError("Static QC2/QC4 outputs differ: " + std::to_string(mismatches));
    }
    ++vectors; values += actual.size();
  }
  return absl::OkStatus();
}
int main() {
  for (auto shape : std::vector<std::pair<int,int>>{{4,3},{28,17},{64,32},{1536,32},{12288,8}}) {
    for (auto scales : std::vector<std::pair<float,float>>{{0.023188971f,0.0305118207f},{0.111712605f,0.334606528f},{0.125f,0.25f}}) {
      auto result=Test(shape.first,shape.second,scales.first,scales.second);
      if(!result.ok()){std::cerr<<result<<'\n';return 1;}
    }
  }
  std::cout << "PASS " << checks << " checks; " << vectors << " reshaped FC invocations; " << values << " output values bitwise QC2/QC4 equal; actual static QS8/QC2W operators verified; no timings collected\n";
}
