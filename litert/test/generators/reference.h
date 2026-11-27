// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {
namespace testing {

class ElementWiseComputation {
 private:
  using Shape = std::vector<int32_t>;
  using Shapes = std::vector<Shape>;

 public:
  ElementWiseComputation() = default;

  template <typename It>
  ElementWiseComputation& InShape(It begin, It end) {
    shapes_.push_back(Shape(begin, end));
    return *this;
  }

  ElementWiseComputation& InShape(Shape shape) {
    shapes_.push_back(std::move(shape));
    return *this;
  }

  template <typename It>
  ElementWiseComputation& OutShape(It begin, It end) {
    out_shape_ = Shape(begin, end);
    return *this;
  }

  ElementWiseComputation& OutShape(Shape shape) {
    out_shape_ = std::move(shape);
    return *this;
  }

  template <typename F, typename Out, typename... Ins>
  Expected<void> Compute(F f, Out* output, const Ins*... inputs) const {
    static constexpr size_t kNumInputs = sizeof...(Ins);
    static_assert(std::conjunction_v<std::is_same<Out, Ins>...>);
    LITERT_ASSIGN_OR_RETURN(auto ctx, Prep<kNumInputs>());

    const auto& [out_shape, out_strides, in_shapes, in_strides] = ctx;
    const auto num_elements = out_shape[0] * out_strides[0];
    std::array<size_t, kNumInputs> in_offsets = {};

    for (auto i = 0; i < num_elements; ++i) {
      std::fill(std::begin(in_offsets), std::end(in_offsets), 0);
      int out_offset = i;

      for (auto d = 0; d < out_shape.size(); ++d) {
        int flat_out_d = out_offset / out_strides[d];

        for (auto in = 0; in < kNumInputs; ++in) {
          in_offsets[in] +=
              std::min(flat_out_d, in_shapes[in][d] - 1) * in_strides[in][d];
        }
        out_offset %= out_strides[d];
      }

      output[i] =
          EvalElement(f, std::array<const Out*, kNumInputs>{inputs...},
                      in_offsets, std::make_index_sequence<kNumInputs>());
    }
    return {};
  }

 private:
  static Shape MakeStrides(const Shape& shape) {
    Shape strides(shape.size(), 1);
    for (int d = shape.size() - 2; d >= 0; --d) {
      strides[d] *= strides[d + 1] * shape[d + 1];
    }
    return strides;
  }

  static bool BroadcastCompatible(const Shape& lhs, const Shape& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (int i = 0; i < lhs.size(); ++i) {
      if (lhs[i] != 1 && rhs[i] != 1 && lhs[i] != rhs[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename F, typename Ins, size_t... Is>
  static Ins EvalElement(F f,
                         const std::array<const Ins*, sizeof...(Is)>& inputs,
                         const std::array<size_t, sizeof...(Is)>& in_offsets,
                         std::index_sequence<Is...>) {
    if constexpr (sizeof...(Is) == 1) {
      return f(inputs[0][in_offsets[0]]);
    } else {
      const std::array<Ins, sizeof...(Is)> vals = {
          inputs[Is][in_offsets[Is]]...};
      return std::reduce(vals.cbegin() + 1, vals.cend(), *vals.cbegin(), F());
    }
  }

  template <size_t NumInputs>
  struct Context {
    Shape out_shape;
    Shape out_strides;
    std::array<Shape, NumInputs> input_shapes;
    std::array<Shape, NumInputs> input_strides;
  };

  template <size_t NumInputs>
  Expected<Context<NumInputs>> Prep() const {
    if (NumInputs != shapes_.size()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "ElementWiseComputation requires same number of inputs as "
                   "input shapes");
    }
    const auto out_rank = out_shape_.size();
    Context<NumInputs> res;
    for (int i = 0; i < NumInputs; ++i) {
      const auto& shape = shapes_[i];
      if (shape.empty()) {
        res.input_shapes[i] = Shape(out_rank, 1);
      } else if (shape.size() == out_rank) {
        if (!BroadcastCompatible(shape, out_shape_)) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Incompatible broadcast");
        }
        res.input_shapes[i] = shape;
      } else {
        return Error(kLiteRtStatusErrorInvalidArgument, "Incompatible rank");
      }
      res.input_strides[i] = MakeStrides(res.input_shapes[i]);
    }

    res.out_shape = out_shape_;
    res.out_strides = MakeStrides(out_shape_);

    return res;
  }

  Shapes shapes_;
  Shape out_shape_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_H_
