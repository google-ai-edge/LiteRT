/* Copyright 2025 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_H_

// The LiterT Tensor API underlying graph is represented using `TensorGroup` and
// `Operation` objects linked together. A `Tensor` is an `(index, tensor group)`
// pair.
//
// Because this API is used to build graph incrementaly, this representation
// allows us to avoid having to carry around a builder object that would hold
// all of the information.
//
// The ownership strategy is the following:
//
// - `Operation`s own their inputs.
// - `TensorGroup`s own the operation that produces them.
// - `Tensor`s own the tensor group they refer to (that ownership is shared).
//
// The links are bidirectional, with a weak pointer used from the inputs of a
// graph towards its outputs.
//
// This ensures that a long as a `Tensor` is alive, all the graph leading to
// that tensor is also kept alive. In turn that means that we can then serialize
// a model from its outputs.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/type_id.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::graph {

struct Operation;

struct Quantization {
  virtual ~Quantization() = default;
  virtual internal::TypeId GetTypeId() const = 0;
  virtual bool IsA(internal::TypeId id) const = 0;

  template <class T>
  absl::StatusOr<T&> As() {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<T*>(this);
    }
    return absl::InvalidArgumentError("Wrong quantization type.");
  }

  template <class T>
  absl::StatusOr<const T&> As() const {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<const T*>(this);
    }
    return absl::InvalidArgumentError("Wrong quantization type.");
  }
};

template <class T>
struct ZeroPointView {
  const T* data;
  // 0 for symmetric/broadcast zero points, 1 for one value per channel/block.
  size_t stride;

  T operator[](size_t i) const { return data[i * stride]; }
};

// Zero points of a quantized tensor, stored as either integers (`int64_t`) or
// floats (`float`, e.g. `-0.5f` for centered INT2):
// - No data (`empty()`): symmetric quantization (all zero).
// - 1 element: broadcast to all channels or blocks.
// - N elements: one per channel or block.
//
// `IsSymmetric()` returns true when empty or when all stored zero points equal
// zero. Conversions to integer types (and narrower integers) fail unless every
// value is finite, integral, and in range; integer-to-float conversions use
// `static_cast` directly (exact for magnitudes up to 2^24 in `float`).
class ZeroPoints {
 public:
  static ZeroPoints From(std::vector<int64_t> values) {
    return FromVector(std::move(values));
  }
  static ZeroPoints From(std::vector<float> values) {
    return FromVector(std::move(values));
  }
  static ZeroPoints Int(std::vector<int64_t> values) {
    return FromVector(std::move(values));
  }
  static ZeroPoints Float(std::vector<float> values) {
    return FromVector(std::move(values));
  }

  ZeroPoints() = default;
  ZeroPoints(const ZeroPoints&) = default;
// GCC 12.4 to 14 report false `-Warray-bounds` and `-Wstringop-overflow`
// errors in the inlined `std::variant` copy assignment of `data_`
// (https://gcc.gnu.org/PR113239).
#if defined(__GNUC__) && !defined(__clang__) && \
    (__GNUC__ * 100 + __GNUC_MINOR__) >= 1204 && __GNUC__ < 15
#define LRT_TENSOR_GCC_PR113239_WORKAROUND 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
  ZeroPoints& operator=(const ZeroPoints&) = default;
#ifdef LRT_TENSOR_GCC_PR113239_WORKAROUND
#pragma GCC diagnostic pop
#undef LRT_TENSOR_GCC_PR113239_WORKAROUND
#endif

  // Moved-from objects are left empty (symmetric) and store integers.
  ZeroPoints(ZeroPoints&& other) noexcept
      : data_(std::exchange(other.data_, {})),
        is_symmetric_(std::exchange(other.is_symmetric_, true)),
        is_uniform_(std::exchange(other.is_uniform_, true)) {}

  ZeroPoints& operator=(ZeroPoints&& other) noexcept {
    if (this != &other) {
      data_ = std::exchange(other.data_, {});
      is_symmetric_ = std::exchange(other.is_symmetric_, true);
      is_uniform_ = std::exchange(other.is_uniform_, true);
    }
    return *this;
  }

  // Implicit conversion preserved for pre-existing code passing
  // `std::vector<int64_t>` or braced integer initializer lists. New callers
  // should use `ZeroPoints::From(...)`, `ZeroPoints::Int(...)`, or
  // `ZeroPoints::Float(...)`.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ZeroPoints(std::vector<int64_t> values)
      : ZeroPoints(FromVector(std::move(values))) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  ZeroPoints(std::initializer_list<int64_t> values)
      : ZeroPoints(FromVector(std::vector<int64_t>(values))) {}
  ~ZeroPoints() = default;

  // Preserved for pre-existing integer callers (e.g., `zero_points.assign(n,
  // 0)`). New callers should assign `ZeroPoints::Int(...)` or
  // `ZeroPoints::Float(...)`.
  void assign(size_t count, int64_t value) {
    *this = Int(std::vector<int64_t>(count, value));
  }

  bool empty() const { return size() == 0; }

  // Returns true when no zero points are stored or all stored values are 0.
  bool IsSymmetric() const { return is_symmetric_; }

  // Returns true when no zero points are stored or all stored values are
  // equal.
  bool IsUniform() const { return is_uniform_; }

  size_t size() const {
    return std::visit([](const auto& zps) { return zps.size(); }, data_);
  }

  // Returns the underlying vector if stored as `T` (`int64_t` or `float`), or
  // `absl::NotFoundError` otherwise.
  template <class T>
  absl::StatusOr<const std::vector<T>&> Data() const {
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, float>) {
      if (const auto* values = std::get_if<std::vector<T>>(&data_)) {
        return *values;
      }
    }
    return absl::NotFoundError(
        "Zero points are not stored as the requested element type.");
  }

  // OK if there are 0, 1, or `n` zero points and all of them can be
  // represented as `T` (e.g. `-0.5f` cannot be an integer).
  template <class T>
  absl::Status Validate(size_t n) const {
    return Visit([this, n](const auto& zps) -> absl::Status {
      if (absl::Status status = CheckCount(zps.size(), n); !status.ok()) {
        return status;
      }
      if (IsSymmetric()) {
        return absl::OkStatus();
      }
      if (is_uniform_) {
        return ConvertValue<T>(zps[0]).status();
      }
      for (const auto zp : zps) {
        LRT_TENSOR_RETURN_IF_ERROR(ConvertValue<T>(zp).status());
      }
      return absl::OkStatus();
    });
  }

  // Converts and broadcasts into `dst`:
  // - empty -> zeros,
  // - one value -> repeated,
  // - `dst.size()` values -> converted.
  //
  // Same errors as `Validate`.
  template <class T>
  absl::Status CopyTo(absl::Span<T> dst) const {
    return Visit([dst](const auto& zps) -> absl::Status {
      if (absl::Status status = CheckCount(zps.size(), dst.size());
          !status.ok()) {
        return status;
      }
      if (zps.empty()) {
        std::fill(dst.begin(), dst.end(), T{0});
      } else if (zps.size() == 1) {
        absl::StatusOr<T> value = ConvertValue<T>(zps[0]);
        if (!value.ok()) {
          return value.status();
        }
        std::fill(dst.begin(), dst.end(), *value);
      } else {
        for (size_t i = 0; i < dst.size(); ++i) {
          absl::StatusOr<T> value = ConvertValue<T>(zps[i]);
          if (!value.ok()) {
            return value.status();
          }
          dst[i] = *value;
        }
      }
      return absl::OkStatus();
    });
  }

  // Returns an `n`-element vector populated by `CopyTo<T>`.
  template <class T>
  absl::StatusOr<std::vector<T>> CopyAs(size_t n) const {
    std::vector<T> vec(n);
    LRT_TENSOR_RETURN_IF_ERROR(CopyTo<T>(absl::MakeSpan(vec)));
    return vec;
  }

  // Returns the single (or uniform) zero point converted to `T` (0 if
  // symmetric). Fails if values differ or cannot be represented as `T`.
  template <class T>
  absl::StatusOr<T> Scalar() const {
    if (IsSymmetric()) {
      return T{0};
    }
    return Visit([this](const auto& zps) -> absl::StatusOr<T> {
      if (!is_uniform_) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected a single or uniform zero point, got ", zps.size(), "."));
      }
      return ConvertValue<T>(zps[0]);
    });
  }

  // Returns the zero point of channel or block `idx` converted to `T`: 0 if
  // empty, the single value if broadcast, or element `idx` if stored per
  // channel/block. Fails if `idx` is out of range or cannot be represented as
  // `T`.
  template <class T>
  absl::StatusOr<T> At(size_t idx) const {
    return Visit([idx](const auto& zps) -> absl::StatusOr<T> {
      if (zps.empty()) {
        return T{0};
      }
      if (zps.size() == 1) {
        return ConvertValue<T>(zps[0]);
      }
      if (idx >= zps.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Zero point index ", idx, " is out of range for size ",
                         zps.size(), "."));
      }
      return ConvertValue<T>(zps[idx]);
    });
  }

  // Calls `func(view)` once with a `ZeroPointView<ElemT>` (`int64_t` or
  // `float`) of length `n` without allocating or copying. Fails with
  // `absl::InvalidArgumentError` if `size()` is not 0, 1, or `n`.
  template <class F>
  absl::Status Visit(size_t n, F&& func) const {
    return Visit([n, &func](const auto& zps) -> absl::Status {
      using ElemT = typename std::decay_t<decltype(zps)>::value_type;
      if (absl::Status status = CheckCount(zps.size(), n); !status.ok()) {
        return status;
      }
      static constexpr ElemT kZero = ElemT{0};
      const ZeroPointView<ElemT> view =
          zps.empty()
              ? ZeroPointView<ElemT>{&kZero, 0}
              : ZeroPointView<ElemT>{zps.data(),
                                     zps.size() == 1 ? size_t{0} : size_t{1}};
      if constexpr (std::is_same_v<
                        std::invoke_result_t<F&, ZeroPointView<ElemT>>,
                        absl::Status>) {
        return std::forward<F>(func)(view);
      } else {
        static_assert(
            std::is_void_v<std::invoke_result_t<F&, ZeroPointView<ElemT>>>,
            "Visit callback must return void or absl::Status.");
        std::forward<F>(func)(view);
        return absl::OkStatus();
      }
    });
  }

  // Gives read-only access to the underlying vector for advanced operations.
  template <class F>
  decltype(auto) Visit(F&& func) const {
    return std::visit(std::forward<F>(func), data_);
  }

  bool operator==(const ZeroPoints& other) const {
    return data_ == other.data_;
  }
  bool operator!=(const ZeroPoints& other) const {
    return !(*this == other);
  }

 private:
  template <class T>
  ZeroPoints(std::vector<T> values, bool is_symmetric, bool is_uniform)
      : data_(std::move(values)),
        is_symmetric_(is_symmetric),
        is_uniform_(is_uniform) {}

  template <class T>
  static ZeroPoints FromVector(std::vector<T> values) {
    bool is_uniform = true;
    bool is_symmetric = true;
    if (!values.empty()) {
      const T first = values[0];
      for (size_t i = 1; i < values.size(); ++i) {
        if (!(values[i] == first)) {
          is_uniform = false;
          break;
        }
      }
      is_symmetric = is_uniform && first == 0;
    }
    return ZeroPoints(std::move(values), is_symmetric, is_uniform);
  }

  static absl::Status CheckCount(size_t zp_size, size_t n) {
    if (zp_size != 0 && zp_size != 1 && zp_size != n) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected 0, 1, or ", n, " zero points, got ", zp_size, "."));
    }
    return absl::OkStatus();
  }

  template <class Dst, class Src>
  static absl::StatusOr<Dst> ConvertValue(Src value) {
    static_assert(std::is_arithmetic_v<Dst> && !std::is_same_v<Dst, bool>,
                  "Zero point target type must be numeric.");
    if constexpr (std::is_same_v<Dst, Src>) {
      return value;
    } else if constexpr (std::is_floating_point_v<Dst>) {
      // Integer-to-float conversion intentionally uses `static_cast` without a
      // precision check (exact for magnitudes up to 2^24 in `float`), unlike
      // float-to-integer conversions below which require exact whole numbers.
      return static_cast<Dst>(value);
    } else if constexpr (std::is_floating_point_v<Src>) {
      if (!std::isfinite(value)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Non-finite float zero point ", value, "."));
      }
      if (std::trunc(value) != value) {
        return absl::UnimplementedError(absl::StrCat(
            "Float zero point ", value,
            " cannot be represented as an integer."));
      }
      const double d = static_cast<double>(value);
      const double min_val =
          static_cast<double>(std::numeric_limits<Dst>::min());
      const double upper_bound =
          std::ldexp(1.0, std::numeric_limits<Dst>::digits);
      if (d < min_val || d >= upper_bound) {
        return absl::UnimplementedError(absl::StrCat(
            "Float zero point ", value, " is out of target integer range."));
      }
      return static_cast<Dst>(d);
    } else if constexpr (std::is_signed_v<Dst>) {
      if (value < std::numeric_limits<Dst>::min() ||
          value > std::numeric_limits<Dst>::max()) {
        return absl::UnimplementedError(absl::StrCat(
            "Zero point ", value, " is out of target integer range."));
      }
      return static_cast<Dst>(value);
    } else {
      if (value < 0 ||
          static_cast<uint64_t>(value) > std::numeric_limits<Dst>::max()) {
        return absl::UnimplementedError(absl::StrCat(
            "Zero point ", value,
            " cannot be represented as the target unsigned integer type."));
      }
      return static_cast<Dst>(value);
    }
  }

  std::variant<std::vector<int64_t>, std::vector<float>> data_;
  bool is_symmetric_ = true;
  bool is_uniform_ = true;
};

// Holds quantization parameters for a tensor.
//
// Dequantization formula for element `i` in channel `c`:
//   w = scales[c] * (q_signed[i] - zp[c])
struct PerChannelAffineQuantization : Quantization {
  PerChannelAffineQuantization() = default;

  PerChannelAffineQuantization(std::vector<float> scales,
                               ZeroPoints zero_points,
                               int quantized_dimension = 0)
      : scales(std::move(scales)),
        zero_points(std::move(zero_points)),
        quantized_dimension(quantized_dimension) {}

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<PerChannelAffineQuantization>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<PerChannelAffineQuantization>();
  }

  std::vector<float> scales;
  ZeroPoints zero_points;
  int quantized_dimension = 0;
};

// Holds blockwise quantization parameters for a tensor.
//
// Dequantization formula for element `i` in block `b`:
//   w = scales[b] * (q_signed[i] - zp[b])
struct BlockwiseQuantization : Quantization {
  BlockwiseQuantization() = default;

  BlockwiseQuantization(std::vector<float> scales, ZeroPoints zero_points,
                        int block_size, int quantized_dimension = 0)
      : scales(std::move(scales)),
        zero_points(std::move(zero_points)),
        block_size(block_size),
        quantized_dimension(quantized_dimension) {}

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<BlockwiseQuantization>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<BlockwiseQuantization>();
  }

  std::vector<float> scales;
  ZeroPoints zero_points;
  int block_size = 0;
  int quantized_dimension = 0;
  // When `scales_tensor_name` is set, the TFLite serializer writes no scale
  // or zero point tensor of its own: the details reference the existing
  // tensor of that name in the same subgraph (e.g. the float32 scale input of
  // a "moe" custom op), and `block_shape` (one entry per dimension) is written
  // when not empty. This is LiteRT Quantizer's multi-axis form for MoE
  // experts. Non-zero `zero_points` with a `scales_tensor_name` are an error.
  // A `block_shape` without `scales_tensor_name` is an error: inline `scales`
  // are written for `block_size` only.
  std::vector<int32_t> block_shape;
  std::string scales_tensor_name;
};

using Shape = std::vector<int32_t>;

// Holds information about a tensor.
struct TensorInformation {
  std::string name;
  std::vector<std::weak_ptr<Operation>> consumers;
  Type type = Type::kUnknown;
  Shape shape;
  std::shared_ptr<Buffer> buffer;
  std::shared_ptr<Quantization> quantization;
  size_t GetSize() const {
    size_t size = 1;
    for (int32_t dim : shape) {
      size *= dim;
    }
    return size;
  }
};

// Holds tensors that must share their lifetime.
//
// This is used to model the fact that an operation's output tensors should all
// live as long a one of them is still in use.
struct TensorGroup {
  std::shared_ptr<Operation> producer;
  std::vector<TensorInformation> tensor_infos;

  // Use for debugging. This should hold the location of the op call that
  // creates this object.
  absl::SourceLocation loc;
  // Keeps track of the validity of this group. If an operation detects a
  // precondition violation, it should return an invalid group.
  //
  // An invalid group's tensors are not guaranteed to exist. The producer should
  // still be set.
  absl::Status status = absl::OkStatus();
};

// Links to a specific tensor in a tensor group.
//
// This is used to simplify single tensor manipulation.
struct Tensor {
  std::shared_ptr<TensorGroup> group;
  size_t index;

  // Checks for handle equality.
  //
  // Warning: This checks whether two Tensor objects point to the same
  // underlying data, not that the tensor properties are equal.
  bool operator==(const Tensor& b) const {
    return group == b.group && index == b.index;
  }

  template <class H>
  friend H AbslHashValue(H h, const Tensor& t) {
    return H::combine(std::move(h), t.group, t.index);
  }
};

class BackendExtension {
 public:
  virtual ~BackendExtension() = default;
  virtual internal::TypeId GetTypeId() const = 0;
};

using ScopeAttrValue = std::variant<int64_t, double, std::string>;

// Represents a hierarchical layer or composite function scope in the graph.
//
// Layer and composite functions (e.g. `RoPE`, `ScaledDotProductAttention`,
// `FeedForward`, `GatedAttention`, `GatedDeltaNet`, `transformer_layer`)
// create a `Scope` via `ScopeGuard` so that hierarchical structure and fusion
// boundaries (boundary inputs, weights, states, outputs, and attributes) are
// preserved on the serialized `Operation` DAG.
//
// Ownership is strictly acyclic:
// - `Operation` holds `std::shared_ptr<Scope> scope` and `Scope` holds
//   `std::shared_ptr<Scope> parent`.
// - `Scope::children` holds `std::weak_ptr<Scope>`.
// - `Scope::boundary_tensors` holds a strong `TensorGroup` reference only for
//   leaf tensors (`producer == nullptr`, e.g. constant weights / graph inputs)
//   and a weak `TensorGroup` reference for operation outputs (`producer !=
//   nullptr`).
struct Scope : public std::enable_shared_from_this<Scope> {
  struct BoundaryEntry {
    std::shared_ptr<TensorGroup> strong_group;
    std::weak_ptr<TensorGroup> weak_group;
    size_t index = 0;

    std::optional<Tensor> Lock() const {
      if (strong_group != nullptr) {
        return Tensor{strong_group, index};
      }
      if (auto locked = weak_group.lock()) {
        return Tensor{std::move(locked), index};
      }
      return std::nullopt;
    }
  };

  std::string kind;
  std::string name;
  std::shared_ptr<Scope> parent;
  std::vector<std::weak_ptr<Scope>> children;
  absl::flat_hash_map<std::string, ScopeAttrValue> attrs;
  absl::flat_hash_map<std::string, BoundaryEntry> boundary_tensors;

  // Returns the slash-separated path from the root scope to `this`, e.g.
  // "layer_0/gated_attention/sdpa".
  std::string FullPath() const;

  // Finds the nearest scope in the parent chain (inclusive of `this`) whose
  // `kind` equals `target_kind`, or nullptr if none matches.
  const Scope* FindAncestor(absl::string_view target_kind) const;
  std::shared_ptr<Scope> FindAncestorShared(absl::string_view target_kind);

  void SetAttr(absl::string_view key, ScopeAttrValue value);
  std::optional<int64_t> GetIntAttr(absl::string_view key) const;
  std::optional<double> GetFloatAttr(absl::string_view key) const;
  std::optional<absl::string_view> GetStringAttr(absl::string_view key) const;

  void BindTensor(absl::string_view role, const Tensor& tensor);
  template <class T, class = decltype(std::declval<const T&>().GetRaw())>
  void BindTensor(absl::string_view role, const T& tensor) {
    BindTensor(role, tensor.GetRaw());
  }
  std::optional<Tensor> GetBoundaryTensor(absl::string_view role) const;
};

// Returns the currently active Scope on the calling thread, or nullptr.
std::shared_ptr<Scope> CurrentScope();

// RAII guard that pushes a new hierarchical `Scope` onto the thread-local
// scope stack. All `Operation`s constructed while the guard is active
// automatically capture `op->scope = CurrentScope()`.
class ScopeGuard {
 public:
  explicit ScopeGuard(absl::string_view kind, absl::string_view name = "");
  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

  ScopeGuard& Attr(absl::string_view key, int64_t value);
  ScopeGuard& Attr(absl::string_view key, int value) {
    return Attr(key, static_cast<int64_t>(value));
  }
  ScopeGuard& Attr(absl::string_view key, bool value) {
    return Attr(key, static_cast<int64_t>(value ? 1 : 0));
  }
  ScopeGuard& Attr(absl::string_view key, double value);
  ScopeGuard& Attr(absl::string_view key, absl::string_view value);
  ScopeGuard& Attr(absl::string_view key, const char* value) {
    return Attr(key, absl::string_view(value));
  }
  ScopeGuard& Bind(absl::string_view role, const Tensor& tensor);
  template <class T, class = decltype(std::declval<const T&>().GetRaw())>
  ScopeGuard& Bind(absl::string_view role, const T& tensor) {
    return Bind(role, tensor.GetRaw());
  }
  template <class T, class = decltype(std::declval<const T&>().GetRaw())>
  ScopeGuard& Bind(absl::string_view role, const std::optional<T>& value) {
    if (value.has_value()) {
      return Bind(role, *value);
    }
    return *this;
  }

  const std::shared_ptr<Scope>& scope() const { return scope_; }

 private:
  std::shared_ptr<Scope> prev_scope_;
  std::shared_ptr<Scope> scope_;
};

#define LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION                        \
  internal::TypeId GetTypeId() const override {                                \
    return internal::TypeId::Get<                                              \
        std::decay_t<std::remove_pointer_t<decltype(this)>>>();                \
  }                                                                            \
  bool IsA(internal::TypeId id) const override {                               \
    return id == internal::TypeId::Get<                                        \
                     std::decay_t<std::remove_pointer_t<decltype(this)>>>() || \
           Operation::IsA(id);                                                 \
  }

// Represents an ML operation.
struct Operation {
  virtual ~Operation();

  virtual internal::TypeId GetTypeId() const {
    return internal::TypeId::Get<Operation>();
  }
  virtual bool IsA(internal::TypeId id) const {
    return id == internal::TypeId::Get<Operation>();
  }

  template <class T>
  absl::StatusOr<T&> As() {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<T*>(this);
    }
    return absl::InvalidArgumentError("Wrong operation type.");
  }

  template <class T>
  absl::StatusOr<const T&> As() const {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<const T*>(this);
    }
    return absl::InvalidArgumentError("Wrong operation type.");
  }

  template <typename T>
  const T* GetExtension() const {
    for (const auto& ext : extensions) {
      if (ext->GetTypeId() == internal::TypeId::Get<T>()) {
        return static_cast<const T*>(ext.get());
      }
    }
    return nullptr;
  }

  std::vector<Tensor> inputs;
  std::weak_ptr<TensorGroup> outputs_group;

  // Hierarchical layer/composite scope active when this operation was created.
  std::shared_ptr<Scope> scope = CurrentScope();

  // For testing and debugging purposes only.
  virtual absl::string_view GetName() const = 0;

  absl::InlinedVector<std::unique_ptr<BackendExtension>, 4> extensions;
};

class OpDebugger {
 public:
  static void DebugOp(const Operation& op);
  static bool Enabled();

 private:
  static std::string GetTensorInfoDebugString(size_t index,
                                              const TensorInformation& info,
                                              const absl::Status& status);
};

// Creates a tensor holding the given status.
Tensor ErrorTensor(absl::Status status, absl::SourceLocation op_loc =
                                            absl::SourceLocation::current());

// Creates a tensor group with a single tensor.
//
// `op_loc` should track the location of the function that creates this tensor.
//
// Note: `op_loc` is intentionally non defaulted to
// `absl::SourceLocation::current()` to force you to pass in the location of the
// function that creates this tensor.
Tensor NewTensor(absl::SourceLocation op_loc);

// Adds a tensor to an existing tensor group.
Tensor NewTensor(std::shared_ptr<TensorGroup>& group);

// Creates a new tensor group holding `count` tensors.
//
// `op_loc` should track the location of the function that creates this tensor.
//
// Note: `op_loc` is intentionally non defaulted to
// `absl::SourceLocation::current()` to force you to pass in the location of the
// function that creates this tensor.
std::shared_ptr<TensorGroup> NewTensorGroup(size_t count,
                                            absl::SourceLocation op_loc);

// Creates a tensor handle from an existing tensor group.
//
// Warning: The tensor may be invalid and should be checked with `GetStatus()`.
Tensor GetTensor(size_t index, std::shared_ptr<TensorGroup> group);

// Materializes tensors held by a tensor group.
absl::StatusOr<std::vector<Tensor>> GetTensors(
    const std::shared_ptr<TensorGroup>& group);

// Locks an operation's outputs and materializes tensors for them.
absl::StatusOr<std::vector<Tensor>> GetOutputs(const Operation& operation);

// Gets a tensor information.
absl::StatusOr<TensorInformation&> GetInfo(Tensor& tensor);

// Gets a tensor information.
absl::StatusOr<const TensorInformation&> GetInfo(const Tensor& tensor);

// Gets the `consumers` of a tensor.
absl::StatusOr<std::vector<std::weak_ptr<Operation>>&> GetConsumers(
    Tensor& tensor);

// Gets the `producer` of a tensor.
absl::StatusOr<std::shared_ptr<Operation>&> GetProducer(const Tensor& tensor);

// Gets the `name` of a tensor.
absl::StatusOr<const std::string&> GetName(const Tensor& tensor);

// Sets the `name` of a tensor.
absl::Status SetName(Tensor& tensor, std::string name);

// Sets the `buffer` of a tensor.
absl::Status SetBuffer(Tensor& tensor, std::shared_ptr<Buffer> buffer);

// Gets the `buffer` of a tensor.
absl::StatusOr<Buffer&> GetBuffer(Tensor& tensor);

// Gets the status of a tensor.
//
// This reflects whether the tensor can still be used to create a graph. An
// operation that detects invalid preconditions may return an invalid tensor.
absl::Status GetStatus(const Tensor& tensor);

// Gets the location where a tensor was created.
absl::SourceLocation GetLocation(const Tensor& tensor);

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_H_
