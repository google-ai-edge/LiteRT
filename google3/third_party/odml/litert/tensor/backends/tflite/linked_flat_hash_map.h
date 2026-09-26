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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_LINKED_FLAT_HASH_MAP_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_LINKED_FLAT_HASH_MAP_H_

#include <cstddef>
#include <list>
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
namespace litert::tensor {

// Wraps a hash map and a linked list in order to allow lookup and consistent
// traversal order.
template <class K, class V>
class LinkedFlatHashMap {
 public:
  using value_type = std::pair<const K, V>;

  using List = std::list<value_type>;
  using Map = absl::flat_hash_map<K, typename List::iterator>;

  template <bool Const>
  class Iterator {
    using ValueType = std::conditional_t<Const, const value_type, value_type>;

   public:
    ValueType& operator*() { return *list_it_; };
    ValueType* operator->() { return &(*list_it_); }

    Iterator& operator--() {
      --list_it_;
      return *this;
    }

    Iterator operator--(int) {
      Iterator copy = *this;
      --list_it_;
      return copy;
    }

    Iterator& operator++() {
      ++list_it_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator copy = *this;
      ++list_it_;
      return copy;
    }

    bool operator==(const Iterator& other) const {
      return other.map_ == map_ && other.list_it_ == list_it_;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

   private:
    friend class LinkedFlatHashMap;

    Iterator(Map* map, typename List::iterator list_it)
        : map_(map), list_it_(list_it) {}

    Map* map_;
    typename List::iterator list_it_;
  };

  using iterator = Iterator<false>;
  using const_iterator = Iterator<true>;

  void clear() {
    map_.clear();
    list_.clear();
  }

  std::pair<iterator, bool> insert(std::pair<K, V> key_value) {
    auto [it, inserted] = map_.emplace(key_value.first, list_.end());
    if (inserted) {
      it->second = list_.insert(list_.end(), std::move(key_value));
    }
    return {{&map_, it->second}, inserted};
  }

  V& operator[](const K& key) {
    auto map_it = map_.find(key);
    if (map_it == map_.end()) {
      return insert({key, V()}).first->second;
    }
    return map_it->second->second;
  }

  size_t size() const { return map_.size(); }
  bool empty() const { return map_.empty(); }
  iterator begin() { return {&map_, list_.begin()}; }
  iterator end() { return {&map_, list_.end()}; }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const { return {&map_, list_.begin()}; }
  const_iterator cend() const { return {&map_, list_.end()}; }

 private:
  Map map_;
  List list_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_LINKED_FLAT_HASH_MAP_H_
