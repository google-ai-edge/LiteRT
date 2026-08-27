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
// Source code origin:
// https://github.com/syoyo/minijson
// SPDX-License-Identifier: MIT Copyright 2023 - Present, Syoyo Fujita.

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_MINIJSON_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_MINIJSON_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace minijson {

// Simple C++ implementation of Python's OrderedDict like dictonary
// (preserves key insertion order)
// Modified for JSON:
// - No duplicated key allowed

template <typename T>
class ordered_dict {
 public:
  bool at(const size_t idx, T* dst) const {
    if (idx >= keys_.size()) {
      return false;
    }
    return at(keys_[idx], dst);
  }

  bool count(const std::string& key) const { return m_.count(key); }

  void insert(const std::string& key, const T& value) {
    if (m_.count(key)) {
      // overwrite existing value
    } else {
      keys_.push_back(key);
    }

    m_[key] = value;
  }

  void insert(const std::string& key, T&& value) {
    if (m_.count(key)) {
      // overwrite existing value
    } else {
      keys_.push_back(key);
    }

    m_[key] = std::move(value);
  }

  bool at(const std::string& key, T* dst) const {
    auto it = m_.find(key);
    if (it == m_.end()) {
      return false;
    }

    *dst = it->second;
    return true;
  }

  const std::vector<std::string>& keys() const { return keys_; }

  size_t size() const { return m_.size(); }

  bool erase(const std::string& key) {
    // simple linear search
    for (size_t i = 0; i < keys_.size(); i++) {
      if (keys_[i] == key) {
        keys_.erase(keys_.begin() + i);
        m_.erase(key);
        return true;
      }
    }

    return false;
  }

 private:
  std::vector<std::string> keys_;
  std::map<std::string, T> m_;
};

}  // namespace minijson

#if defined(MINIJSON_USE_STRTOD)
// Use stdlib's strtod
#include <cstring>
#else

namespace minijson {
namespace simdjson {
namespace internal {

double from_chars(const char* first) noexcept;
double from_chars(const char* first, const char* end) noexcept;

char* to_chars(char* first, const char* last, double value);

}  // namespace internal
}  // namespace simdjson
}  // namespace minijson

#endif

namespace minijson {

namespace detail {

double from_chars(const char* p);
const char* my_strchr(const char* p, int ch);

}  // namespace detail

namespace detail {

//
// Usage:
//  - set_input()
//  - scan_string()
//    - success: use `token_buffer` string
//    - error: use `error_message`
//
struct string_parser {
  // input string must be UTF-8
  void set_input(const std::string& s) { input = s; }

  bool scan_string();

  void reset() {
    if (!input.empty()) {
      current = input[0];
    } else {
      current = '\0';
    }
    curr_idx = 0;
    token_buffer.clear();
  }

  // fetch next token.
  unsigned char get() {
    if ((curr_idx + 1) < input.size()) {
      curr_idx++;
      current = input[curr_idx];
      return current;
    }
    current = '\0';
    return current;
  }

  bool eof() {
    if (input.empty()) {
      return true;
    }

    if (curr_idx >= input.size()) {
      return true;
    }

    return false;
  }

  void add(const unsigned char c) { token_buffer += c; }

  void add(const int i) {
    // use lower 8bit
    token_buffer += static_cast<unsigned char>(i & 0xff);
  }

  int get_codepoint();

  bool next_byte_in_range(std::initializer_list<int> ranges);

  std::string error_message;
  std::string token_buffer;  // output

  unsigned char current{'\0'};
  size_t curr_idx{0};

  std::string input;
};

}  // namespace detail

typedef enum {
  unknown_type,
  null_type,
  boolean_type,
  number_type,
  string_type,
  array_type,
  object_type,
} type;

typedef enum {
  no_error,
  undefined_error,
  invalid_token_error,
  unknown_type_error,
  memory_allocation_error,
  corrupted_json_error,
  duplicated_key_error,
} error;

class value;

typedef bool boolean;
typedef double number;
typedef std::string string;
typedef ordered_dict<value> object;
typedef std::vector<value> array;
typedef struct {
} null_t;

// null_t null;

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<null_t> {
  static constexpr uint32_t type_id() { return 0; }
};

template <>
struct TypeTraits<boolean> {
  static constexpr uint32_t type_id() { return 1; }
};

template <>
struct TypeTraits<number> {
  static constexpr uint32_t type_id() { return 2; }
};

template <>
struct TypeTraits<string> {
  static constexpr uint32_t type_id() { return 3; }
};

template <>
struct TypeTraits<object> {
  static constexpr uint32_t type_id() { return 4; }
};

template <>
struct TypeTraits<array> {
  static constexpr uint32_t type_id() { return 5; }
};

class value {
 private:
  type t_;
  union {
    null_t n;
    boolean b;
    number d;
    std::string* s;
    array* a;
    object* o;
  } u_;

  void _free_u() {
    if (t_ == string_type) {
      delete this->u_.s;
      this->u_.s = nullptr;
    }
    if (t_ == array_type) {
      delete this->u_.a;
      this->u_.a = nullptr;
    }
    if (t_ == object_type) {
      delete this->u_.o;
      this->u_.o = nullptr;
    }
  }

 public:
  // NOLINTBEGIN(*-explicit-constructor)
  value() : t_(unknown_type), u_() {}
  value(null_t n) : t_(null_type), u_() { u_.n = n; }
  value(boolean b) : t_(boolean_type), u_() { u_.b = b; }
  value(number d) : t_(boolean_type), u_() { u_.d = d; }
  value(const char* s) : t_(string_type), u_() { u_.s = new std::string(s); }
  value(const std::string& s) : t_(string_type), u_() {
    u_.s = new std::string(s);
  }
  value(const array& a) : t_(array_type), u_() { u_.a = new array(a); }
  value(const object& o) : t_(object_type), u_() { u_.o = new object(o); }
  value(const value& v) : t_(v.t_), u_() {
    if (t_ == array_type) {
      u_.a = new array();
      *u_.a = *v.u_.a;
    } else if (t_ == object_type) {
      u_.o = new object();
      *u_.o = *v.u_.o;
    } else if (t_ == string_type) {
      u_.s = new std::string();
      *u_.s = *v.u_.s;
    } else {
      u_.d = v.u_.d;
    }
  }
  ~value() { _free_u(); }
  // NOLINTEND(*-explicit-constructor)

  template <typename T>
  bool is() const {
    if (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id() &&
        t_ == null_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id() &&
        t_ == boolean_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<number>::type_id() &&
        t_ == number_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<std::string>::type_id() &&
        t_ == string_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<array>::type_id() &&
        t_ == array_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<object>::type_id() &&
        t_ == object_type)
      return true;
    return false;
  }

  template <typename T>
  const T* as() const {
    if ((t_ == array_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<array>::type_id())) {
      return reinterpret_cast<const T*>(u_.a);
    }

    if ((t_ == object_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<object>::type_id())) {
      return reinterpret_cast<const T*>(u_.o);
    }

    if ((t_ == string_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<std::string>::type_id())) {
      return reinterpret_cast<const T*>(u_.s);
    }

    if ((t_ == null_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id())) {
      return reinterpret_cast<const T*>(&u_.n);
    }

    if ((t_ == boolean_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id())) {
      return reinterpret_cast<const T*>(&u_.b);
    }

    if ((t_ == number_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<number>::type_id())) {
      return reinterpret_cast<const T*>(&u_.d);
    }

    return nullptr;
  }

  template <typename T>
  T* as() {
    if ((t_ == array_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<array>::type_id())) {
      return reinterpret_cast<T*>(u_.a);
    }

    if ((t_ == object_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<object>::type_id())) {
      return reinterpret_cast<T*>(u_.o);
    }

    if ((t_ == string_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<string>::type_id())) {
      return reinterpret_cast<T*>(u_.s);
    }

    if ((t_ == null_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id())) {
      return reinterpret_cast<T*>(&u_.n);
    }

    if ((t_ == boolean_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id())) {
      return reinterpret_cast<T*>(&u_.b);
    }

    if ((t_ == number_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<number>::type_id())) {
      return reinterpret_cast<T*>(&u_.d);
    }

    return nullptr;
  }

  // NOLINTBEGIN(misc-unconventional-assign-operator)
  null_t& operator=(null_t& n) {
    t_ = null_type;
    u_.n = n;
    return u_.n;
  }
  boolean& operator=(boolean b) {
    t_ = boolean_type;
    u_.b = b;
    return u_.b;
  }
  number& operator=(number d) {
    t_ = number_type;
    u_.d = d;
    return u_.d;
  }
  const std::string& operator=(const char* s) {
    _free_u();
    t_ = string_type;
    u_.s = new std::string(s);
    return *u_.s;
  }
  const std::string& operator=(const std::string& s) {
    _free_u();
    t_ = string_type;
    u_.s = new std::string(s);
    return *u_.s;
  }
  const object& operator=(const object& o) {
    _free_u();
    t_ = object_type;
    u_.o = new object(o);
    return *u_.o;
  }
  const array& operator=(const array& a) {
    _free_u();
    t_ = array_type;
    u_.a = new array(a);
    return *u_.a;
  }
  const value& operator=(const value& v) {
    _free_u();
    t_ = v.t_;
    if (t_ == array_type) {
      u_.a = new array(*v.u_.a);
    } else if (t_ == object_type) {
      u_.o = new object(*v.u_.o);
    } else if (t_ == string_type) {
      u_.s = new std::string(*v.u_.s);
    } else {
      u_.d = v.u_.d;
    }
    return *this;
  }
  // NOLINTEND(misc-unconventional-assign-operator)

  std::string type_name() const {
    if (t_ == array_type) {
      return "array";
    }

    if (t_ == object_type) {
      return "object";
    }

    if (t_ == string_type) {
      return "string";
    }

    if (t_ == null_type) {
      return "null";
    }

    if (t_ == boolean_type) {
      return "boolean";
    }

    if (t_ == number_type) {
      return "number";
    }

    return "[[invalid]]";
  }

  std::string str(const char* p) const {
    std::stringstream ss;
    ss << '"';
    while (*p) {
      if (*p == '\n') {
        ss << "\\n";
      } else if (*p == '\r') {
        ss << "\\r";
      } else if (*p == '\t') {
        ss << "\\t";
      } else if (detail::my_strchr("\"", *p)) {
        ss << "\\" << *p;
      } else {
        ss << *p;
      }
      p++;
    }
    ss << '"';
    return ss.str();
  }

  std::string str() const {
    std::stringstream ss;
    if (t_ == unknown_type) {
      ss << "undefined";
    } else if (t_ == null_type) {
      ss << "null";
    } else if (t_ == boolean_type) {
      ss << (u_.b ? "true" : "false");
    } else if (t_ == number_type) {
      ss << double(u_.d);
    } else if (t_ == string_type) {
      ss << str(u_.s->c_str());
    } else if (const array* pa = as<array>()) {
      array::const_iterator i;
      ss << "[";
      // array a = get<array>();
      for (i = pa->begin(); i != pa->end(); i++) {
        if (i != pa->begin()) ss << ", ";
        ss << i->str();
      }
      ss << "]";
    } else if (auto po = as<object>()) {
      // object::const_iterator i;
      ss << "{";
      // object o = get<object>();
      for (size_t i = 0; i < po->size(); i++) {
        if (i > 0) ss << ", ";
        ss << "\"" << po->keys()[i] << "\"";

        value v;
        if (po->at(i, &v)) {
          ss << ": " << v.str();
        } else {
          // TODO: report error
          ss << ": null";
        }
      }
      ss << "}";
    }
    return ss.str();
  }
};

#define MINIJSON_SKIP(i)                           \
  while (*i && detail::my_strchr("\r\n \t", *i)) { \
    i++;                                           \
  }

template <typename Iter>
inline error parse_object(Iter& i, value& v) {
  object o;
  i++;
  MINIJSON_SKIP(i)
  if (!(*i)) {
    return corrupted_json_error;
  }
  if (*i != '\x7d') {
    while (*i) {
      value vk, vv;
      error e = parse_string(i, vk);
      if (e != no_error) return e;
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
      if (*i != ':') return invalid_token_error;
      i++;
      e = parse_any(i, vv);
      if (e != no_error) return e;

      auto ps = vk.as<std::string>();
      if (!ps) {
        return unknown_type_error;
      }

      if (o.count(*ps)) {
        return duplicated_key_error;
      }
      o.insert(*ps, vv);

      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
      if (*i == '\x7d') break;
      if (*i != ',') return invalid_token_error;
      i++;
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
#ifdef __MINIJSON_LIBERAL
      if (*i == '\x7d') break;
#endif
    }
  }
  v = value(o);
  i++;
  return no_error;
}

template <typename Iter>
inline error parse_array(Iter& i, value& v) {
  array a;
  i++;
  MINIJSON_SKIP(i)
  if (!(*i)) {
    return corrupted_json_error;
  }
  if (*i != ']') {
    while (*i) {
      value va;
      error e = parse_any(i, va);
      if (e != no_error) return e;
      a.push_back(va);
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
      if (*i == ']') break;
      if (*i != ',') return invalid_token_error;
      i++;
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
#ifdef __MINIJSON_LIBERAL
      if (*i == '\x7d') break;
#endif
    }
  }
  v = value(a);
  i++;
  return no_error;
}

template <typename Iter>
inline error parse_null(Iter& i, value& v) {
  Iter p = i;
  if (*i == 'n' && *(i + 1) == 'u' && *(i + 2) == 'l' && *(i + 3) == 'l') {
    i += 4;
    v = null_t();
  }
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return undefined_error;
  }
  return no_error;
}

template <typename Iter>
inline error parse_boolean(Iter& i, value& v) {
  Iter p = i;
  if (*i == 't' && *(i + 1) == 'r' && *(i + 2) == 'u' && *(i + 3) == 'e') {
    i += 4;
    v = static_cast<boolean>(true);
  } else if (*i == 'f' && *(i + 1) == 'a' && *(i + 2) == 'l' &&
             *(i + 3) == 's' && *(i + 4) == 'e') {
    i += 5;
    v = static_cast<boolean>(false);
  }
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return undefined_error;
  }
  return no_error;
}

template <typename Iter>
inline error parse_number(Iter& i, value& v) {
  Iter p = i;

  if (*i == '-') {
    i++;
  }

#define MINIJSON_IS_NUM(x) ('0' <= x && x <= '9')
#define MINIJSON_IS_ALNUM(x) \
  (('0' <= x && x <= '9') || ('a' <= x && x <= 'f') || ('A' <= x && x <= 'F'))
  if (*i == '0' && *(i + 1) == 'x' && MINIJSON_IS_ALNUM(*(i + 2))) {
    i += 3;
    while (MINIJSON_IS_ALNUM(*i)) i++;
    v = static_cast<number>(detail::from_chars(p));
  } else {
    while (MINIJSON_IS_NUM(*i)) i++;
    if (*i == '.') {
      i++;
      if (!MINIJSON_IS_NUM(*i)) {
        i = p;
        return invalid_token_error;
      }
      while (MINIJSON_IS_NUM(*i)) i++;
    }
    if (*i == 'e') {
      i++;
      if (!MINIJSON_IS_NUM(*i)) {
        i = p;
        return invalid_token_error;
      }
      while (MINIJSON_IS_NUM(*i)) i++;
    }
    v = static_cast<number>(detail::from_chars(p));
  }
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }
  return no_error;
}

template <typename Iter>
inline error parse_string(Iter& i, value& v) {
  if (*i != '"') return invalid_token_error;

  Iter s = i;
  char t = *i++;  // = '"'
  Iter p = i;

#if 0
  std::stringstream ss;
  while (*i && *i != t) {
    if (*i == '\\' && *(i + 1)) {
      i++;
      if (*i == 'n')
        ss << "\n";
      else if (*i == 'r')
        ss << "\r";
      else if (*i == 't')
        ss << "\t";
      else
        ss << *i;
    } else {
      ss << *i;
    }
    i++;
  }
#else
  // read until '"'
  while (*i && *i != t) {
    if (*i == '\\' && *(i + 1)) {
      i++;
    }
    i++;
  }

#endif
  if (!*i) return invalid_token_error;
  if (i < p) {
    return corrupted_json_error;
  }

#if 0
  v = std::string(p, size_t(i - p));

  i++;
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }

#else

  i++;
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }

  // include first and last '"' char
  std::string buf(s, size_t(i - s));

  detail::string_parser str_parser;
  str_parser.set_input(buf);

  if (!str_parser.scan_string()) {
    // TODO: error message
    // str_parser.error_message;
    return invalid_token_error;
  } else {
    v = str_parser.token_buffer;
  }

#endif

  return no_error;
}

template <typename Iter>
inline error parse_any(Iter& i, value& v) {
  MINIJSON_SKIP(i)
  if (*i == '\x7b') return parse_object(i, v);
  if (*i == '[') return parse_array(i, v);
  if (*i == 't' || *i == 'f') return parse_boolean(i, v);
  if (*i == 'n') return parse_null(i, v);
  if ((*i == '-') || ('0' <= *i && *i <= '9')) return parse_number(i, v);
  if (*i == '"') return parse_string(i, v);
  return invalid_token_error;
}

template <typename Iter>
inline error parse(Iter& i, value& v) {
  return parse_any(i, v);
}

#undef MINIJSON_SKIP

inline const char* errstr(error e) {
  const char* s = "unknown error";
  switch (e) {
    case no_error: {
      s = "no error";
      break;
    }
    case undefined_error: {
      s = "undefined";
      break;
    }
    case invalid_token_error: {
      s = "invalid token";
      break;
    }
    case unknown_type_error: {
      s = "unknown type";
      break;
    }
    case memory_allocation_error: {
      s = "memory allocation error";
      break;
    }
    case corrupted_json_error: {
      s = "input is corrupted";
      break;
    }
    case duplicated_key_error: {
      s = "duplicated key found";
      break;
    }
      // default: return "unknown error";
  }

  return s;
}

}  // namespace minijson

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_MINIJSON_H_
