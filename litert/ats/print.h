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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_PRINT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_PRINT_H_

#include <array>
#include <cstddef>
#include <list>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::testing {

// Utility base class for POD-like types that can be printed in various formats
// either individually or grouped together as a row of data.
template <typename... Fs>
class Printable {
 public:
  using Fields = std::tuple<Fs...>;
  static constexpr size_t kNumFields = sizeof...(Fs);
  using Keys = std::array<absl::string_view, kNumFields>;

  // Print the fields in a human readable format.
  template <typename Sink>
  void Print(Sink& sink) const {
    PrintImpl(sink, std::make_index_sequence<kNumFields>());
  }

  // Print the fields in CSV format.
  template <typename Sink>
  void CsvAppendHeader(Sink& sink) const {
    sink << absl::StrJoin(keys_, ",");
  }

  // Append a row to a CSV formatted string.
  template <typename Sink>
  void CsvAppendRow(Sink& sink) const {
    CsvAppendRowImpl(sink, std::make_index_sequence<kNumFields>());
  }

  virtual ~Printable() = default;

 protected:
  template <typename... Ks>
  explicit Printable(absl::string_view name, Ks... keys)
      : keys_({keys...}), name_(name) {}

 private:
  template <typename Sink, size_t... Is>
  void PrintImpl(Sink& sink, std::index_sequence<Is...>) const {
    const auto fields = GetFields();
    sink << absl::StreamFormat("\t%s\n", name_);
    ((sink << absl::StreamFormat("\t\t%s: %v\n", keys_[Is],
                                 std::get<Is>(fields))),
     ...);
  }

  template <typename Sink, size_t... Is>
  void CsvAppendRowImpl(Sink& sink, std::index_sequence<Is...>) const {
    const auto fields = GetFields();
    ((sink << absl::StreamFormat("%v%s", std::get<Is>(fields),
                                 Is == kNumFields - 1 ? "" : ",")),
     ...);
  }

  virtual Fields GetFields() const = 0;

  const Keys keys_;
  absl::string_view name_;
};

// A collection of printables that represent a row of data.
template <typename... Ps>
class PrintableRow {
 public:
  using Printables = std::tuple<const Ps&...>;

  // Print the row in a human readable format.
  template <typename Sink>
  void Print(Sink& sink) const {
    sink << absl::StreamFormat("%s\n", Name());
    PrintImpl(sink, std::make_index_sequence<sizeof...(Ps)>());
  }

  // Sink the header of a csv for this row.
  template <typename Sink>
  void CsvHeader(Sink& sink) const {
    CsvHeaderImpl(sink, std::make_index_sequence<sizeof...(Ps)>());
  }

  // Sink this as a row of data to a csv.
  template <typename Sink>
  void CsvRow(Sink& sink) const {
    CsvRowImpl(sink, std::make_index_sequence<sizeof...(Ps)>());
  }

  virtual ~PrintableRow() = default;

 private:
  template <typename Sink, size_t... Is>
  void PrintImpl(Sink& sink, std::index_sequence<Is...>) const {
    (std::get<Is>(GetPrintables()).Print(sink), ...);
  }

  template <typename Sink, size_t... Is>
  void CsvRowImpl(Sink& sink, std::index_sequence<Is...>) const {
    (
        [this, &sink]() {
          std::get<Is>(GetPrintables()).CsvAppendRow(sink);
          if (Is != sizeof...(Ps) - 1) {
            sink << ",";
          }
        }(),
        ...);
    sink << std::endl;
  }

  template <typename Sink, size_t... Is>
  void CsvHeaderImpl(Sink& sink, std::index_sequence<Is...>) const {
    (
        [this, &sink]() {
          std::get<Is>(GetPrintables()).CsvAppendHeader(sink);
          if (Is != sizeof...(Ps) - 1) {
            sink << ",";
          }
        }(),
        ...);
    sink << std::endl;
  }

  virtual Printables GetPrintables() const = 0;
  virtual std::string Name() const = 0;
};

// A collection of many rows of data that can be printed in a single CSV file.
template <typename P>
class PrintableCollection {
 public:
  // Print everything in a human readable format.s
  template <typename Sink>
  void Print(Sink& sink) const {
    static constexpr absl::string_view kLine = "====================";
    const auto name = Name();
    const auto header = absl::StrFormat("%s %s", name, kLine);
    const auto footer = absl::StrJoin(
        std::vector<absl::string_view>(kLine.size() + name.size() + 2, "="),
        "");
    sink << absl::StreamFormat("%s\n\n", header);
    for (const auto& row : rows_) {
      row.Print(sink);
      sink << std::endl;
    }
    sink << absl::StreamFormat("%s\n\n", footer);
  }

  // Print everything in a CSV format.
  template <typename Sink>
  void Csv(Sink& sink) const {
    if (rows_.empty()) {
      return;
    }
    rows_.front().CsvHeader(sink);
    for (const auto& row : rows_) {
      row.CsvRow(sink);
    }
  }

  // Create a new row of data.
  P& NewEntry() { return rows_.emplace_back(); }

  // Get all of the rows of data.
  const std::list<P>& Rows() const { return rows_; }

  virtual ~PrintableCollection() = default;

 private:
  virtual absl::string_view Name() const = 0;

  // Want stable references to pass around.
  std::list<P> rows_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_PRINT_H_
