// Copyright 2024 Google LLC.
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

#include "litert/tools/outliner/outliner_gui.h"

#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/tools/dump.h"
#include "litert/tools/outliner/outliner_util.h"

namespace litert::tools {

namespace {

const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* REVERSE = "\033[7m";
const char* CLEAR_ALL = "\033[2J\033[H";
const char* CLEAR_LINE = "\033[K";
const char* GREEN = "\033[32m";
const char* RED = "\033[31m";
const char* CYAN = "\033[36m";
const char* YELLOW = "\033[33m";
const char* GRAY = "\033[90m";
const char* BG_BLUE = "\033[44m";

std::string Truncate(absl::string_view s, size_t width) {
  if (s.length() <= width) return std::string(s);
  return absl::StrFormat("%s...", s.substr(0, width - 3));
}

char getch() {
  char buf = 0;
  struct termios old = {0};
  if (tcgetattr(0, &old) < 0) return 0;
  struct termios raw = old;
  raw.c_lflag &= ~ICANON;
  raw.c_lflag &= ~ECHO;
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &raw) < 0) return 0;
  if (read(0, &buf, 1) < 0) buf = 0;
  tcsetattr(0, TCSADRAIN, &old);
  return buf;
}

}  // namespace

OutlinerGui::OutlinerGui(LiteRtModelT& model) : model_(model) {}

litert::Expected<void> OutlinerGui::Run() {
  std::cout << CLEAR_ALL << std::flush;
  while (running_) {
    Render();
    HandleInput();
  }
  std::cout << CLEAR_ALL << "Returning to terminal...\n" << std::flush;
  return {};
}

void OutlinerGui::Render() {
  // Fixed vertical sections:
  // 1-3: Header
  // 4-28: Op List (25 rows)
  // 29-33: Properties Panel
  // 34: Search Line
  // 35-37: Boundaries & Messages
  // 38: Controls

  MoveCursor(1, 1);
  std::cout << BG_BLUE << BOLD << " LiteRT Graph Outliner " << RESET << " | "
            << "Subgraph " << current_sg_idx_ << "/"
            << model_.NumSubgraphs() - 1 << " | Total Ops: "
            << model_.Subgraphs()[current_sg_idx_]->Ops().size() << CLEAR_LINE
            << "\n";
  std::cout << GRAY
            << "j/k: Scroll | s: Start | e: End | /: Search | o: Outline | "
               "TAB: Subgraph | q: Quit"
            << RESET << CLEAR_LINE << "\n";
  std::cout << "---------------------------------------------------------------"
               "-----------------"
            << CLEAR_LINE << "\n";

  auto& sg = *model_.Subgraphs()[current_sg_idx_];
  auto ops = sg.Ops();
  const int kListHeight = 25;

  for (int i = 0; i < kListHeight; ++i) {
    int op_idx = scroll_offset_ + i;
    MoveCursor(4 + i, 1);
    std::cout << CLEAR_LINE;
    if (op_idx >= ops.size()) {
      continue;
    }

    auto* op = ops[op_idx];
    bool is_cursor = (op_idx == cursor_pos_);
    bool is_identified =
        std::find(identified_op_indices_.begin(), identified_op_indices_.end(),
                  op_idx) != identified_op_indices_.end();

    if (is_identified)
      std::cout << YELLOW << "┃ " << RESET;
    else
      std::cout << "  ";

    if (is_cursor) std::cout << REVERSE << CYAN << BOLD;

    std::cout << absl::StrFormat("[%3d] ", op->OpIndex());
    std::stringstream ss;
    litert::internal::Dump(*op, ss);
    std::cout << Truncate(ss.str(), 100);
    std::cout << RESET;
  }

  MoveCursor(29, 1);
  std::cout << "---------------------------------------------------------------"
               "-----------------"
            << CLEAR_LINE << "\n";
  if (cursor_pos_ < ops.size()) {
    auto* op = ops[cursor_pos_];
    std::cout << BOLD << CYAN << "Selected Op Details (Index " << op->OpIndex()
              << "):" << RESET << CLEAR_LINE << "\n";
    std::cout << "  In:  " << CLEAR_LINE;
    for (auto* t : op->Inputs())
      if (t) std::cout << Truncate(t->Name(), 40) << " ";
    std::cout << "\n  Out: " << CLEAR_LINE;
    for (auto* t : op->Outputs())
      if (t) std::cout << Truncate(t->Name(), 40) << " ";
    std::cout << "\n";
  } else {
    std::cout << "\n\n\n";
  }

  MoveCursor(34, 1);
  std::cout << "---------------------------------------------------------------"
               "-----------------"
            << CLEAR_LINE << "\n";
  if (search_mode_) {
    std::cout << BOLD << YELLOW << "SEARCH TENSOR: " << RESET << search_query_
              << "_" << CLEAR_LINE << "\n";
  } else {
    std::cout << BOLD << GREEN << "Outline Boundaries:" << RESET << CLEAR_LINE
              << "\n";
    std::cout << "  Starts: "
              << (final_options_.start_tensors.empty()
                      ? GRAY + std::string("(none)")
                      : RESET +
                            absl::StrJoin(final_options_.start_tensors, ", "))
              << CLEAR_LINE << "\n";
    std::cout << "  Ends:   "
              << (final_options_.end_tensors.empty()
                      ? GRAY + std::string("(none)")
                      : RESET + absl::StrJoin(final_options_.end_tensors, ", "))
              << CLEAR_LINE << "\n";
  }

  MoveCursor(38, 1);
  if (!error_message_.empty()) {
    std::cout << RED << BOLD << ">> ERROR: " << error_message_ << RESET
              << CLEAR_LINE;
  } else if (!status_message_.empty()) {
    std::cout << YELLOW << BOLD << ">> STATUS: " << status_message_ << RESET
              << CLEAR_LINE;
  } else {
    std::cout << CLEAR_LINE;
  }

  std::cout << std::flush;
}

void OutlinerGui::HandleInput() {
  char c = getch();

  if (search_mode_) {
    if (c == 27) {  // ESC
      search_mode_ = false;
      search_query_.clear();
    } else if (c == 127 || c == 8) {  // Backspace
      if (!search_query_.empty()) search_query_.pop_back();
    } else if (c == '\n' || c == '\r') {
      search_mode_ = false;
      // Already jumped to first match during typing, or could refine here.
    } else if (isprint(c)) {
      search_query_ += c;
      // Live jump to first match
      auto& sg = *model_.Subgraphs()[current_sg_idx_];
      for (size_t i = 0; i < sg.Ops().size(); ++i) {
        bool match = false;
        for (auto* t : sg.Ops()[i]->Inputs())
          if (t && absl::StrContains(t->Name(), search_query_)) match = true;
        for (auto* t : sg.Ops()[i]->Outputs())
          if (t && absl::StrContains(t->Name(), search_query_)) match = true;
        if (match) {
          cursor_pos_ = i;
          if (cursor_pos_ < scroll_offset_) scroll_offset_ = cursor_pos_;
          if (cursor_pos_ >= scroll_offset_ + 25)
            scroll_offset_ = cursor_pos_ - 12;
          break;
        }
      }
    }
    return;
  }

  error_message_ = "";
  status_message_ = "";
  auto& sg = *model_.Subgraphs()[current_sg_idx_];
  auto ops = sg.Ops();

  switch (c) {
    case 'k':  // Up
      if (cursor_pos_ > 0) cursor_pos_--;
      if (cursor_pos_ < scroll_offset_) scroll_offset_ = cursor_pos_;
      break;
    case 'j':  // Down
      if (cursor_pos_ < (int)ops.size() - 1) cursor_pos_++;
      if (cursor_pos_ >= scroll_offset_ + 25) scroll_offset_++;
      break;
    case '/':  // Search
      search_mode_ = true;
      search_query_ = "";
      break;
    case '\t':  // Tab
      current_sg_idx_ = (current_sg_idx_ + 1) % model_.NumSubgraphs();
      cursor_pos_ = 0;
      scroll_offset_ = 0;
      final_options_.start_tensors.clear();
      final_options_.end_tensors.clear();
      identified_op_indices_.clear();
      break;
    case 's': {  // Set Start
      auto* op = ops[cursor_pos_];
      if (!op->Outputs().empty()) {
        final_options_.start_tensors.push_back(
            std::string(op->Outputs()[0]->Name()));
        UpdateSelection();
      }
      break;
    }
    case 'e': {  // Set End
      auto* op = ops[cursor_pos_];
      if (!op->Outputs().empty()) {
        final_options_.end_tensors.push_back(
            std::string(op->Outputs()[0]->Name()));
        UpdateSelection();
      }
      break;
    }
    case 'c':  // Clear
      final_options_.start_tensors.clear();
      final_options_.end_tensors.clear();
      identified_op_indices_.clear();
      break;
    case 'q':
      running_ = false;
      break;
    case 'o':
      if (identified_op_indices_.empty()) {
        error_message_ = "Set boundaries [s] and [e] first!";
      } else {
        confirmed_ = true;
        running_ = false;
        final_subgraph_index_ = current_sg_idx_;
      }
      break;
  }
}

void OutlinerGui::UpdateSelection() {
  if (final_options_.start_tensors.empty() ||
      final_options_.end_tensors.empty()) {
    return;
  }
  auto& sg = *model_.Subgraphs()[current_sg_idx_];
  auto res = IdentifyIdentifiedOps(sg, final_options_.start_tensors,
                                   final_options_.end_tensors);
  if (!res) {
    error_message_ = res.Error().Message();
    identified_op_indices_.clear();
  } else {
    identified_op_indices_.clear();
    for (auto* op : *res) {
      identified_op_indices_.push_back(op->OpIndex());
    }
    status_message_ = absl::StrFormat("Selected %d ops. Press [o] to outline.",
                                      identified_op_indices_.size());
  }
}

void OutlinerGui::ClearScreen() { std::cout << CLEAR_ALL << std::flush; }

void OutlinerGui::MoveCursor(int row, int col) {
  std::cout << "\033[" << row << ";" << col << "H" << std::flush;
}

}  // namespace litert::tools
