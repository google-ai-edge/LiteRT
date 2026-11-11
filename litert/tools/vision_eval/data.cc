// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/data.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef FAILED
#undef FAILED
#endif
#ifdef GetMessage
#undef GetMessage
#endif
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#endif

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/tools/vision_eval/image.h"
#include "google/protobuf/descriptor.h"  // from @com_google_protobuf
#include "google/protobuf/message.h"  // from @com_google_protobuf
#include "google/protobuf/text_format.h"  // from @com_google_protobuf

namespace litert::vision {
namespace {
constexpr size_t kMaxTensorBytes = 512u * 1024u * 1024u;

bool ModelType(ElementType type) {
  return type == FLOAT32 || type == INT8 || type == UINT8;
}

Expected<void> SetFlag(google::protobuf::Message& message, const std::string& key,
                       const std::string& value) {
  using Field = google::protobuf::FieldDescriptor;
  const auto* field = message.GetDescriptor()->FindFieldByName(key);
  if (!field || field->is_repeated() ||
      field->cpp_type() == Field::CPPTYPE_MESSAGE) {
    return Invalid("Unknown scalar flag: --" + key);
  }
  const auto* reflection = message.GetReflection();
  if (field->cpp_type() == Field::CPPTYPE_STRING) {
    reflection->SetString(&message, field, value);
    return {};
  }
  if (field->cpp_type() == Field::CPPTYPE_BOOL) {
    if (value != "true" && value != "false" && value != "1" && value != "0") {
      return Invalid("Expected boolean: --" + key);
    }
    reflection->SetBool(&message, field, value == "true" || value == "1");
    return {};
  }
  if (field->cpp_type() == Field::CPPTYPE_ENUM) {
    auto name = value;
    absl::c_transform(name, name.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    if (key == "summary_format") {
      if (name == "TEXTPROTO" || name == "TEXT") name = "TEXT_PROTO";
      if (name == "BINARYPROTO" || name == "BINARY") name = "BINARY_PROTO";
    }
    const auto* enumeration = field->enum_type()->FindValueByName(name);
    if (!enumeration) return Invalid("Invalid enum: --" + key + "=" + value);
    reflection->SetEnum(&message, field, enumeration);
    return {};
  }
  if (!google::protobuf::TextFormat::ParseFieldValueFromString(value, field,
                                                              &message)) {
    return Invalid("Invalid value: --" + key);
  }
  return {};
}
}  // namespace

Error Invalid(const std::string& message) {
  return Error(kLiteRtStatusErrorInvalidArgument, message);
}

namespace {

void CloseNativeHandle(intptr_t& handle) {
#if defined(_WIN32)
  if (handle != -1) {
    CloseHandle(reinterpret_cast<HANDLE>(handle));
    handle = -1;
  }
#else
  // NOMUTANTS(ROR,SBR): Valid test fds are >= 3 because stdio occupies 0..2.
  if (handle >= 0) {
    close(static_cast<int>(handle));
    handle = -1;
  }
#endif
}

Expected<size_t> ReadRawChunk(intptr_t handle, uint8_t* dst, size_t max_len) {
#if defined(_WIN32)
  constexpr DWORD kMaxChunk = 1u << 30;
  const DWORD to_read =
      static_cast<DWORD>(std::min<size_t>(max_len, kMaxChunk));
  DWORD bytes_read = 0;
  if (!ReadFile(reinterpret_cast<HANDLE>(handle), dst, to_read, &bytes_read,
                nullptr)) {
    return Invalid("Cannot read file");
  }
  return static_cast<size_t>(bytes_read);
#else
  constexpr size_t kMaxChunk = 1u << 30;
  const size_t to_read = std::min(max_len, kMaxChunk);
  ssize_t n = 0;
  // NOMUTANTS(LCR): Signal-interrupted read() retry is not deterministically
  // triggerable in unit tests.
  do {
    n = read(static_cast<int>(handle), dst, to_read);
  } while (n < 0 && errno == EINTR);
  // NOMUTANTS(ROR): OS read() failure on an open regular-file fd requires fault
  // injection.
  if (n < 0) return Invalid("Cannot read file");
  return static_cast<size_t>(n);
#endif
}

Expected<void> WriteRawAll(intptr_t handle, const uint8_t* src, size_t size) {
  size_t total = 0;
  while (total < size) {
#if defined(_WIN32)
    constexpr DWORD kMaxChunk = 1u << 30;
    const DWORD to_write =
        static_cast<DWORD>(std::min<size_t>(size - total, kMaxChunk));
    DWORD written = 0;
    if (!WriteFile(reinterpret_cast<HANDLE>(handle), src + total, to_write,
                   &written, nullptr) ||
        written == 0) {
      return Invalid("Cannot write file");
    }
    total += written;
#else
    constexpr size_t kMaxChunk = 1u << 30;
    const size_t to_write = std::min(size - total, kMaxChunk);
    ssize_t n = 0;
    // NOMUTANTS(LCR): Signal-interrupted write() retry is not deterministically
    // triggerable in unit tests.
    do {
      n = write(static_cast<int>(handle), src + total, to_write);
    } while (n < 0 && errno == EINTR);
    // NOMUTANTS(ROR): OS write() failure on an open regular-file fd requires
    // disk-full or fault injection.
    if (n <= 0) return Invalid("Cannot write file");
    total += static_cast<size_t>(n);
#endif
  }
  return {};
}

}  // namespace

FileReader::FileReader(FileReader&& other) noexcept
    : handle_(other.handle_),
      buffer_(std::move(other.buffer_)),
      pos_(other.pos_),
      end_(other.end_) {
  other.handle_ = -1;
  other.pos_ = 0;
  other.end_ = 0;
}

FileReader& FileReader::operator=(FileReader&& other) noexcept {
  // NOMUTANTS(ROR): Self-move guard; Clang rejects literal self-move in tests.
  if (this != &other) {
    CloseNativeHandle(handle_);
    handle_ = other.handle_;
    buffer_ = std::move(other.buffer_);
    pos_ = other.pos_;
    end_ = other.end_;
    other.handle_ = -1;
    other.pos_ = 0;
    other.end_ = 0;
  }
  return *this;
}

FileReader::~FileReader() { CloseNativeHandle(handle_); }

Expected<FileReader> FileReader::Open(const std::filesystem::path& path) {
  FileReader reader;
#if defined(_WIN32)
  HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Invalid("Cannot open file: " + PathToUtf8(path));
  }
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(handle, &info) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    CloseHandle(handle);
    return Invalid("Cannot open directory as file: " + PathToUtf8(path));
  }
  reader.handle_ = reinterpret_cast<intptr_t>(handle);
#else
  int fd = -1;
  // NOMUTANTS(LCR): Signal-interrupted open() retry is not deterministically
  // triggerable in unit tests.
  do {
    fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return Invalid("Cannot open file: " + PathToUtf8(path));
  }
  struct stat st;
  // NOMUTANTS(LCR): fstat() failure on an open fd requires kernel/FS fault
  // injection.
  if (fstat(fd, &st) != 0 || S_ISDIR(st.st_mode)) {
    close(fd);
    return Invalid("Cannot open directory as file: " + PathToUtf8(path));
  }
  reader.handle_ = fd;
#endif
  reader.buffer_.resize(kBufferSize);
  return reader;
}

Expected<size_t> FileReader::Read(void* data, size_t size) {
  if (handle_ == -1) return Invalid("FileReader is not open");
  if (size == 0) return 0;
  if (!data) return Invalid("Null buffer for non-empty read");
  auto* out = static_cast<uint8_t*>(data);
  size_t copied = 0;
  while (copied < size) {
    const size_t buffered = end_ - pos_;
    if (buffered > 0) {
      const size_t take = std::min(size - copied, buffered);
      std::memcpy(out + copied, buffer_.data() + pos_, take);
      pos_ += take;
      copied += take;
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        const size_t n, ReadRawChunk(handle_, buffer_.data(), buffer_.size()));
    pos_ = 0;
    end_ = n;
    if (n == 0) break;
  }
  return copied;
}

FileWriter::FileWriter(FileWriter&& other) noexcept
    : handle_(other.handle_), buffer_(std::move(other.buffer_)) {
  other.handle_ = -1;
}

FileWriter& FileWriter::operator=(FileWriter&& other) noexcept {
  // NOMUTANTS(ROR): Self-move guard; Clang rejects literal self-move in tests.
  if (this != &other) {
    CloseNativeHandle(handle_);
    handle_ = other.handle_;
    buffer_ = std::move(other.buffer_);
    other.handle_ = -1;
  }
  return *this;
}

FileWriter::~FileWriter() { CloseNativeHandle(handle_); }

Expected<FileWriter> FileWriter::Open(const std::filesystem::path& path) {
  FileWriter writer;
#if defined(_WIN32)
  HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Invalid("Cannot write: " + PathToUtf8(path));
  }
  writer.handle_ = reinterpret_cast<intptr_t>(handle);
#else
  int fd = -1;
  // NOMUTANTS(LCR): Signal-interrupted open() retry is not deterministically
  // triggerable in unit tests.
  do {
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return Invalid("Cannot write: " + PathToUtf8(path));
  }
  writer.handle_ = fd;
#endif
  writer.buffer_.reserve(kBufferSize);
  return writer;
}

Expected<void> FileWriter::FlushBuffer() {
  if (buffer_.empty()) return {};
  const auto status = WriteRawAll(handle_, buffer_.data(), buffer_.size());
  buffer_.clear();
  return status;
}

Expected<void> FileWriter::Write(const void* data, size_t size) {
  if (handle_ == -1) return Invalid("FileWriter is not open");
  if (size == 0) return {};
  if (!data) return Invalid("Null data for non-empty write");
  const auto* bytes = static_cast<const uint8_t*>(data);
  // NOMUTANTS(ROR): Direct-write threshold is a buffering optimization; output
  // bytes are identical.
  if (size >= kBufferSize) {
    LITERT_RETURN_IF_ERROR(FlushBuffer());
    return WriteRawAll(handle_, bytes, size);
  }
  // NOMUTANTS(AOR,ROR): Buffer flush threshold is a buffering optimization;
  // output bytes are identical.
  if (buffer_.size() + size > kBufferSize) {
    LITERT_RETURN_IF_ERROR(FlushBuffer());
  }
  buffer_.insert(buffer_.end(), bytes, bytes + size);
  return {};
}

Expected<void> FileWriter::Close() {
  if (handle_ == -1) return Invalid("FileWriter is not open");
  const auto flush_status = FlushBuffer();
#if defined(_WIN32)
  const bool closed = CloseHandle(reinterpret_cast<HANDLE>(handle_)) != 0;
  handle_ = -1;
  if (!flush_status) return flush_status;
  if (!closed) return Invalid("Cannot close file");
#else
  const int rc = close(static_cast<int>(handle_));
  handle_ = -1;
  if (!flush_status) return flush_status;
  // NOMUTANTS(LCR): close() failure on a regular-file fd requires kernel/NFS
  // fault injection.
  if (rc != 0 && errno != EINTR) return Invalid("Cannot close file");
#endif
  return {};
}

Expected<std::vector<uint8_t>> ReadFileBytes(const std::filesystem::path& path,
                                             size_t max_bytes) {
#if defined(_WIN32)
  HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Invalid("Cannot open file: " + PathToUtf8(path));
  }
  intptr_t guard_handle = reinterpret_cast<intptr_t>(handle);
  struct Guard {
    intptr_t& h;
    ~Guard() { CloseNativeHandle(h); }
  } guard{guard_handle};
  BY_HANDLE_FILE_INFORMATION info;
  if (GetFileType(handle) != FILE_TYPE_DISK ||
      !GetFileInformationByHandle(handle, &info) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    return Invalid("Not a regular file: " + PathToUtf8(path));
  }
  const uint64_t file_size =
      (static_cast<uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
  if (file_size > max_bytes) {
    return Invalid("File exceeds size limit: " + PathToUtf8(path));
  }
  std::vector<uint8_t> buffer(static_cast<size_t>(file_size));
  size_t total = 0;
  while (total < buffer.size()) {
    LITERT_ASSIGN_OR_RETURN(const size_t n,
                            ReadRawChunk(guard_handle, buffer.data() + total,
                                         buffer.size() - total));
    if (n == 0) return Invalid("Cannot read file: " + PathToUtf8(path));
    total += n;
  }
  return buffer;
#else
  int fd = -1;
  // NOMUTANTS(LCR): Signal-interrupted open() retry is not deterministically
  // triggerable in unit tests.
  do {
    fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return Invalid("Cannot open file: " + PathToUtf8(path));
  }
  intptr_t guard_handle = fd;
  struct Guard {
    intptr_t& h;
    ~Guard() { CloseNativeHandle(h); }
  } guard{guard_handle};
  struct stat st;
  // NOMUTANTS(LCR): fstat() failure and negative st_size on an open fd require
  // kernel/FS fault injection.
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
    return Invalid("Not a regular file: " + PathToUtf8(path));
  }
  if (static_cast<uint64_t>(st.st_size) > max_bytes) {
    return Invalid("File exceeds size limit: " + PathToUtf8(path));
  }
  std::vector<uint8_t> buffer(static_cast<size_t>(st.st_size));
  size_t total = 0;
  while (total < buffer.size()) {
    LITERT_ASSIGN_OR_RETURN(const size_t n,
                            ReadRawChunk(guard_handle, buffer.data() + total,
                                         buffer.size() - total));
    // NOMUTANTS(ROR): Concurrent truncation at EOF mid-read requires a racing
    // writer.
    if (n == 0) return Invalid("Cannot read file: " + PathToUtf8(path));
    total += n;
  }
  return buffer;
#endif
}

Expected<std::string> ReadFileUntilEof(const std::filesystem::path& path,
                                       size_t max_bytes) {
  LITERT_ASSIGN_OR_RETURN(FileReader reader, FileReader::Open(path));
  std::string out;
  char chunk[4096];
  while (true) {
    LITERT_ASSIGN_OR_RETURN(const size_t n, reader.Read(chunk, sizeof(chunk)));
    if (n == 0) return out;
    if (n > max_bytes - out.size()) {
      return Invalid("File exceeds size limit: " + PathToUtf8(path));
    }
    out.append(chunk, n);
  }
}

Expected<void> ReadProto(const std::filesystem::path& path,
                         google::protobuf::Message& message) {
  auto raw = ReadFileBytes(path, 64u * 1024u * 1024u);
  if (!raw) {
    return Invalid("Cannot read proto or file exceeds 64 MiB: " +
                   PathToUtf8(path));
  }
  const absl::string_view bytes(reinterpret_cast<const char*>(raw->data()),
                                raw->size());
  bool parsed = false;
  const auto ext = path.extension();
  if (ext == ".binpb") {
    parsed = message.ParseFromString(bytes);
  } else if (ext == ".txtpb") {
    parsed = google::protobuf::TextFormat::ParseFromString(bytes, &message);
  } else {
    return Invalid("Proto filenames must end in .txtpb or .binpb: " +
                   PathToUtf8(path));
  }
  if (!parsed || !message.IsInitialized()) {
    return Invalid("Invalid protobuf: " + PathToUtf8(path));
  }
  return {};
}

Expected<void> WriteBytes(const std::filesystem::path& path, const void* data,
                          size_t size) {
  if (path.empty()) return Invalid("Cannot write empty path");
  if (size > 0 && !data) return Invalid("Null data for non-empty write");
  const auto parent = path.parent_path();
  std::error_code ec;
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec || !std::filesystem::is_directory(parent, ec) || ec) {
      return Invalid("Cannot create directory: " + PathToUtf8(parent));
    }
  }
  // The run lock serializes writers. Readers see only complete files.
  auto temporary = path;
  temporary += ".tmp";
  {
    LITERT_ASSIGN_OR_RETURN(FileWriter out, FileWriter::Open(temporary));
    // NOMUTANTS(UOI): Mid-write OS I/O failure after open() requires disk-full
    // or fault injection.
    if (!out.Write(data, size)) {
      (void)out.Close();
      std::filesystem::remove(temporary, ec);
      return Invalid("Cannot write: " + PathToUtf8(temporary));
    }
    // NOMUTANTS(UOI): OS flush/close failure after open() requires disk-full or
    // fault injection.
    if (!out.Close()) {
      std::filesystem::remove(temporary, ec);
      return Invalid("Cannot write: " + PathToUtf8(temporary));
    }
  }
  std::filesystem::rename(temporary, path, ec);
  if (ec) {
    std::error_code remove_ec;
    std::filesystem::remove(temporary, remove_ec);
    return Invalid("Cannot rename temporary file to: " + PathToUtf8(path));
  }
  return {};
}

Expected<void> WriteProto(const std::filesystem::path& path,
                          const google::protobuf::Message& message,
                          ProtoFormat format) {
  std::string bytes;
  const bool ok = format == TEXT_PROTO
                      ? google::protobuf::TextFormat::PrintToString(message, &bytes)
                      : message.SerializeToString(&bytes);
  if (!ok) return Invalid("Cannot serialize protobuf");
  return WriteBytes(path, bytes.data(), bytes.size());
}

Expected<size_t> TensorBytes(const TensorSpec& spec) {
  size_t size;
  switch (spec.type()) {
    case FLOAT32:
    case INT32:
      size = 4;
      break;
    case INT8:
    case UINT8:
    case BOOL:
      size = 1;
      break;
    default:
      return Invalid("Unknown tensor element type");
  }
  if (spec.has_scale()) {
    const int low = spec.type() == INT8 ? -128 : 0;
    const int high = spec.type() == INT8 ? 127 : 255;
    if ((spec.type() != INT8 && spec.type() != UINT8) ||
        !std::isfinite(spec.scale()) || spec.scale() <= 0 ||
        !std::isfinite(1.0f / spec.scale()) || spec.zero_point() < low ||
        spec.zero_point() > high) {
      return Invalid("Invalid tensor quantization");
    }
  }
  if (spec.shape_size() > 8) return Invalid("Tensor rank exceeds eight");
  for (auto dim : spec.shape()) {
    if (dim <= 0 || size > kMaxTensorBytes / static_cast<size_t>(dim)) {
      return Invalid("Dynamic/empty/oversized tensor shape");
    }
    size *= dim;
  }
  return size;
}

Expected<std::vector<uint8_t>> ReadTensor(const std::filesystem::path& path,
                                          const TensorSpec& spec) {
  LITERT_ASSIGN_OR_RETURN(auto count, TensorBytes(spec));
  auto bytes = ReadFileBytes(path, count);
  if (!bytes || bytes->size() != count) {
    return Invalid("Tensor byte count mismatch: " + PathToUtf8(path));
  }
  return bytes;
}

Expected<std::vector<float>> ReadFloats(const std::filesystem::path& path,
                                        const TensorSpec& spec) {
  if (spec.type() != FLOAT32) return Invalid("Expected FLOAT32 predictions");
  LITERT_ASSIGN_OR_RETURN(auto bytes, ReadTensor(path, spec));
  std::vector<float> result(bytes.size() / sizeof(float));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  for (auto f : result) {
    if (!std::isfinite(f)) {
      return Invalid("Tensor contains NaN or infinity: " + PathToUtf8(path));
    }
  }
  return result;
}

Expected<FileIdentity> Identify(const std::filesystem::path& path) {
  if (path.empty()) return Invalid("Empty file path");
  std::error_code ec;
  const auto absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    return Invalid("Cannot resolve path: " + PathToUtf8(path));
  }
  const bool regular = std::filesystem::is_regular_file(absolute, ec);
  if (ec || !regular) {
    return Invalid("Missing regular file: " + PathToUtf8(absolute));
  }
  FileIdentity file;
  file.set_path(PathToUtf8(absolute));
  return file;
}

Expected<void> VerifyFile(const FileIdentity& file) {
  if (file.path().empty()) {
    return Invalid("Incomplete file identity");
  }
  std::error_code ec;
  const bool regular =
      std::filesystem::is_regular_file(Utf8ToPath(file.path()), ec);
  if (ec || !regular) {
    return Invalid("Missing file: " + file.path());
  }
  return {};
}

bool SafeName(const std::string& value) {
  return !value.empty() && value.size() <= 128 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-';
         });
}

std::filesystem::path Resolve(const std::filesystem::path& root,
                              const std::filesystem::path& path) {
  if (root.empty() || path.empty()) return root.empty() ? path : root;
  return root / path;
}

Expected<ImagePreprocessing> ImageOptions(const Preprocessing& p) {
  ImagePreprocessing out;
  out.interpolation = p.interpolation();
  out.resize = p.resize();
  out.crop_h = p.crop_h();
  out.crop_w = p.crop_w();
  out.antialias = p.antialias();
  if (p.mean_size() && p.mean_size() != 3) {
    return Invalid("Expected three channel means");
  }
  if (p.stddev_size() && p.stddev_size() != 3) {
    return Invalid("Expected three channel standard deviations");
  }
  for (int i = 0; i < 3; ++i) {
    if (p.mean_size()) out.mean[i] = p.mean(i);
    if (p.stddev_size()) out.stddev[i] = p.stddev(i);
  }
  LITERT_RETURN_IF_ERROR(ValidatePreprocessing(out));
  return out;
}

Expected<void> ValidateManifest(const DatasetManifest& manifest, Task task) {
  if (manifest.version() != 1 || manifest.samples_size() == 0) {
    return Invalid("Expected version-1 manifest with samples");
  }
  std::set<std::string> ids, input_names, output_names;
  for (const auto& input : manifest.inputs()) {
    if (!SafeName(input.name()) || !input_names.insert(input.name()).second) {
      return Invalid("Duplicate or invalid input name");
    }
    if (!ModelType(input.type())) {
      return Invalid("Model inputs require FLOAT32, INT8 or UINT8");
    }
    LITERT_RETURN_IF_ERROR(TensorBytes(input));
  }
  for (const auto& output : manifest.outputs()) {
    const auto& tensor = output.tensor();
    const auto& scoring = output.scoring();
    if (!SafeName(tensor.name()) || !output_names.insert(tensor.name()).second) {
      return Invalid("Duplicate or invalid output name");
    }
    if (!ModelType(tensor.type())) {
      return Invalid("Model outputs require FLOAT32, INT8 or UINT8");
    }
    LITERT_RETURN_IF_ERROR(TensorBytes(tensor));
    const auto selected = task == AUTO ? scoring.task() : task;
    if (selected == AUTO) return Invalid("An output needs an evaluation task");
    if (selected == SEGMENTATION &&
        (scoring.classes() <= 0 || scoring.classes() > 4096)) {
      return Invalid("Segmentation class count must be 1..4096");
    }
    if (selected == DEPTH &&
        (!std::isfinite(scoring.min_depth()) ||
         !std::isfinite(scoring.max_depth()) ||
         !(0 < scoring.min_depth() && scoring.min_depth() < scoring.max_depth()))) {
      return Invalid("Invalid depth bounds");
    }
  }
  for (const auto& sample : manifest.samples()) {
    if (!SafeName(sample.id()) || !ids.insert(sample.id()).second) {
      return Invalid("Sample IDs must be unique and filename safe");
    }
    if (sample.has_image()) {
      if (sample.image().empty() || sample.inputs_size() || !sample.has_label() ||
          sample.label() < 0 || (task != AUTO && task != CLASSIFICATION)) {
        return Invalid("Image samples require classification labels and no raw inputs");
      }
    } else {
      std::set<std::string> names;
      for (const auto& input : sample.inputs()) {
        if (!input_names.count(input.name()) || input.file().path().empty() ||
            !names.insert(input.name()).second) {
          return Invalid("Unknown, duplicate or missing prepared input");
        }
      }
      if (names != input_names || names.empty()) {
        return Invalid("Prepared sample does not bind every input");
      }
    }
    std::set<std::string> targets;
    for (const auto& target : sample.targets()) {
      if (!output_names.count(target.name()) ||
          !targets.insert(target.name()).second || target.file().path().empty()) {
        return Invalid("Invalid target binding");
      }
      LITERT_RETURN_IF_ERROR(TensorBytes(target.tensor()));
    }
    if (!sample.has_label() && targets.empty()) {
      return Invalid("Sample has no labels or targets");
    }
  }
  return {};
}

Expected<DatasetManifest> LoadManifest(const EvaluationConfig& config) {
  DatasetManifest result;
  const auto manifest_path = Utf8ToPath(config.manifest());
  if (manifest_path.extension() == ".tsv") {
    LITERT_ASSIGN_OR_RETURN(
        auto rows,
        ReadSamples(manifest_path, Utf8ToPath(config.image_root()), 0, 0));
    for (const auto& row : rows) {
      std::error_code ec;
      const auto image_path = std::filesystem::absolute(row.path, ec);
      if (ec) {
        return Invalid("Cannot resolve image path: " + PathToUtf8(row.path));
      }
      auto* sample = result.add_samples();
      sample->set_id(std::to_string(row.id));
      sample->set_label(row.label);
      sample->set_image(PathToUtf8(image_path));
    }
  } else {
    LITERT_RETURN_IF_ERROR(ReadProto(manifest_path, result));
  }
  LITERT_RETURN_IF_ERROR(ValidateManifest(result, config.task()));
  const size_t begin = static_cast<size_t>(config.start_sample());
  if (begin >= static_cast<size_t>(result.samples_size())) {
    return Invalid("Sample start is out of range");
  }
  if (config.max_samples() &&
      begin + config.max_samples() < static_cast<size_t>(result.samples_size())) {
    result.mutable_samples()->DeleteSubrange(
        begin + config.max_samples(),
        result.samples_size() - begin - config.max_samples());
  }
  if (begin) result.mutable_samples()->DeleteSubrange(0, begin);
  return result;
}

Expected<void> ValidateConfig(const EvaluationConfig& config) {
  if (config.manifest().empty() || config.output_dir().empty() ||
      (!config.score_only() && config.model().empty())) {
    return Invalid("manifest, output_dir and an inference model are required");
  }
  if (config.score_only() &&
      (config.predictions_dir().empty() || !config.baseline_model().empty())) {
    return Invalid("Offline scoring requires predictions_dir and no baseline_model");
  }
  if (config.signature_index() < 0 || config.warmup_runs() < 0 ||
      config.start_sample() < 0 || config.max_samples() < 0) {
    return Invalid("Negative signature, warmup or sample selection");
  }
  if (!std::isfinite(config.atol()) || !std::isfinite(config.rtol()) ||
      config.atol() < 0 || config.rtol() < 0 ||
      !std::isfinite(config.max_code_error()) ||
      !std::isfinite(config.mean_code_error()) ||
      config.max_code_error() < 0 || config.mean_code_error() < 0) {
    return Invalid("Invalid numerical tolerance");
  }
  if ((config.require_output_parity() || config.require_prediction_agreement()) &&
      config.baseline_model().empty() && config.reference_dir().empty()) {
    return Invalid("Comparison gates require a baseline or reference directory");
  }
  if (config.baseline_accelerator() != "inherit" &&
      config.baseline_accelerator() != "cpu") {
    return Invalid("baseline_accelerator must be inherit or cpu");
  }
  if (config.baseline_accelerator() == "cpu" && config.baseline_model().empty()) {
    return Invalid("CPU comparison requires baseline_model");
  }
  if (config.baseline_num_threads() <= 0) {
    return Invalid("baseline_num_threads must be positive");
  }
  if (!config.reference_dir().empty() && !config.baseline_model().empty()) {
    return Invalid("Select baseline_model or reference_dir, not both");
  }
  LITERT_RETURN_IF_ERROR(ImageOptions(config.preprocessing()));
  return {};
}

Expected<EvaluationConfig> ParseArgs(int argc, char** argv) {
  EvaluationConfig config;
  std::set<std::string> seen;
  // Load the whole config before applying explicit scalar flags, regardless of
  // where --config appears in the command line.
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--config=", 0) == 0) {
      if (!seen.insert("config").second) return Invalid("Repeated --config");
      LITERT_RETURN_IF_ERROR(ReadProto(Utf8ToPath(arg.substr(9)), config));
    }
  }
  seen.clear();
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") continue;
    const auto equal = arg.find('=');
    if (arg.rfind("--", 0) != 0 || equal == std::string::npos) {
      return Invalid("Expected --name=value: " + arg);
    }
    const auto name = arg.substr(2, equal - 2);
    const auto value = arg.substr(equal + 1);
    if (name == "config") continue;
    if (!seen.insert(name).second) return Invalid("Repeated flag: --" + name);
    auto* message = static_cast<google::protobuf::Message*>(&config);
    if (!config.GetDescriptor()->FindFieldByName(name)) {
      if (config.backend().GetDescriptor()->FindFieldByName(name)) {
        message = config.mutable_backend();
      } else if (config.preprocessing().GetDescriptor()->FindFieldByName(name)) {
        message = config.mutable_preprocessing();
      }
    }
    LITERT_RETURN_IF_ERROR(SetFlag(*message, name, value));
  }
  LITERT_RETURN_IF_ERROR(ValidateConfig(config));
  return config;
}

void PrintUsage() {
  std::cerr
      << "litert_vision_eval --model=model.tflite --manifest=manifest.txtpb "
         "--output_dir=results [options]\n"
      << "  --config=run.txtpb|run.binpb  CLI values override config fields\n"
      << "  --task=auto|classification|segmentation|depth|normals\n"
      << "  --summary_format=textproto|binaryproto\n"
      << "  --dump_predictions=true --dump_inputs=true\n"
      << "  --accelerator=cpu|gpu|npu --require_full_acceleration=true\n"
      << "  --baseline_model=reference.tflite --baseline_accelerator=inherit|cpu\n"
      << "  --score_only=true --predictions_dir=saved --reference_dir=reference\n"
      << "  --start_sample=0 --max_samples=3 --warmup_runs=0\n"
      << "  Backend and preprocessing scalar fields are also CLI flags.\n"
      << "  See litert/tools/vision_eval/README.md for tensor contracts.\n";
}
}  // namespace litert::vision
