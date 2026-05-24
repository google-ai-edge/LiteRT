#ifndef HAILO_MOCK_SDK_HPP_
#define HAILO_MOCK_SDK_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>

#define HAILO_SUCCESS 0
#define HAILO_DEFAULT_VSTREAM_TIMEOUT_MS 1000
#define HAILO_DEFAULT_QUEUE_SIZE 16

typedef int hailo_status;
typedef int hailo_format_type_t;
#define HAILO_FORMAT_TYPE_AUTO 0

namespace hailort {

class MemoryView {
public:
  MemoryView(void* addr, size_t size) : addr_(addr), size_(size) {}
  void* data() const { return addr_; }
  size_t size() const { return size_; }
private:
  void* addr_;
  size_t size_;
};

template <typename T>
class Expected {
public:
  Expected(hailo_status status) : status_(status), has_value_(false) {}
  Expected(T&& value) : status_(HAILO_SUCCESS), value_(std::move(value)), has_value_(true) {}
  Expected(const T& value) : status_(HAILO_SUCCESS), value_(value), has_value_(true) {}

  bool operator!() const { return !has_value_; }
  explicit operator bool() const { return has_value_; }

  hailo_status status() const { return status_; }
  T& value() { return value_; }
  const T& value() const { return value_; }

private:
  hailo_status status_;
  T value_;
  bool has_value_;
};

class Hef {
public:
  static Expected<Hef> create_from_buffer(const MemoryView& memory_view) {
    return Expected<Hef>(Hef());
  }
};

class ConfigureParams {};
struct hailo_vstream_params_t {};

class InputVStream {
public:
  size_t get_frame_size() const { return 1024; } // dummy
  hailo_status write(const MemoryView& buffer) { return HAILO_SUCCESS; }
};

class OutputVStream {
public:
  size_t get_frame_size() const { return 1024; } // dummy
  hailo_status read(const MemoryView& buffer) { return HAILO_SUCCESS; }
};

class ConfiguredNetworkGroup {
public:
  Expected<std::map<std::string, hailo_vstream_params_t>> make_input_vstream_params(
      bool, hailo_format_type_t, uint32_t, uint16_t) {
    std::map<std::string, hailo_vstream_params_t> params;
    params["input_0"] = hailo_vstream_params_t();
    return Expected<std::map<std::string, hailo_vstream_params_t>>(params);
  }
  Expected<std::map<std::string, hailo_vstream_params_t>> make_output_vstream_params(
      bool, hailo_format_type_t, uint32_t, uint16_t) {
    std::map<std::string, hailo_vstream_params_t> params;
    params["output_0"] = hailo_vstream_params_t();
    return Expected<std::map<std::string, hailo_vstream_params_t>>(params);
  }
};

class VDevice {
public:
  static Expected<std::unique_ptr<VDevice>> create() {
    return Expected<std::unique_ptr<VDevice>>(std::make_unique<VDevice>());
  }
  Expected<ConfigureParams> create_configure_params(Hef& hef) {
    return Expected<ConfigureParams>(ConfigureParams());
  }
  Expected<std::vector<std::shared_ptr<ConfiguredNetworkGroup>>> configure(Hef& hef, const ConfigureParams& params) {
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> groups;
    groups.push_back(std::make_shared<ConfiguredNetworkGroup>());
    return Expected<std::vector<std::shared_ptr<ConfiguredNetworkGroup>>>(groups);
  }
};

class VStream {
public:
  static Expected<std::vector<InputVStream>> create_input_vstreams(
      ConfiguredNetworkGroup& group, const std::map<std::string, hailo_vstream_params_t>& params) {
    std::vector<InputVStream> streams;
    for (size_t i = 0; i < params.size(); ++i) {
      streams.emplace_back();
    }
    return Expected<std::vector<InputVStream>>(std::move(streams));
  }
  static Expected<std::vector<OutputVStream>> create_output_vstreams(
      ConfiguredNetworkGroup& group, const std::map<std::string, hailo_vstream_params_t>& params) {
    std::vector<OutputVStream> streams;
    for (size_t i = 0; i < params.size(); ++i) {
      streams.emplace_back();
    }
    return Expected<std::vector<OutputVStream>>(std::move(streams));
  }
};

} // namespace hailort

#endif // HAILO_MOCK_SDK_HPP_
