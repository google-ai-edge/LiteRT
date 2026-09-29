/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "litert/vendors/aml/compiler/aml_dispatch_info.h"

#include <sstream>

std::string SerializeDispatchInfo(const AML_Dispatch_Info &dispatch_info)
{
    std::ostringstream oss(std::ios::binary);

    auto write_string = [&](const std::string &s)
    {
        size_t len = s.size();
        oss.write((char *)&len, sizeof(len));
        oss.write(s.data(), len);
    };

    auto write_string_vector = [&](const std::vector<std::string> &values)
    {
        size_t num_values = values.size();
        oss.write((char *)&num_values, sizeof(num_values));
        for (const auto &value : values)
        {
            write_string(value);
        }
    };

    write_string(dispatch_info.model_path);
    write_string(dispatch_info.model_names);
    write_string(dispatch_info.graph_names);
    write_string_vector(dispatch_info.graph_inputs);
    write_string_vector(dispatch_info.graph_outputs);
    oss.write(reinterpret_cast<const char *>(&dispatch_info.subgraph_idx),
              sizeof(dispatch_info.subgraph_idx));

    const size_t adla_bin_size = dispatch_info.adla_bin_size > 0
                                     ? dispatch_info.adla_bin_size
                                     : dispatch_info.adla_bin.size();
    oss.write(reinterpret_cast<const char *>(&adla_bin_size), sizeof(adla_bin_size));
    if (adla_bin_size > 0)
    {
        oss.write(reinterpret_cast<const char *>(dispatch_info.adla_bin.data()),
                  adla_bin_size);
    }

    return oss.str();
}

AML_Dispatch_Info DeserializeDispatchInfo(const std::string &buffer)
{
    AML_Dispatch_Info dispatch_info;
    std::istringstream iss(buffer, std::ios::binary);

    auto read_string = [&](std::string &s)
    {
        size_t len;
        iss.read((char *)&len, sizeof(len));
        s.resize(len);
        if (len > 0)
        {
            iss.read(&s[0], len);
        }
    };

    auto read_string_vector = [&](std::vector<std::string> &values)
    {
        size_t num_values;
        iss.read((char *)&num_values, sizeof(num_values));
        values.resize(num_values);
        for (auto &value : values)
        {
            read_string(value);
        }
    };

    read_string(dispatch_info.model_path);
    read_string(dispatch_info.model_names);
    read_string(dispatch_info.graph_names);
    read_string_vector(dispatch_info.graph_inputs);
    read_string_vector(dispatch_info.graph_outputs);

    // Prefer remaining-byte checks over eof(): eof() is only set after a failed
    // read-past-end, so it is unreliable for optional trailing fields.
    const auto remaining = [&]() -> std::streamoff {
      const auto pos = iss.tellg();
      if (pos < 0) {
        return 0;
      }
      iss.seekg(0, std::ios::end);
      const auto end = iss.tellg();
      iss.seekg(pos);
      if (end < 0 || end < pos) {
        return 0;
      }
      return end - pos;
    };

    if (remaining() >= static_cast<std::streamoff>(sizeof(dispatch_info.subgraph_idx)))
    {
        iss.read(reinterpret_cast<char *>(&dispatch_info.subgraph_idx),
                 sizeof(dispatch_info.subgraph_idx));
    }

    if (remaining() >= static_cast<std::streamoff>(sizeof(size_t)))
    {
        size_t adla_bin_size = 0;
        iss.read(reinterpret_cast<char *>(&adla_bin_size), sizeof(adla_bin_size));
        dispatch_info.adla_bin_size = adla_bin_size;
        if (adla_bin_size > 0 &&
            remaining() >= static_cast<std::streamoff>(adla_bin_size))
        {
            dispatch_info.adla_bin.resize(adla_bin_size);
            iss.read(reinterpret_cast<char *>(dispatch_info.adla_bin.data()),
                     static_cast<std::streamsize>(adla_bin_size));
            if (!iss.good() && !iss.eof())
            {
                dispatch_info.adla_bin.clear();
                dispatch_info.adla_bin_size = 0;
            }
        }
        else if (adla_bin_size > 0)
        {
            // Declared size does not match remaining bytes — treat as missing.
            dispatch_info.adla_bin_size = 0;
            dispatch_info.adla_bin.clear();
        }
    }

    return dispatch_info;
}
