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

#include "litert/vendors/nvidia/dispatch/aot_artifact.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/nvidia/bytecode.h"

namespace litert::nvidia {
namespace {

class TemporaryArtifact {
 public:
  TemporaryArtifact() {
    char directory_template[] = "/tmp/litert_nvidia_aot_test_XXXXXX";
    char* directory = mkdtemp(directory_template);
    if (directory != nullptr) {
      directory_ = directory;
      path_ = directory_ + "/artifact.bin";
    }
  }

  ~TemporaryArtifact() {
    if (!path_.empty()) {
      chmod(path_.c_str(), S_IRUSR | S_IWUSR);
      unlink(path_.c_str());
      unlink((path_ + ".replacement").c_str());
    }
    if (!directory_.empty()) {
      rmdir(directory_.c_str());
    }
  }

  bool valid() const { return !path_.empty(); }
  const std::string& path() const { return path_; }

  bool Write(const std::vector<uint8_t>& bytes, mode_t mode,
             const std::string& suffix = "") const {
    const std::string output_path = path_ + suffix;
    const int fd =
        open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
             S_IRUSR | S_IWUSR);
    if (fd < 0) {
      return false;
    }
    size_t written = 0;
    while (written < bytes.size()) {
      const ssize_t result =
          write(fd, bytes.data() + written, bytes.size() - written);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        close(fd);
        return false;
      }
      written += static_cast<size_t>(result);
    }
    if (close(fd) != 0) {
      return false;
    }
    return chmod(output_path.c_str(), mode) == 0;
  }

 private:
  std::string directory_;
  std::string path_;
};

TensorRtAotFileIdentity IdentityFromStat(const struct stat& stat_buffer) {
  return {static_cast<uint64_t>(stat_buffer.st_dev),
          static_cast<uint64_t>(stat_buffer.st_ino),
          static_cast<int64_t>(stat_buffer.st_mtim.tv_sec),
          static_cast<int64_t>(stat_buffer.st_mtim.tv_nsec),
          static_cast<int64_t>(stat_buffer.st_ctim.tv_sec),
          static_cast<int64_t>(stat_buffer.st_ctim.tv_nsec)};
}

TensorRtAotLocator LocatorFor(const TemporaryArtifact& artifact,
                              const std::vector<uint8_t>& contents) {
  struct stat stat_buffer{};
  EXPECT_EQ(stat(artifact.path().c_str(), &stat_buffer), 0);
  return {artifact.path(), static_cast<uint64_t>(contents.size()),
          FingerprintTensorRtAotArtifact(contents.data(), contents.size()),
          IdentityFromStat(stat_buffer)};
}

TEST(MappedAotArtifactTest, TrustsExactReadOnlyFileIdentity) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> contents = {1, 3, 5, 7, 9};
  ASSERT_TRUE(artifact.Write(contents, S_IRUSR));
  const auto locator = LocatorFor(artifact, contents);

  auto mapping = MappedAotArtifact::Open(locator);
  ASSERT_TRUE(mapping.HasValue()) << mapping.Error().Message();
  EXPECT_EQ((*mapping)->validation(),
            AotArtifactValidation::kTrustedFileIdentity);
  EXPECT_EQ(std::vector<uint8_t>((*mapping)->data(),
                                 (*mapping)->data() + (*mapping)->size()),
            contents);
}

TEST(MappedAotArtifactTest, ReadOnlySealClearsWriteBits) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> contents = {1, 3, 5, 7, 9};
  ASSERT_TRUE(artifact.Write(contents, S_IRUSR));

  struct stat stat_buffer{};
  ASSERT_EQ(stat(artifact.path().c_str(), &stat_buffer), 0);
  EXPECT_EQ(stat_buffer.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH), 0);

  // Root can bypass discretionary access-control mode bits.
  if (geteuid() != 0) {
    const int fd = open(artifact.path().c_str(), O_WRONLY | O_CLOEXEC);
    EXPECT_LT(fd, 0);
    if (fd >= 0) {
      close(fd);
    }
  }
}

TEST(MappedAotArtifactTest, ForcedAuditDetectsInPlaceChangedBytes) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> original = {1, 3, 5, 7, 9};
  const std::vector<uint8_t> changed = {1, 3, 5, 7, 8};
  ASSERT_TRUE(artifact.Write(original, S_IRUSR));
  const auto locator = LocatorFor(artifact, original);
  ASSERT_EQ(chmod(artifact.path().c_str(), S_IRUSR | S_IWUSR), 0);
  ASSERT_TRUE(artifact.Write(changed, S_IRUSR));

  ASSERT_EQ(setenv("LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION", "1",
                   /*overwrite=*/1),
            0);
  auto mapping = MappedAotArtifact::Open(locator);
  unsetenv("LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION");
  EXPECT_FALSE(mapping.HasValue());
}

TEST(MappedAotArtifactTest, ValidReplacementFallsBackToFingerprint) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> contents = {2, 4, 6, 8};
  ASSERT_TRUE(artifact.Write(contents, S_IRUSR));
  const auto locator = LocatorFor(artifact, contents);
  ASSERT_TRUE(artifact.Write(contents, S_IRUSR, ".replacement"));
  ASSERT_EQ(rename((artifact.path() + ".replacement").c_str(),
                   artifact.path().c_str()),
            0);

  auto mapping = MappedAotArtifact::Open(locator);
  ASSERT_TRUE(mapping.HasValue()) << mapping.Error().Message();
  EXPECT_EQ((*mapping)->validation(),
            AotArtifactValidation::kComputedFingerprint);
}

TEST(MappedAotArtifactTest, ChangedReplacementFailsFingerprintValidation) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> original = {2, 4, 6, 8};
  const std::vector<uint8_t> changed = {2, 4, 6, 7};
  ASSERT_TRUE(artifact.Write(original, S_IRUSR));
  const auto locator = LocatorFor(artifact, original);
  ASSERT_TRUE(artifact.Write(changed, S_IRUSR, ".replacement"));
  ASSERT_EQ(rename((artifact.path() + ".replacement").c_str(),
                   artifact.path().c_str()),
            0);

  auto mapping = MappedAotArtifact::Open(locator);
  EXPECT_FALSE(mapping.HasValue());
}

TEST(MappedAotArtifactTest, WritableArtifactCannotUseIdentityFastPath) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> contents = {10, 20, 30, 40};
  ASSERT_TRUE(artifact.Write(contents, S_IRUSR | S_IWUSR));
  const auto locator = LocatorFor(artifact, contents);

  auto first = MappedAotArtifact::Open(locator);
  ASSERT_TRUE(first.HasValue()) << first.Error().Message();
  EXPECT_EQ((*first)->validation(),
            AotArtifactValidation::kComputedFingerprint);
  auto second = MappedAotArtifact::Open(locator);
  ASSERT_TRUE(second.HasValue()) << second.Error().Message();
  EXPECT_EQ((*second)->validation(), AotArtifactValidation::kProcessCache);
}

TEST(MappedAotArtifactTest, ForcedAuditComputesMatchingContentFingerprint) {
  TemporaryArtifact artifact;
  ASSERT_TRUE(artifact.valid());
  const std::vector<uint8_t> contents = {11, 22, 33, 44};
  ASSERT_TRUE(artifact.Write(contents, S_IRUSR));
  const auto locator = LocatorFor(artifact, contents);

  ASSERT_EQ(setenv("LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION", "1",
                   /*overwrite=*/1),
            0);
  auto mapping = MappedAotArtifact::Open(locator);
  unsetenv("LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION");
  ASSERT_TRUE(mapping.HasValue()) << mapping.Error().Message();
  EXPECT_EQ((*mapping)->validation(),
            AotArtifactValidation::kComputedFingerprint);
}

}  // namespace
}  // namespace litert::nvidia
