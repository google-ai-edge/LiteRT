import os
import unittest

from tools.clangrttool.clangrttool import normalize_clang_lip_path, ClangRuntimeToolError

clang_lib_path_common = "Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/14.0.0/lib/darwin"
xcode_developer_dir_default = "/Applications/Xcode.app/Contents/Developer"
xcode_developer_dir_non_default1 = "/Applications/Xcode_14.app/Contents/Developer"
xcode_developer_dir_non_default2 = "/Applications/Xcode14.app/Contents/Developer"
xcode_beta_developer_dir_default = "/Applications/Xcode-beta.app/Contents/Developer"
xcode_beta_developer_dir_non_default = "/Users/USER/Downloads/Xcode-beta.app/Contents/Developer"
non_xcode_developer_dir = "/Applications/ProjectBuilder.app/Contents/Developer"

class ClangRuntimeToolTest(unittest.TestCase):

  def assertNormalizedPath(self, local, remote, common=clang_lib_path_common):
    clang_lib_path = os.path.join(remote, common)
    expected = os.path.join(local, common)
    normalized = normalize_clang_lip_path(clang_lib_path, local)

    self.assertEqual(expected, normalized)


  def test_local_and_remote_location_match(self):
    self.assertNormalizedPath(
      local=xcode_developer_dir_default,
      remote=xcode_developer_dir_default
    )


  def test_local_default_and_remote_non_default_location(self):
    self.assertNormalizedPath(
      local=xcode_developer_dir_default,
      remote=xcode_developer_dir_non_default1
    )


  def test_local_non_default_and_remote_default_location(self):
    self.assertNormalizedPath(
      local=xcode_developer_dir_non_default1,
      remote=xcode_developer_dir_default
    )


  def test_local_non_default_and_remote_non_default_location(self):
    self.assertNormalizedPath(
      local=xcode_developer_dir_non_default1,
      remote=xcode_developer_dir_non_default2
    )


  def test_local_default_beta_and_remote_non_default_beta_location(self):
    self.assertNormalizedPath(
      local=xcode_beta_developer_dir_default,
      remote=xcode_beta_developer_dir_non_default
    )


  def test_non_local_default_beta_and_remote_default_beta_location(self):
    self.assertNormalizedPath(
      local=xcode_beta_developer_dir_non_default,
      remote=xcode_beta_developer_dir_default
    )


  def test_local_non_xcode_developer_dir(self):
    self.assertNormalizedPath(
      local=non_xcode_developer_dir,
      remote=xcode_developer_dir_default
    )


  def test_remote_non_xcode_developer_dir(self):
    self.assertNormalizedPath(
      local=xcode_developer_dir_default,
      remote=non_xcode_developer_dir
    )

if __name__ == '__main__':
  unittest.main()
