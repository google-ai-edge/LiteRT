# buildifier: disable=load-on-top

workspace(name = "litert")

# buildifier: disable=load-on-top

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

http_archive(
    name = "rules_platform",
    sha256 = "0aadd1bd350091aa1f9b6f2fbcac8cd98201476289454e475b28801ecf85d3fd",
    url = "https://github.com/bazelbuild/rules_platform/releases/download/0.1.0/rules_platform-0.1.0.tar.gz",
)

# Use recent platoforms version to support uefi platform.
http_archive(
    name = "platforms",
    sha256 = "3384eb1c30762704fbe38e440204e114154086c8fc8a8c2e3e28441028c019a8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
    ],
)

# Use 3.22.0 (from 3.5.1 of tensorflow) to fix binary signing issue on MacOS Tahoe.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "a78f26c22ac8d6e3f3fcaad50eace4d9c767688bd7254b75bdf4a6735b299f6a",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/3.22.0/rules_apple.3.22.0.tar.gz",
)

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "f7a67197cd8a79debfe70b8cef4dc19d03039af02cc561e31e0718e98cad83ac",
    url = "https://github.com/bazelbuild/rules_swift/releases/download/2.9.0/rules_swift.2.9.0.tar.gz",
)

# Lower the version from 1.24.5 that tensorflow uses to 1.23.1, the highest version which don't have
# issues with missing LC_UUID, DEVELOPER_DIR or SDKROOT on MacOS Tahoe.
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "ee20cc5c0bab47065473c8033d462374dd38d172406ecc8de5c8f08487943f2f",
    url = "https://github.com/bazelbuild/apple_support/releases/download/1.23.1/apple_support.1.23.1.tar.gz",
)

http_archive(
    name = "bazel_features",
    sha256 = "c26b4e69cf02fea24511a108d158188b9d8174426311aac59ce803a78d107648",
    strip_prefix = "bazel_features-1.43.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.43.0/bazel_features-v1.43.0.tar.gz",
)

# Download coremltools of the same version of tensorflow, but with a custom patchcmd until
# tensorflow is updated to do the same patchcmd.
http_archive(
    name = "coremltools",
    build_file = "@//third_party/coremltools:coremltools.BUILD",
    patch_cmds = [
        # Append "mlmodel/format/" to the import path of all proto files.
        "sed -i -e 's|import public \"|import public \"mlmodel/format/|g' mlmodel/format/*.proto",
    ],
    sha256 = "37d4d141718c70102f763363a8b018191882a179f4ce5291168d066a84d01c9d",
    strip_prefix = "coremltools-8.0",
    url = "https://github.com/apple/coremltools/archive/8.0.tar.gz",
)

# Load the custom repository rule to select either a local TensorFlow source or a remote http_archive.
load("//litert:tensorflow_source_rules.bzl", "tensorflow_source_repo")

tensorflow_source_repo(
    name = "org_tensorflow",
    patches = ["//:PATCH.flatbuffers_windows_no_bash"],
    protobuf_patches = ["//:PATCH.protobuf_port_msvc_compat"],
    sha256 = "977114079cda0d6aa9d05bc73ae2c2e3d36705fbb041e631a564e4d42e1e1dd9",
    strip_prefix = "tensorflow-49e7f1937d1509dd7fea41bff9ccc994baa97258",
    urls = ["https://github.com/tensorflow/tensorflow/archive/49e7f1937d1509dd7fea41bff9ccc994baa97258.tar.gz"],
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

# Toolchains for ML projects hermetic builds.
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "9dbee8f24cc1b430bf9c2a6661ab70cbca89979322ddc7742305a05ff637ab6b",
    strip_prefix = "rules_ml_toolchain-545c80f1026d526ea9c7aaa410bf0b52c9a82e74",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/545c80f1026d526ea9c7aaa410bf0b52c9a82e74.tar.gz",
    ],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("//third_party/tqdm:workspace.bzl", tqdm = "repo")

tqdm()

load("//third_party/dawn:workspace.bzl", dawn = "repo")

dawn()

load("//third_party/lark:workspace.bzl", lark = "repo")

lark()

load("//third_party/xdsl:workspace.bzl", xdsl = "repo")

xdsl()

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "litert_maven",
    artifacts = [
        "androidx.lifecycle:lifecycle-common:2.8.7",
        "com.google.android.play:ai-delivery:0.1.1-alpha01",
        "com.google.guava:guava:33.4.6-android",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.8.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.0",
    ],
    repositories = [
        "https://jcenter.bintray.com",
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
    ],
    version_conflict_policy = "pinned",
)

# Kotlin rules
http_archive(
    name = "rules_kotlin",
    sha256 = "e1448a56b2462407b2688dea86df5c375b36a0991bd478c2ddd94c97168125e2",
    url = "https://github.com/bazelbuild/rules_kotlin/releases/download/v2.1.3/rules_kotlin-v2.1.3.tar.gz",
)

# Sentencepiece
http_archive(
    name = "sentencepiece",
    build_file = "@//:BUILD.sentencepiece",
    patch_cmds = [
        # Empty config.h seems enough.
        "touch config.h",
        # Replace third_party/absl/ with absl/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/absl/|#include \"absl/|g' *.h *.cc",
        # Replace third_party/darts_clone/ with include/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/darts_clone/|#include \"include/|g' *.h *.cc",
    ],
    patches = ["@//:PATCH.sentencepiece"],
    sha256 = "9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86",
    strip_prefix = "sentencepiece-0.2.0/src",
    url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz",
)

# Darts Clone
http_archive(
    name = "darts_clone",
    build_file = "@//:BUILD.darts_clone",
    sha256 = "4a562824ec2fbb0ef7bd0058d9f73300173d20757b33bb69baa7e50349f65820",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    url = "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.tar.gz",
)

# tomlplusplus
http_archive(
    name = "tomlplusplus",
    build_file = "@//:BUILD.tomlplusplus",
    patch_cmds = [
        "echo '#define TOML_IMPLEMENTATION' > toml.cc",
        "echo '#include \"toml.hpp\"' >> toml.cc",
    ],
    sha256 = "8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155",
    strip_prefix = "tomlplusplus-3.4.0",
    url = "https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz",
)

load("@rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()  # if you want the default. Otherwise see custom kotlinc distribution below

load("@rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()  # to use the default toolchain, otherwise see toolchains below

load("//third_party/stblib:workspace.bzl", stblib = "repo")

stblib()

# TEST DATA ########################################################################################

load("//third_party/models:workspace.bzl", "models")

models()

# VENDOR SDKS ######################################################################################

# QUALCOMM ---------------------------------------------------------------------------------------

# The actual macro call will be set during configure for now.
load("//third_party/qairt:workspace.bzl", "qairt")

qairt()

# MEDIATEK ---------------------------------------------------------------------------------------

# Currently only works with local sdk
load("//third_party/neuro_pilot:workspace.bzl", "neuro_pilot")

neuro_pilot()

# GOOGLE TENSOR ----------------------------------------------------------------------------------
load("//third_party/google_tensor:workspace.bzl", "google_tensor")

google_tensor()

# LiteRT GPU ----------------------------------------------------------------------------------
load("//third_party/litert_gpu:workspace.bzl", "litert_gpu")

litert_gpu()

# LiteRT Prebuilts ---------------------------------------------------------------------------------
load("//third_party/litert_prebuilts:workspace.bzl", "litert_prebuilts")

litert_prebuilts()

# INTEL OPENVINO ---------------------------------------------------------------------------------
load("//third_party/intel_openvino:openvino.bzl", "openvino_configure")

openvino_configure()

# SAMSUNG EXYNOS ----------------------------------------------------------------------------------
load("//third_party/exynos_ai_litecore:workspace.bzl", "exynos_ai_litecore")

exynos_ai_litecore()
