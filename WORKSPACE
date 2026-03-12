workspace(name = "litert")

# -----------------------------------------------------------------------------
# Core Bazel rules
# -----------------------------------------------------------------------------

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Shell rules
http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl",
     "rules_shell_dependencies",
     "rules_shell_toolchains")

rules_shell_dependencies()
rules_shell_toolchains()

# Platform rules
http_archive(
    name = "rules_platform",
    sha256 = "0aadd1bd350091aa1f9b6f2fbcac8cd98201476289454e475b28801ecf85d3fd",
    urls = [
        "https://github.com/bazelbuild/rules_platform/releases/download/0.1.0/rules_platform-0.1.0.tar.gz",
    ],
)

# -----------------------------------------------------------------------------
# Core ML dependencies
# -----------------------------------------------------------------------------

http_archive(
    name = "coremltools",
    build_file = "@//third_party/coremltools:coremltools.BUILD",
    sha256 = "37d4d141718c70102f763363a8b018191882a179f4ce5291168d066a84d01c9d",
    strip_prefix = "coremltools-8.0",
    url = "https://github.com/apple/coremltools/archive/8.0.tar.gz",
    patch_cmds = [
        "sed -i -e 's|import public \"|import public \"mlmodel/format/|g' mlmodel/format/*.proto",
    ],
)

# -----------------------------------------------------------------------------
# TensorFlow Source
# -----------------------------------------------------------------------------

load("//litert:tensorflow_source_rules.bzl", "tensorflow_source_repo")

tensorflow_source_repo(
    name = "org_tensorflow",
    sha256 = "3a26196b1a9cee6e56a17f2334b0be32c23c4a7367648cae942b217091728a98",
    strip_prefix = "tensorflow-f2d8e35dc5369fb4002f57d95303eb551e85f138",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/f2d8e35dc5369fb4002f57d95303eb551e85f138.tar.gz"
    ],
    patches = ["//:PATCH.tf_xla_tsl_win_copts"],
    protobuf_patches = ["//:PATCH.protobuf_port_msvc_compat"],
)

# TensorFlow workspace initialization
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

# -----------------------------------------------------------------------------
# Hermetic Python environment
# -----------------------------------------------------------------------------

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl",
     "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = [
        "@org_tensorflow//:WORKSPACE",
    ],
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl",
     "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

# -----------------------------------------------------------------------------
# TensorFlow additional workspaces
# -----------------------------------------------------------------------------

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace2()
tf_workspace1()
tf_workspace0()

# -----------------------------------------------------------------------------
# Python wheel configuration
# -----------------------------------------------------------------------------

load("@xla//third_party/py:python_wheel.bzl",
     "python_wheel_version_suffix_repository")

python_wheel_version_suffix_repository(
    name = "tf_wheel_version_suffix"
)

# -----------------------------------------------------------------------------
# ML Toolchains (CUDA / NCCL)
# -----------------------------------------------------------------------------

http_archive(
    name = "rules_ml_toolchain",
    sha256 = "9dbee8f24cc1b430bf9c2a6661ab70cbca89979322ddc7742305a05ff637ab6b",
    strip_prefix = "rules_ml_toolchain-545c80f1026d526ea9c7aaa410bf0b52c9a82e74",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/545c80f1026d526ea9c7aaa410bf0b52c9a82e74.tar.gz"
    ],
)

load("@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
     "cc_toolchain_deps")

cc_toolchain_deps()

register_toolchains(
    "@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64"
)

# CUDA
load("@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
     "cuda_json_init_repository")

cuda_json_init_repository()

load("@cuda_redist_json//:distributions.bzl",
     "CUDA_REDISTRIBUTIONS",
     "CUDNN_REDISTRIBUTIONS")

load("@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
     "cuda_redist_init_repositories",
     "cudnn_redist_init_repository")

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS
)

load("@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
     "cuda_configure")

cuda_configure(name = "local_config_cuda")

# NCCL
load("@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
     "nccl_redist_init_repository")

nccl_redist_init_repository()

load("@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
     "nccl_configure")

nccl_configure(name = "local_config_nccl")

# -----------------------------------------------------------------------------
# Third-party libraries
# -----------------------------------------------------------------------------

load("//third_party/tqdm:workspace.bzl", tqdm = "repo")
load("//third_party/dawn:workspace.bzl", dawn = "repo")
load("//third_party/lark:workspace.bzl", lark = "repo")
load("//third_party/xdsl:workspace.bzl", xdsl = "repo")

tqdm()
dawn()
lark()
xdsl()

# -----------------------------------------------------------------------------
# JVM / Android dependencies
# -----------------------------------------------------------------------------

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
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
    ],
    version_conflict_policy = "pinned",
)

# -----------------------------------------------------------------------------
# Kotlin
# -----------------------------------------------------------------------------

http_archive(
    name = "rules_kotlin",
    sha256 = "e1448a56b2462407b2688dea86df5c375b36a0991bd478c2ddd94c97168125e2",
    url = "https://github.com/bazelbuild/rules_kotlin/releases/download/v2.1.3/rules_kotlin-v2.1.3.tar.gz",
)

load("@rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")
load("@rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kotlin_repositories()
kt_register_toolchains()

# -----------------------------------------------------------------------------
# Vendor SDKs
# -----------------------------------------------------------------------------

load("//third_party/qairt:workspace.bzl", "qairt")
load("//third_party/neuro_pilot:workspace.bzl", "neuro_pilot")
load("//third_party/google_tensor:workspace.bzl", "google_tensor")
load("//third_party/litert_gpu:workspace.bzl", "litert_gpu")
load("//third_party/litert_prebuilts:workspace.bzl", "litert_prebuilts")
load("//third_party/exynos_ai_litecore:workspace.bzl", "exynos_ai_litecore")

qairt()
neuro_pilot()
google_tensor()
litert_gpu()
litert_prebuilts()
exynos_ai_litecore()

# Intel OpenVINO
load("//third_party/intel_openvino:openvino.bzl",
     "openvino_configure")

openvino_configure()
