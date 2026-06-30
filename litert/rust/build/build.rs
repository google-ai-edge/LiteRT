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

use build_print::info;
use std::env;
use std::fs::{self, File};
use std::io::{self, copy};
use std::path::{Path, PathBuf};
use std::process::Command;

// Constansts that are used by the build script.

// The environment variable that contains the output directory where it's possible to write or modify files.
const OUT_DIR_ENV_VAR: &str = "OUT_DIR";
// Cargo environment variable to definbe target os and architecture.
const CARGO_CFG_TARGET_OS_ENV_VAR: &str = "CARGO_CFG_TARGET_OS";
const CARGO_CFG_TARGET_ARCH_ENV_VAR: &str = "CARGO_CFG_TARGET_ARCH";
// If we are generating docuemntation in hermetic environment for docs.rs
const CARGO_DOCS_RS: &str = "DOCS_RS";

const LITERT_RUNTIME_DOWNLOAD_URL: &str = "https://storage.googleapis.com/litert/binaries/latest/";
const LITERT_CC_SDK_DOWNLOAD_URL: &str =
    "https://github.com/google-ai-edge/LiteRT/releases/latest/download/litert_cc_sdk.zip";

#[derive(Debug, PartialEq)]
enum TargetPlatform {
    AndroidArm64,
    AndroidX86,
    LinuxArm64,
    LinuxX86,
    MacosArm64,
    WindowsX86,
}

impl TargetPlatform {
    // Create target platform from cargo environment variables.
    fn from_cargo_env() -> Result<Self, String> {
        let os = env::var(CARGO_CFG_TARGET_OS_ENV_VAR).unwrap_or_else(|_| "unknown".to_string());
        let arch =
            env::var(CARGO_CFG_TARGET_ARCH_ENV_VAR).unwrap_or_else(|_| "unknown".to_string());
        info!("Target os {} platform {}", os.as_str(), arch.as_str());

        match (os.as_str(), arch.as_str()) {
            ("android", "aarch64") => Ok(TargetPlatform::AndroidArm64),
            ("android", "x86_64") => Ok(TargetPlatform::AndroidX86),
            ("linux", "aarch64") => Ok(TargetPlatform::LinuxArm64),
            ("linux", "x86_64") => Ok(TargetPlatform::LinuxX86),
            ("macos", "aarch64") => Ok(TargetPlatform::MacosArm64),
            ("windows", "x86_64") => Ok(TargetPlatform::WindowsX86),
            _ => Err(format!("Unknown target platform os:{} aarch:{}", os, arch)),
        }
    }

    // See https://ai.google.dev/edge/litert/next/cpp_sdk
    fn runtime_name(&self) -> String {
        match self {
            TargetPlatform::AndroidArm64
            | TargetPlatform::AndroidX86
            | TargetPlatform::LinuxArm64
            | TargetPlatform::LinuxX86 => "libLiteRt.so".to_string(),
            TargetPlatform::MacosArm64 => "libLiteRt.dylib".to_string(),
            TargetPlatform::WindowsX86 => "libLiteRt.dll".to_string(),
        }
    }

    fn runtime_directory(&self) -> String {
        match self {
            TargetPlatform::AndroidArm64 => "android_arm64".to_string(),
            TargetPlatform::AndroidX86 => "android_x86_64".to_string(),
            TargetPlatform::LinuxArm64 => "linux_arm64".to_string(),
            TargetPlatform::LinuxX86 => "linux_x86_64".to_string(),
            TargetPlatform::MacosArm64 => "macos_arm64".to_string(),
            TargetPlatform::WindowsX86 => "windows_x86_64".to_string(),
        }
    }
}

// Helper function to check if a tool is installed
fn check_tool_installed(tool: &str) -> Result<(), String> {
    match Command::new(tool).arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(format!("Required tool '{}' is not installed or not in PATH.", tool)),
    }
}

// Resolve a program name against PATH, returning its absolute path. If `program`
// already contains a path separator it is returned as-is when it exists.
fn resolve_in_path(program: &str) -> Option<PathBuf> {
    let direct = Path::new(program);
    if direct.components().count() > 1 {
        return if direct.is_file() { Some(direct.to_path_buf()) } else { None };
    }
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var).map(|dir| dir.join(program)).find(|candidate| candidate.is_file())
}

// Find an archiver and ranlib that belong to the same toolchain as the given C
// compiler. CMake would otherwise select each independently from PATH, which can
// mix incompatible toolchains (for example GNU binutils `ar` with LLVM
// `llvm-ranlib`) and corrupt static archives. Resolving both from the
// compiler's own directory keeps them mutually consistent on any platform.
fn matched_archiver_tools(c_compiler: &str) -> Option<(PathBuf, PathBuf)> {
    let compiler_dir = resolve_in_path(c_compiler)?.parent()?.to_path_buf();
    // A clang install ships `llvm-ar`/`llvm-ranlib`; a gcc install ships the
    // binutils `ar`/`ranlib` siblings. Prefer the LLVM pair, then plain names.
    for (ar_name, ranlib_name) in [("llvm-ar", "llvm-ranlib"), ("ar", "ranlib")] {
        let ar = compiler_dir.join(ar_name);
        let ranlib = compiler_dir.join(ranlib_name);
        if ar.is_file() && ranlib.is_file() {
            return Some((ar, ranlib));
        }
    }
    None
}

// Helper function to download a file
fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(None) // Disable total request timeout
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let mut response = client.get(url).send()?;
    let mut dest = File::create(path)?;
    copy(&mut response, &mut dest)?;
    Ok(())
}

fn unzip_archive(
    archive_path: &Path,
    extract_to: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let file = File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    // Assume that the directory of the archive has one top directory, that the function return.
    let mut root_directory = PathBuf::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let file_name = file.mangled_name();
        let outpath = extract_to.join(&file_name);

        // Extract the root directory of the file.
        if let Some(rd) = Path::new(&file_name).iter().next() {
            root_directory = Path::new(rd).to_path_buf();
        }
        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(root_directory)
}

fn download_runtime(
    target_plarform: &TargetPlatform,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Downloading LiteRT ");

    let runtime_name = target_plarform.runtime_name();
    let runtime_dir = target_plarform.runtime_directory();
    let runtime_download_url = String::from(LITERT_RUNTIME_DOWNLOAD_URL)
        + runtime_dir.as_str()
        + "/"
        + runtime_name.as_str();
    let runtime_local_name = out_dir.join(runtime_name);
    download_file(&runtime_download_url, &runtime_local_name)?;
    return Ok(());
}

struct LiteRTSdk {
    sdk_root_dir: PathBuf,
    sdk_build_dir: PathBuf,
}

impl LiteRTSdk {
    fn new(sdk_root_dir: PathBuf, sdk_build_dir: PathBuf) -> Self {
        Self { sdk_root_dir, sdk_build_dir }
    }
}

// Following LiteRT C++ SDK build process.
// https://ai.google.dev/edge/litert/next/cpp_sdk
fn download_and_build_cpp_sdk(
    target_platform: &TargetPlatform,
    out_dir: &Path,
) -> Result<LiteRTSdk, Box<dyn std::error::Error>> {
    info!("Downloading LiteRT C++ SDK...");
    let sdk_zip_path = out_dir.join("litert_cc_sdk.zip");
    download_file(LITERT_CC_SDK_DOWNLOAD_URL, &sdk_zip_path)?;

    info!("Unzipping LiteRT C++ SDK...");
    let sdk_extract_dir = out_dir.join("litert_cc_sdk_extracted");
    if !sdk_extract_dir.exists() {
        fs::create_dir_all(&sdk_extract_dir)?;
    }
    let zip_root_dir = unzip_archive(&sdk_zip_path, &sdk_extract_dir)?;
    let sdk_root = sdk_extract_dir.join(zip_root_dir);

    info!("Downloading prebuilt runtime for SDK to {}", sdk_root.display());
    download_runtime(target_platform, &sdk_root)?;

    let build_dir = out_dir.join("litert_cc_sdk_build");
    info!("Building C++ SDK with CMake in {}", build_dir.display());

    let mut cmake_configure = Command::new("cmake");
    cmake_configure
        .arg("-S")
        .arg(&sdk_root)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_C_COMPILER=clang")
        .arg("-DCMAKE_CXX_COMPILER=clang++");

    // CMake otherwise selects `ar` and `ranlib` independently via PATH search.
    // On setups where PATH mixes toolchains (for example a Homebrew install that
    // puts GNU binutils `ar`/`ranlib` ahead of everything while LLVM provides
    // `llvm-ranlib`), it creates static archives with one toolchain and indexes
    // them with another. The archive formats are incompatible, and indexing the
    // empty `libabsl_crc_cpu_detect.a` then fails with:
    //   llvm-ranlib: error: ...: '__.SYMDEF': The end of the file was unexpectedly encountered
    // Pin both tools to the matched pair that ships alongside the chosen
    // compiler so the build is consistent regardless of PATH ordering. This is
    // platform agnostic: it uses whichever `clang` the developer has rather than
    // dictating a specific toolchain.
    if let Some((ar, ranlib)) = matched_archiver_tools("clang") {
        info!("Using archiver {} and ranlib {}", ar.display(), ranlib.display());
        cmake_configure
            .arg(format!("-DCMAKE_AR={}", ar.display()))
            .arg(format!("-DCMAKE_RANLIB={}", ranlib.display()));
    }

    let status = cmake_configure.status()?;

    if !status.success() {
        return Err(format!("CMake configure failed with status: {}", status).into());
    }

    // Only build the static C++ API library that the Rust binding links against.
    // Building the full project also compiles the SDK's demo executables
    // (run_model_simple, dump_model_simple), which transitively require the
    // imported runtime via a path hardcoded to "libLiteRt.so". On macOS the
    // runtime is "libLiteRt.dylib", so those demo targets fail to build even
    // though the binding itself never needs them.
    let status = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("litert_cc_api")
        .arg("-j")
        .status()?;

    if !status.success() {
        return Err(format!("CMake build failed with status: {}", status).into());
    }

    Ok(LiteRTSdk::new(sdk_root, build_dir))
}

fn dump_all_env_vars() {
    for (key, value) in env::vars() {
        info!("Environment: {}: {}", key, value);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dump_all_env_vars();
    println!("cargo::rustc-check-cfg=cfg(bazel_bindgen, cargo_bindgen, docsrs)");
    // Check if we are currently generating documentation
    let is_doc_gen = env::var(CARGO_DOCS_RS).is_ok();

    if is_doc_gen {
        info!("Skipping heavy lifting because we are just building docs!");
        println!("cargo::rustc-check-cfg=cfg(docsrs)");
        println!("cargo::rustc-cfg=docsrs");
        return Ok(());
    }
    println!("cargo::rerun-if-changed=build/build.rs");

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        info!("Manifest dir {}", manifest_dir);
    }

    let target_platform = TargetPlatform::from_cargo_env()?;

    check_tool_installed("cmake")?;
    check_tool_installed("clang")?;

    let out_dir = PathBuf::from(env::var(OUT_DIR_ENV_VAR)?);
    let litert_sdk = download_and_build_cpp_sdk(&target_platform, &out_dir)?;
    info!("Runtime dir {}", litert_sdk.sdk_root_dir.display());
    info!("Build dir {}", litert_sdk.sdk_build_dir.display());

    println!("cargo::rustc-link-search=native={}", litert_sdk.sdk_root_dir.display());
    println!("cargo::rustc-link-lib=dylib=LiteRt");
    println!("cargo::rustc-link-search=native={}", litert_sdk.sdk_build_dir.display());
    println!("cargo::rustc-link-lib=static=litert_cc_api");

    // Generate bindings from the downloaded SDK's own headers rather than the
    // in-tree headers reached via wrapper.h's "../c/..." relative includes.
    // The prebuilt runtime (libLiteRt) is built from the published SDK, so its
    // headers match the runtime ABI exactly. The in-tree headers can be ahead
    // of the published runtime (for example they add an LiteRtEnvironment
    // parameter to model creation, or declare APIs whose types the released
    // SDK has not shipped yet), which would produce bindings that do not match
    // the linked library. Mirror wrapper.h's include set against the SDK copies.
    let sdk_wrapper_path = out_dir.join("sdk_wrapper.h");
    let sdk_wrapper_contents = "\
#include \"litert/c/internal/litert_logging.h\"
#include \"litert/c/litert_common.h\"
#include \"litert/c/litert_compiled_model.h\"
#include \"litert/c/litert_environment.h\"
#include \"litert/c/litert_environment_options.h\"
#include \"litert/c/litert_event.h\"
#include \"litert/c/litert_metrics.h\"
#include \"litert/c/litert_model.h\"
#include \"litert/c/litert_op_options.h\"
#include \"litert/c/litert_opaque_options.h\"
#include \"litert/c/litert_options.h\"
#include \"litert/c/litert_tensor_buffer.h\"
#include \"litert/c/litert_tensor_buffer_requirements.h\"
";
    fs::write(&sdk_wrapper_path, sdk_wrapper_contents)?;

    let bindings = bindgen::Builder::default()
        .header(sdk_wrapper_path.to_str().expect("OUT_DIR path is not valid UTF-8"))
        // Add the include path so clang can find dependent headers
        .clang_arg(format!("-I{}", litert_sdk.sdk_root_dir.display()))
        .clang_arg(format!("-I{}", litert_sdk.sdk_root_dir.join("litert").join("c").display()))
        .clang_arg(format!("-I{}", litert_sdk.sdk_build_dir.join("include").display()))
        .clang_arg("-DLITERT_DISABLE_GPU")
        .layout_tests(false)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let bindings_out_path = out_dir.join("bindings.rs");
    info!("Writing binding.rs to {}", bindings_out_path.display());
    bindings.write_to_file(bindings_out_path).expect("Couldn't write bindings!");
    println!("cargo::rustc-cfg=cargo_bindgen");

    Ok(())
}
