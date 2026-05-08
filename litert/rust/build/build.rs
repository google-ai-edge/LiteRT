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
        _ => Err(format!(
            "Required tool '{}' is not installed or not in PATH.",
            tool
        )),
    }
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
        Self {
            sdk_root_dir,
            sdk_build_dir,
        }
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

    info!(
        "Downloading prebuilt runtime for SDK to {}",
        sdk_root.display()
    );
    download_runtime(target_platform, &sdk_root)?;

    let build_dir = out_dir.join("litert_cc_sdk_build");
    info!("Building C++ SDK with CMake in {}", build_dir.display());

    let status = Command::new("cmake")
        .arg("-S")
        .arg(&sdk_root)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_C_COMPILER=clang")
        .arg("-DCMAKE_CXX_COMPILER=clang++")
        .status()?;

    if !status.success() {
        return Err(format!("CMake configure failed with status: {}", status).into());
    }

    let status = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
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
    println!("cargo:rustc-check-cfg=cfg(bindgen_rs_file, cargo_bindgen, docsrs)");
    // Check if we are currently generating documentation
    let is_doc_gen = env::var(CARGO_DOCS_RS).is_ok();

    if is_doc_gen {
        info!("Skipping heavy lifting because we are just building docs!");
        println!("cargo:rustc-check-cfg=cfg(docsrs)");
        println!("cargo:rustc-cfg=docsrs");
        return Ok(());
    }
    println!("cargo::rerun-if-changed=build/build.rs");
    println!("cargo::rerun-if-changed=wrapper.h");

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        info!("Manifest dir {}", manifest_dir);
    }

    let target_platform = TargetPlatform::from_cargo_env()?;

    check_tool_installed("cmake")?;
    check_tool_installed("clang")?;

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let litert_sdk = download_and_build_cpp_sdk(&target_platform, &out_dir)?;
    info!("Runtime dir {}", litert_sdk.sdk_root_dir.display());
    info!("Build dir {}", litert_sdk.sdk_build_dir.display());

    println!(
        "cargo::rustc-link-search=native={}",
        litert_sdk.sdk_root_dir.display()
    );
    println!("cargo::rustc-link-lib=dylib=LiteRt");
    println!(
        "cargo::rustc-link-search=native={}",
        litert_sdk.sdk_build_dir.display()
    );
    println!("cargo::rustc-link-lib=static=litert_cc_api");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // Add the include path so clang can find dependent headers
        .clang_arg(format!("-I{}", litert_sdk.sdk_root_dir.display()))
        .clang_arg(format!(
            "-I{}",
            litert_sdk.sdk_root_dir.join("litert").join("c").display()
        ))
        .clang_arg(format!(
            "-I{}",
            litert_sdk.sdk_build_dir.join("include").display()
        ))
        .clang_arg("-DLITERT_DISABLE_GPU")
        .layout_tests(false)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    // Get the output directory where cargo wants us to build things
    let out_dir = PathBuf::from(env::var(OUT_DIR_ENV_VAR)?);
    let bindings_out_path = out_dir.join("bindings.rs");
    info!("Writing binding.rs to {}", bindings_out_path.display());
    bindings
        .write_to_file(bindings_out_path)
        .expect("Couldn't write bindings!");
    println!("cargo::rustc-check-cfg=cfg(bindgen_rs_file, cargo_bindgen)");
    println!("cargo::rustc-cfg=cargo_bindgen");

    Ok(())
}
