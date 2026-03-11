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
// The environment variable that points to LiteRT sources, if not set a copy of sources will be
// downloaded from github.
const RUST_LITERT_SOURCE_DIR_ENV_VAR: &str = "RUST_LITERT_SOURCE_DIR";
// The environment variable that points to pre-build runtime. If not set, the runtime will be
// either downloaded or build.
const RUST_LITERT_RUNTIME_LIBRARY_DIR_ENV_VAR: &str = "RUST_LITERT_RUNTIME_LIBRARY_DIR";
// When the feature is set, the runtime will be build in Docker.
const CARGO_FEATURE_BUILD_RUNTIME_WITH_DOCKER: &str = "CARGO_FEATURE_BUILD_RUNTIME_WITH_DOCKER";
// Cargo environment variable to definbe target os and architecture.
const CARGO_CFG_TARGET_OS_ENV_VAR: &str = "CARGO_CFG_TARGET_OS";
const CARGO_CFG_TARGET_ARCH_ENV_VAR: &str = "CARGO_CFG_TARGET_ARCH";

// The URL of LiterRT release archive on Github.
// The sources are downloaded from github if the sources are not provided by the environment variable.
const LITERT_RELEASE_ARCHIVE_URL: &str =
       "https://github.com/google-ai-edge/LiteRT/archive/refs/heads/main.zip";
    // "https://github.com/MaxGubin/LiteRT/archive/refs/heads/main.zip";

// Different paths that are used during the binary build process by Docker.
const DOCKER_BUILD_SCRIPT_PATH: &str = "docker_build/build_with_docker.sh";
const LITERT_RUNTIME_LIBRARY_DIR: &str = "litert_runtime";
const DOCKER_BUILT_RUNTIME_LIBRARY_DIR: &str =
    "litert_build_container:/litert_build/bazel-bin/litert/c/libLiteRt.so";

const LITERT_RUNTIME_DOWNLOAD_URL: &str = "https://storage.googleapis.com/litert/binaries/latest/";

// Different paths related to preparing sources
const BUILD_CONFIG_H_OUT: &str = "litert/build_common/build_config.h";
const BUILD_INCLUDE_PATH: &str = "litert/rust";
const ADDITIONAL_INCLUDE_PATH: &str = "litert/rust/build";

// A helper macro to panic with a clear message if a command fails
macro_rules! run_command {
    ($cmd:expr) => {
        let status = $cmd.status().expect("Failed to execute command");
        if !status.success() {
            panic!("Command failed: {:?}", $cmd);
        }
    };
}

// Target platform that the runtime will be build for.
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

// Unzip sources from an archive to a directory.
fn unzip_sources(
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

// Uses already installed LiteRT sources or download them from github.
fn get_litert_sources() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(litert_dir) = env::var(RUST_LITERT_SOURCE_DIR_ENV_VAR) {
        info!("Using LiteRT sources from {}", litert_dir);
        return Ok(PathBuf::from(litert_dir));
    }
    let out_dir = PathBuf::from(env::var(OUT_DIR_ENV_VAR)?);
    let download_archive_path = out_dir.join("litert_source.zip");
    info!(
        "Downloading LiteRT sources to {}...",
        download_archive_path.display()
    );
    download_file(LITERT_RELEASE_ARCHIVE_URL, &download_archive_path)?;
    info!("Unzipping LiteRT sources to {}...", out_dir.display());
    let zip_root_dir = unzip_sources(&download_archive_path, &out_dir)?;

    let source_root_dir = out_dir.join(zip_root_dir);
    Ok(source_root_dir)
}

// Some sources need to be modified based on the configuration.
// It's done by Bazel or Cmake. In case of Rust, this is simply a hack that adds path to a predefined
// config file.
fn prepare_sources(sources_path: &PathBuf) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    info!("Preparing sources");
    let build_config_out_path = sources_path.join(BUILD_CONFIG_H_OUT);
    if !build_config_out_path.exists() {
        let additional_path = sources_path.join(ADDITIONAL_INCLUDE_PATH);
        info!("Adding additional path {}", additional_path.display());
        // The source isn't configured, point it to the dummy config file
        return Ok(Some(additional_path));
    }
    Ok(None)
}

// Download LiteRT runtime library from Google Cloud Storage.
fn download_runtime(
    target_plarform: &TargetPlatform,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    info!("Downloading LiteRT ");
    let out_dir = PathBuf::from(env::var(OUT_DIR_ENV_VAR)?);

    let litert_runtime_dir = out_dir.join(LITERT_RUNTIME_LIBRARY_DIR);
    if !litert_runtime_dir.exists() {
        fs::create_dir_all(&litert_runtime_dir)?;
    }

    let runtime_name = target_plarform.runtime_name();
    let runtime_dir = target_plarform.runtime_directory();
    // Builld a path to the runtime library in Google Cloud Storage.
    let runtime_download_url = String::from(LITERT_RUNTIME_DOWNLOAD_URL)
        + runtime_dir.as_str()
        + "/"
        + runtime_name.as_str();
    let runtime_local_name = litert_runtime_dir.join(runtime_name);
    download_file(&runtime_download_url, &runtime_local_name)?;
    return Ok(litert_runtime_dir);
}

// Uses existing LiteRT runtime library or download or build it with docker.
fn get_litert_runtime_library(
    litert_source_dir: &Path,
    target_platform: &TargetPlatform,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(litert_dir) = env::var(RUST_LITERT_RUNTIME_LIBRARY_DIR_ENV_VAR) {
        info!("Using LiteRT runtime library from {}", litert_dir);
        return Ok(PathBuf::from(litert_dir));
    }
    if env::var(CARGO_FEATURE_BUILD_RUNTIME_WITH_DOCKER).is_err() {
        return download_runtime(target_platform);
    }
    info!("Building LiteRT runtime library with docker...");
    check_tool_installed("docker")?;

    let build_script_path = litert_source_dir.join(DOCKER_BUILD_SCRIPT_PATH);
    run_command!(Command::new("bash")
        .current_dir(&litert_source_dir)
        .arg(&build_script_path));
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    // Create a directory where the library will be located.
    let litert_runtime_dir = out_dir.join(LITERT_RUNTIME_LIBRARY_DIR);
    if !litert_runtime_dir.exists() {
        fs::create_dir_all(&litert_runtime_dir)?;
    }
    let litert_runtime_lib = litert_runtime_dir.join(target_platform.runtime_name());
    info!(
        "Copying LiteRT runtime library to {}",
        litert_runtime_dir.display()
    );
    run_command!(Command::new("docker")
        .arg("cp")
        .arg(DOCKER_BUILT_RUNTIME_LIBRARY_DIR)
        .arg(&litert_runtime_lib));
    Ok(litert_runtime_dir)
}

// Dump all environment variables. A debug function
// to investigate build issues.
fn dump_all_env_vars() {
    // env::vars() returns an iterator of (Key, Value)
    for (key, value) in env::vars() {
        info!("Environment: {}: {}", key, value);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dump_all_env_vars();
    println!("cargo::rerun-if-changed=build/build.rs");
    println!("cargo::rerun-if-changed=wrapper.h");

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        info!("Manifest dir {}", manifest_dir);
    }

    let target_platform = TargetPlatform::from_cargo_env()?;

    let litert_source_dir = get_litert_sources()?;
    let litert_include_dir = litert_source_dir.join(BUILD_INCLUDE_PATH);
    let litert_runtime_dir = get_litert_runtime_library(&litert_source_dir, &target_platform)?;
    let additional_include_dir = prepare_sources(&litert_source_dir)?;

    println!(
        "cargo::rustc-link-search=native={}",
        litert_runtime_dir.display()
    );
    println!("cargo::rustc-link-lib=dylib=LiteRt");

    check_tool_installed("clang")?;
    let mut bindgen_config = bindgen::Builder::default()
        .header("wrapper.h")
        // Add the include path so clang can find dependent headers
        .clang_arg(format!("-I{}", litert_source_dir.display()))
        .clang_arg(format!("-I{}", litert_include_dir.display()));
    if let Some(additional_include_dir_p) = additional_include_dir {
        bindgen_config =
            bindgen_config.clang_arg(format!("-I{}", additional_include_dir_p.display()));
    }

    let bindings = bindgen_config
        .clang_arg("-DLITERT_DISABLE_GPU")
        .layout_tests(false)
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    // Get the output directory where cargo wants us to build things
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let bindings_out_path = out_dir.join("bindings.rs");
    info!("Writing binding.rs to {}", bindings_out_path.display());
    bindings
        .write_to_file(bindings_out_path)
        .expect("Couldn't write bindings!");
    println!("cargo::rustc-check-cfg=cfg(bindgen_rs_file, cargo_bindgen)");
    println!("cargo::rustc-cfg=cargo_bindgen");

    Ok(())
}
