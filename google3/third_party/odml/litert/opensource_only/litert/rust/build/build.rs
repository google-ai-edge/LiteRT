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

use build_print::{info, println};
use std::env;
use std::fs::{self, File};
use std::io::{self, copy};
use std::path::{Path, PathBuf};
use std::process::Command;

// Constansts that are used by the build script.

// Environment variable that contains the output directory where cargo wants use to build things.
const OUT_DIR_ENV_VAR: &str = "OUT_DIR";
const RUST_LITERT_SOURCE_DIR: &str = "RUST_LITERT_SOURCE_DIR";
const RUST_LITERT_RUNTIME_LIBRARY_DIR: &str = "RUST_LITERT_RUNTIME_LIBRARY_DIR";

// The URL of LiterRT release archive on Github.
const LITERT_RELEASE_ARCHIVE_URL: &str =
    "https://github.com/google-ai-edge/LiteRT/archive/refs/heads/main.zip";

// Different paths that are used during the binary build process.
const DOCKER_BUILD_SCRIPT_PATH: &str = "docker_build/build_with_docker.sh";
const LITERT_RUNTIME_LIBRARY_DIR: &str = "litert_runtime";
const DOCKER_BUILT_RUNTIME_LIBRARY_DIR: &str =
    "litert_build_container:/litert_build/bazel-bin/litert/runtime/libcompiled_model.a";

// Different paths related to preparing sources
const BUILD_CONFIG_H_IN: &str = "build/build_config.h";
const BUILD_CONFIG_H_OUT: &str = "litert/build_common/build_config.h";

// A helper macro to panic with a clear message if a command fails
macro_rules! run_command {
    ($cmd:expr) => {
        let status = $cmd.status().expect("Failed to execute command");
        if !status.success() {
            panic!("Command failed: {:?}", $cmd);
        }
    };
}

// Helper function to check for tools
fn check_tool_installed(tool: &str) -> Result<(), String> {
    match Command::new(tool).arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(format!("Required tool '{}' is not installed or not in PATH.", tool)),
    }
}

// Helper function to download a file
fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut response = reqwest::blocking::get(url)?;
    let mut dest = File::create(path)?;
    copy(&mut response, &mut dest)?;
    Ok(())
}

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
    if let Ok(litert_dir) = env::var(RUST_LITERT_SOURCE_DIR) {
        println!("Using LiteRT sources from {}", litert_dir);
        return Ok(PathBuf::from(litert_dir));
    }
    let out_dir = PathBuf::from(env::var(OUT_DIR_ENV_VAR)?);
    let download_archive_path = out_dir.join("litert_source.zip");
    info!("Downloading LiteRT sources to {}...", download_archive_path.display());
    download_file(LITERT_RELEASE_ARCHIVE_URL, &download_archive_path)?;
    info!("Unzipping LiteRT sources to {}...", out_dir.display());
    let zip_root_dir = unzip_sources(&download_archive_path, &out_dir)?;

    let source_root_dir = out_dir.join(zip_root_dir);
    Ok(source_root_dir)
}

// Some sources need to be modified based on the configuration.
// It's done by Bazel or Cmake. In case of Rust, this is simply a hack that renames a file.
fn prepare_sources(sources_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    info!("Preparing sources");
    let build_config_in_path = Path::new(BUILD_CONFIG_H_IN);
    let build_config_out_path = sources_path.join(BUILD_CONFIG_H_OUT);
    if build_config_in_path.exists() && !build_config_out_path.exists() {
        info!(
            "Copy file from {} to {}",
            build_config_in_path.display(),
            build_config_out_path.display()
        );
        fs::copy(build_config_in_path, build_config_out_path)?;
    }
    Ok(())
}

// Uses existing LiteRT runtime library of build it with docker.
fn get_litert_runtime_library(
    litert_source_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(litert_dir) = env::var(RUST_LITERT_RUNTIME_LIBRARY_DIR) {
        println!("Using LiteRT runtime library from {}", litert_dir);
        return Ok(PathBuf::from(litert_dir));
    }
    println!("Building LiteRT runtime library with docker...");
    check_tool_installed("docker")?;

    let build_script_path = litert_source_dir.join(DOCKER_BUILD_SCRIPT_PATH);
    run_command!(Command::new("bash").current_dir(&litert_source_dir).arg(&build_script_path));
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let litert_runtime_dir = out_dir.join(LITERT_RUNTIME_LIBRARY_DIR);
    println!("Copying LiteRT runtime library built to {}", litert_runtime_dir.display());
    run_command!(Command::new("docker")
        .arg("cp")
        .arg(DOCKER_BUILT_RUNTIME_LIBRARY_DIR)
        .arg(&litert_runtime_dir));
    Ok(litert_runtime_dir)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build/build.rs");

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        println!("Manifest dir {}", manifest_dir);
    }
    let litert_source_dir = get_litert_sources()?;
    let litert_runtime_dir = get_litert_runtime_library(&litert_source_dir)?;
    prepare_sources(&litert_source_dir)?;

    println!("cargo:rustc-link-search=native={}", litert_runtime_dir.display());
    println!("cargo:rustc-link-lib=static=compiled_model");

    check_tool_installed("clang")?;
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // Add the include path so clang can find dependent headers
        .clang_arg(format!("-I{}", litert_source_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    // Get the output directory where cargo wants us to build things
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let bindings_out_path = out_dir.join("bindings.rs");
    bindings.write_to_file(bindings_out_path).expect("Couldn't write bindings!");

    Ok(())
}
