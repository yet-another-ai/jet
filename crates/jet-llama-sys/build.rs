use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "generate-bindings")]
use std::process::{Command, Stdio};

#[cfg(feature = "generate-bindings")]
fn compiler_system_include_args() -> Vec<String> {
    let compiler = env::var_os("CC").unwrap_or_else(|| "cc".into());
    let Ok(output) = Command::new(compiler)
        .args(["-E", "-x", "c", "-", "-v"])
        .stdin(Stdio::null())
        .output()
    else {
        return Vec::new();
    };
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    let mut in_search_list = false;
    diagnostics
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line == "#include <...> search starts here:" {
                in_search_list = true;
                return None;
            }
            if line == "End of search list." {
                in_search_list = false;
                return None;
            }
            (in_search_list && Path::new(line).is_dir()).then(|| format!("-isystem{line}"))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let workspace_root = manifest_dir
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.clone());
    let source_dir = env::var_os("LLAMA_CPP_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("vendor/llama.cpp"));
    let include_dir = source_dir.join("include");
    let ggml_include_dir = source_dir.join("ggml/include");
    let common_dir = source_dir.join("common");
    let vendor_dir = source_dir.join("vendor");
    let wrapper = manifest_dir.join("src/wrapper.cpp");
    let wrapper_header = manifest_dir.join("src/wrapper.h");
    let checked_bindings = manifest_dir.join("src/bindings.rs");

    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("llama.h").display()
    );
    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!("cargo:rerun-if-changed={}", wrapper_header.display());
    println!("cargo:rerun-if-changed={}", checked_bindings.display());
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=JET_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let metal = env::var_os("CARGO_FEATURE_METAL").is_some();
    let vision = env::var_os("CARGO_FEATURE_VISION").is_some();
    if cuda && !matches!(target_os.as_str(), "windows" | "linux") {
        return Err("the cuda feature currently supports Windows and Linux targets".into());
    }
    if metal && target_os != "macos" {
        return Err("the metal feature currently supports macOS targets".into());
    }
    let mut config = cmake::Config::new(&source_dir);
    config
        .profile("Release")
        .build_target(if vision { "mtmd" } else { "llama-common" })
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_STATIC", "ON")
        .define("GGML_BACKEND_DL", "OFF")
        .define("LLAMA_BUILD_COMMON", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_MTMD", if vision { "ON" } else { "OFF" })
        .define("MTMD_VIDEO", "OFF")
        .define("LLAMA_LLGUIDANCE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_BLAS", "OFF")
        .define(
            "GGML_ACCELERATE",
            if matches!(target_os.as_str(), "macos" | "ios") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define("GGML_CUDA", if cuda { "ON" } else { "OFF" })
        .define("GGML_CUDA_NCCL", "OFF")
        .define("GGML_METAL", if metal { "ON" } else { "OFF" })
        .define("GGML_METAL_EMBED_LIBRARY", if metal { "ON" } else { "OFF" })
        .define("GGML_RPC", "OFF")
        .define("GGML_VULKAN", if vulkan { "ON" } else { "OFF" })
        .define("GGML_HIP", "OFF")
        .define("GGML_SYCL", "OFF");
    if cuda {
        if let Some(architectures) = env::var_os("JET_CUDA_ARCHITECTURES") {
            config.define("CMAKE_CUDA_ARCHITECTURES", architectures);
        }
        println!(
            "cargo:rerun-if-changed={}",
            source_dir
                .join("ggml/src/ggml-cuda/CMakeLists.txt")
                .display()
        );
    }
    if metal {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join("ggml/src/ggml-metal").display()
        );
    }

    let dst = config.build();
    if vision {
        // Both targets share one configured build tree; mtmd does not depend on common.
        config.build_target("llama-common").build();
    }

    let mut wrapper_build = cc::Build::new();
    wrapper_build
        .cpp(true)
        .std("c++17")
        .file(&wrapper)
        .include(&include_dir)
        .include(&common_dir)
        .include(&vendor_dir)
        .include(&ggml_include_dir)
        .warnings(false);
    if vision {
        wrapper_build
            .file(manifest_dir.join("src/vision.cpp"))
            .include(source_dir.join("tools/mtmd"));
    }
    wrapper_build.compile("jet_llama_wrapper");

    let mut link_paths = vec![
        dst.join("build/common"),
        dst.join("build/src"),
        dst.join("build/ggml/src"),
        dst.join("build/ggml/src/ggml-cpu"),
        dst.join("build/vendor/cpp-httplib"),
        dst.join("build/tools/mtmd"),
        dst.join("build/vendor/hash"),
        dst.join("lib"),
        dst.join("lib64"),
    ];
    if vulkan {
        link_paths.push(dst.join("build/ggml/src/ggml-vulkan"));
        emit_vulkan_sdk_search_paths(&target_os);
    }
    if cuda {
        link_paths.push(dst.join("build/ggml/src/ggml-cuda"));
    }
    if metal {
        link_paths.push(dst.join("build/ggml/src/ggml-metal"));
    }
    for path in link_paths {
        // Visual Studio is a multi-configuration generator and puts static
        // libraries beneath the selected configuration, even for Cargo debug.
        if target_env == "msvc" {
            println!(
                "cargo:rustc-link-search=native={}",
                path.join("Release").display()
            );
        }
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for library in [
        "llama-common",
        "llama-common-base",
        "cpp-httplib",
        "llama",
        "ggml",
        "ggml-base",
        "ggml-cpu",
    ] {
        println!("cargo:rustc-link-lib=static={library}");
    }
    if vision {
        println!("cargo:rustc-link-lib=static=mtmd");
        println!("cargo:rustc-link-lib=static=vendor-hash");
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join("src/vision.cpp").display()
        );
    }
    if vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    if cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        emit_cuda_links(&dst.join("build/CMakeCache.txt"), &target_os)?;
    }
    if metal {
        println!("cargo:rustc-link-lib=static=ggml-metal");
        for framework in ["Foundation", "Metal", "MetalKit"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
    }
    match target_os.as_str() {
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            if vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            if vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "windows" => {
            // ggml-cpu reads Windows processor information from the registry.
            println!("cargo:rustc-link-lib=advapi32");
            if vulkan {
                let library = if target_env == "msvc" {
                    "vulkan-1"
                } else {
                    "vulkan"
                };
                println!("cargo:rustc-link-lib={library}");
            }
        }
        _ => {}
    }

    let out_bindings = PathBuf::from(env::var("OUT_DIR").unwrap_or_default()).join("bindings.rs");
    #[cfg(feature = "generate-bindings")]
    {
        let mut binding_builder = bindgen::Builder::default()
            .header(wrapper_header.display().to_string())
            .clang_arg(format!("-I{}", include_dir.display()))
            .clang_arg(format!("-I{}", ggml_include_dir.display()))
            .allowlist_type("llama_.*")
            .allowlist_function("llama_.*")
            .allowlist_var("LLAMA_.*")
            .allowlist_function("jet_.*")
            // FILE is only passed by pointer. Its libc-specific layout must
            // never enter the checked bindings used across supported targets.
            .blocklist_type("FILE")
            .raw_line("#[repr(C)] pub struct FILE { _unused: [u8; 0] }")
            .derive_default(true)
            .generate_comments(false);
        for argument in compiler_system_include_args() {
            binding_builder = binding_builder.clang_arg(argument);
        }
        let bindings = binding_builder.generate()?;
        bindings.write_to_file(&checked_bindings)?;
        bindings.write_to_file(out_bindings)?;
    }
    #[cfg(not(feature = "generate-bindings"))]
    {
        fs::copy(checked_bindings, out_bindings)?;
    }
    Ok(())
}

fn emit_vulkan_sdk_search_paths(target_os: &str) {
    let Some(sdk) = env::var_os("VULKAN_SDK").map(PathBuf::from) else {
        return;
    };
    let candidates: &[&str] = match target_os {
        "windows" => &["Lib"],
        "macos" | "ios" => &["lib", "macOS/lib", "Lib"],
        _ => &["lib", "Lib"],
    };
    for candidate in candidates {
        let path = sdk.join(candidate);
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
}

fn emit_cuda_links(cache_path: &Path, target_os: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cache = fs::read_to_string(cache_path)?;
    let libraries: &[(&str, &str, &str, bool)] = if target_os == "windows" {
        &[
            (
                "CUDA_cudart_static_LIBRARY",
                "static",
                "cudart_static",
                true,
            ),
            ("CUDA_cublas_LIBRARY", "dylib", "cublas", true),
            ("CUDA_cublasLt_LIBRARY", "dylib", "cublasLt", true),
            ("CUDA_culibos_LIBRARY", "static", "culibos", false),
            ("CUDA_cuda_driver_LIBRARY", "dylib", "cuda", true),
        ]
    } else {
        &[
            (
                "CUDA_cudart_static_LIBRARY",
                "static",
                "cudart_static",
                true,
            ),
            (
                "CUDA_cublas_static_LIBRARY",
                "static",
                "cublas_static",
                true,
            ),
            (
                "CUDA_cublasLt_static_LIBRARY",
                "static",
                "cublasLt_static",
                true,
            ),
            ("CUDA_culibos_LIBRARY", "static", "culibos", true),
            ("CUDA_cuda_driver_LIBRARY", "dylib", "cuda", true),
        ]
    };
    for &(variable, kind, library, required) in libraries {
        let prefix = format!("{variable}:");
        let value = cache
            .lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| line.split_once('=').map(|(_, value)| value))
            .filter(|value| !value.is_empty() && !value.ends_with("-NOTFOUND"));
        let Some(value) = value else {
            if required {
                return Err(format!("CMake did not resolve {variable} for the CUDA build").into());
            }
            continue;
        };
        let path = Path::new(value);
        let directory = path
            .parent()
            .ok_or_else(|| format!("invalid CUDA library path for {variable}: {value}"))?;
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib={kind}={library}");
    }
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=rt");
    }
    Ok(())
}
