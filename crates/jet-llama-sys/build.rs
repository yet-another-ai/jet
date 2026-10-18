use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "generate-bindings")]
use std::process::{Command, Stdio};

#[cfg(feature = "generate-bindings")]
fn binding_builder(
    wrapper_header: &Path,
    include_dir: &Path,
    ggml_include_dir: &Path,
) -> bindgen::Builder {
    bindgen::Builder::default()
        .header(wrapper_header.display().to_string())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", ggml_include_dir.display()))
        .allowlist_type("llama_.*")
        .allowlist_function("llama_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_function("jet_.*")
        // FILE is only passed by pointer. Its libc-specific layout must never
        // enter bindings shared by the Rust and C++ halves of Jet.
        .blocklist_type("FILE")
        .raw_line("#[repr(C)] pub struct FILE { _unused: [u8; 0] }")
        .derive_default(true)
        .generate_comments(false)
}

#[cfg(feature = "generate-bindings")]
fn extern_blocks(bindings: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    let mut extern_depth = 0_i32;
    for line in bindings.lines() {
        if extern_depth == 0 && line.trim() == "unsafe extern \"C\" {" {
            extern_depth = 1;
            output.push_str(line);
            output.push('\n');
            continue;
        }
        if extern_depth > 0 {
            output.push_str(line);
            output.push('\n');
            extern_depth += line.matches('{').count() as i32;
            extern_depth -= line.matches('}').count() as i32;
        }
    }
    if extern_depth != 0 {
        return Err("unterminated extern block in checked native bindings".into());
    }
    Ok(output)
}

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
    let checked_emscripten_bindings = manifest_dir.join("src/bindings_emscripten.rs");

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
    println!(
        "cargo:rerun-if-changed={}",
        checked_emscripten_bindings.display()
    );
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=JET_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=JET_CPU_PORTABLE");
    println!("cargo:rerun-if-env-changed=JET_EMSCRIPTEN_SYSROOT");

    let out_bindings = PathBuf::from(env::var("OUT_DIR").unwrap_or_default()).join("bindings.rs");
    if env::var_os("CARGO_FEATURE_EXTERNAL_LINK").is_some() {
        let target = env::var("TARGET")?;
        if target == "wasm32-unknown-emscripten" {
            if env::var_os("CARGO_FEATURE_WEBGPU").is_none() {
                return Err(
                    "external-link on wasm32-unknown-emscripten requires the webgpu feature".into(),
                );
            }
            // The native bindings contain 64-bit layout tests. Select bindings
            // generated against Emscripten's memory32 ABI.
            #[cfg(not(feature = "generate-bindings"))]
            {
                if !checked_emscripten_bindings.is_file() {
                    return Err(format!(
                        "missing {}; run `mise run playground:bindings` or `sh scripts/generate-emscripten-bindings.sh` before building the WebGPU playground",
                        checked_emscripten_bindings.display()
                    )
                    .into());
                }
                fs::copy(&checked_emscripten_bindings, out_bindings)?;
            }
            #[cfg(feature = "generate-bindings")]
            {
                let emscripten_sysroot = env::var("JET_EMSCRIPTEN_SYSROOT")?;
                let mut bindings =
                    binding_builder(&wrapper_header, &include_dir, &ggml_include_dir)
                        // Function declarations recursively pull these opaque ggml
                        // and Jet types into native bindings. Emscripten libclang
                        // omits those declarations, so allowlist the types directly.
                        .allowlist_type("ggml_.*")
                        .allowlist_type("gguf_.*")
                        .allowlist_type("jet_.*")
                        .clang_arg("--target=wasm32-unknown-emscripten")
                        .clang_arg(format!("--sysroot={emscripten_sysroot}"))
                        .generate()?
                        .to_string();
                // libclang omits static-link function declarations for the
                // Emscripten target. They carry no target layout, so reuse the
                // extern blocks generated from the same headers while keeping
                // every type and layout assertion Emscripten-generated.
                bindings.push_str(&extern_blocks(&fs::read_to_string(&checked_bindings)?)?);
                fs::write(&checked_emscripten_bindings, &bindings)?;
                fs::write(out_bindings, bindings)?;
            }
        } else {
            // Native all-feature checks do not link llama.cpp, but still need
            // bindings whose ABI matches their target.
            fs::copy(checked_bindings, out_bindings)?;
        }
        return Ok(());
    }

    if env::var_os("CARGO_FEATURE_WEBGPU").is_some() {
        return Err(
            "the webgpu feature requires external-link; Jet's WebGPU llama.cpp is built by the wllama/Emscripten pipeline"
                .into(),
        );
    }

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
    if target_env == "msvc"
        && env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .split(',')
            .any(|feature| feature == "crt-static")
    {
        // CMake's modern MSVC runtime property overrides the /MT compiler flag.
        config
            .static_crt(true)
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
    }
    // CI archives must run on other CPUs, not just the build runner's ISA.
    if env::var("JET_CPU_PORTABLE").as_deref() == Ok("1") {
        config.define("GGML_NATIVE", "OFF");
        for instruction in ["SSE42", "AVX", "AVX2", "BMI2", "FMA", "F16C"] {
            config.define(format!("GGML_{instruction}"), "OFF");
        }
        if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
            config.define("GGML_CPU_ARM_ARCH", "armv8-a");
        }
    }
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
    if target_env == "msvc" {
        // llama.cpp headers contain UTF-8 literals; do not use the host code page.
        wrapper_build.flag("/utf-8");
    }
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

    #[cfg(feature = "generate-bindings")]
    {
        let mut binding_builder = binding_builder(&wrapper_header, &include_dir, &ggml_include_dir);
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
