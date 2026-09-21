use std::env;
use std::path::{Path, PathBuf};

#[cfg(not(feature = "generate-bindings"))]
use std::fs;
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

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut config = cmake::Config::new(&source_dir);
    config
        .profile("Release")
        .build_target("llama-common")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_STATIC", "ON")
        .define("GGML_BACKEND_DL", "OFF")
        .define("LLAMA_BUILD_COMMON", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_LLGUIDANCE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_BLAS", "OFF")
        .define("GGML_CUDA", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_RPC", "OFF")
        .define("GGML_VULKAN", "OFF")
        .define("GGML_HIP", "OFF")
        .define("GGML_SYCL", "OFF");

    let dst = config.build();

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(&wrapper)
        .include(&include_dir)
        .include(&common_dir)
        .include(&vendor_dir)
        .include(&ggml_include_dir)
        .warnings(false)
        .compile("jet_llama_wrapper");

    for path in [
        dst.join("build/common"),
        dst.join("build/src"),
        dst.join("build/ggml/src"),
        dst.join("build/ggml/src/ggml-cpu"),
        dst.join("build/vendor/cpp-httplib"),
        dst.join("lib"),
        dst.join("lib64"),
    ] {
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
    match target_os.as_str() {
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
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
