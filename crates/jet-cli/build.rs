use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");
    if env::var_os("CARGO_FEATURE_CUDA").is_none()
        || env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc")
    {
        return Ok(());
    }

    let bin = cuda_bin_dir().ok_or(
        "CUDA Toolkit bin directory not found; set CUDAToolkit_ROOT or CUDA_PATH for the CUDA CLI build",
    )?;
    let blas_dlls = fs::read_dir(&bin)?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name())
        .filter_map(|name| name.into_string().ok())
        .filter(|name| {
            let name = name.to_ascii_lowercase();
            name.starts_with("cublas64_") && name.ends_with(".dll")
        })
        .collect::<Vec<_>>();
    let [blas] = blas_dlls.as_slice() else {
        return Err(format!("expected one cuBLAS DLL in {}", bin.display()).into());
    };
    let suffix = &blas["cublas".len()..];
    let lt = format!("cublasLt{suffix}");
    if !bin.join(&lt).is_file() {
        return Err(format!("missing CUDA DLL {}", bin.join(&lt).display()).into());
    }

    // Delay CUDA imports so a machine without an NVIDIA driver can still start
    // the CLI and let --backend auto select Vulkan or CPU.
    for dll in [blas.as_str(), lt.as_str(), "nvcuda.dll"] {
        println!("cargo:rustc-link-arg=/DELAYLOAD:{dll}");
    }
    println!("cargo:rustc-link-lib=delayimp");
    Ok(())
}

fn cuda_bin_dir() -> Option<PathBuf> {
    for name in ["CUDAToolkit_ROOT", "CUDA_PATH"] {
        if let Some(root) = env::var_os(name) {
            let bin = PathBuf::from(root).join("bin");
            if bin.is_dir() {
                return Some(bin);
            }
        }
    }
    if let Some(compiler) = env::var_os("CUDACXX")
        && let Some(bin) = Path::new(&compiler).parent().filter(|path| path.is_dir())
    {
        return Some(bin.to_path_buf());
    }
    env::split_paths(&env::var_os("PATH")?).find(|dir| dir.join("nvcc.exe").is_file())
}
