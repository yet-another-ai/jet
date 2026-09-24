#Requires -Version 7.0
param(
    [string]$CudaToolkitRoot = $(if ($env:CUDAToolkit_ROOT) { $env:CUDAToolkit_ROOT } else { $env:CUDA_PATH }),
    [ValidateSet('cuda', 'cuda,vulkan', 'cuda,vision', 'cuda,vulkan,vision')]
    [string]$Features = 'cuda,vulkan',
    [string]$OutputDirectory,
    [string]$BinaryPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($IsWindows -ne $true) {
    throw 'CUDA runtime packaging currently targets Windows; Linux CUDA libraries are linked statically.'
}
if (-not $CudaToolkitRoot) {
    throw 'Set CUDAToolkit_ROOT (or CUDA_PATH) to the CUDA Toolkit used for the build.'
}
$toolkit = (Resolve-Path -LiteralPath $CudaToolkitRoot).Path
$bin = Join-Path $toolkit 'bin'
if (-not (Test-Path -LiteralPath $bin -PathType Container)) {
    throw "CUDA Toolkit binary directory is missing: $bin"
}
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dist = [System.IO.Path]::GetFullPath((Join-Path $repo 'dist'))
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $dist 'jet-cuda-windows-x64'
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not $OutputDirectory.StartsWith(($dist.TrimEnd('\') + '\'), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must be a child of $dist"
}
$archive = "$OutputDirectory.zip"

function Assert-SafeOutputPath([string]$Path) {
    $cursor = [System.IO.Path]::GetFullPath($Path)
    while ($cursor.Length -ge $dist.Length) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes.HasFlag([System.IO.FileAttributes]::ReparsePoint)) {
                throw "Refusing to modify an output path through a reparse point: $cursor"
            }
        }
        if ($cursor -eq $dist) { break }
        $cursor = Split-Path -Parent $cursor
    }
}
Assert-SafeOutputPath $OutputDirectory

if (-not $BinaryPath) {
    $nvcc = Join-Path $bin 'nvcc.exe'
    if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
        throw "CUDA compiler is missing: $nvcc"
    }
    $env:CUDAToolkit_ROOT = $toolkit
    $env:CUDA_PATH = $toolkit
    $env:CUDACXX = $nvcc
    $env:PATH = "$bin;$env:PATH"
    Push-Location $repo
    try {
        & mise exec -- cargo build --release -p jet-cli --features $Features
        if ($LASTEXITCODE -ne 0) { throw "CUDA build failed ($LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $repo 'target' }
    $BinaryPath = Join-Path $target 'release/jet.exe'
}
$BinaryPath = (Resolve-Path -LiteralPath $BinaryPath).Path

$readobj = Get-Command llvm-readobj -ErrorAction SilentlyContinue
if (-not $readobj) {
    throw 'llvm-readobj is required to inspect DLL imports; run through mise or install LLVM.'
}

function Get-Imports([string]$Path) {
    $lines = & $readobj.Source --coff-imports $Path
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect imports: $Path" }
    @($lines | ForEach-Object {
        if ($_ -match '^\s+Name: ([^\\/]+\.dll)\s*$') { $Matches[1] }
    } | Sort-Object -Unique)
}

$toolkitDlls = @{}
foreach ($file in Get-ChildItem -LiteralPath $bin -File -Filter '*.dll') {
    $toolkitDlls[$file.Name.ToLowerInvariant()] = $file.FullName
}
$redistributable = '^(?i:cublas(?:Lt)?64_|cudart64_|nvrtc64_|nvrtc-builtins64_|(?:lib)?nvJitLink(?:64)?_)'
$cudaDependency = '^(?i:cublas|cudart|nvrtc|(?:lib)?nvjitlink|nvfatbin|nvptxcompiler)'

$binaryImports = Get-Imports $BinaryPath
if (-not ($binaryImports | Where-Object { $_ -match '^(?i:cublas64_).*\.dll$' })) {
    throw "The executable does not import a CUDA cuBLAS DLL: $BinaryPath"
}

# Build in a fresh staging directory; never reuse DLLs from an older Toolkit.
$staging = Join-Path (Split-Path -Parent $OutputDirectory) ('.jet-cuda-stage-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging -Force | Out-Null
try {
    Copy-Item -LiteralPath $BinaryPath -Destination (Join-Path $staging 'jet.exe')
    $pending = [System.Collections.Generic.Queue[string]]::new()
    $pending.Enqueue((Join-Path $staging 'jet.exe'))
    $copied = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    # cuBLAS may load cuBLASLt and cudart internally, without PE import entries.
    foreach ($import in $binaryImports) {
        if ($import -notmatch '^(?i:cublas64_)([^.]+)\.dll$') { continue }
        foreach ($name in @("cublasLt64_$($Matches[1]).dll", "cudart64_$($Matches[1]).dll")) {
            $key = $name.ToLowerInvariant()
            if (-not $toolkitDlls.ContainsKey($key)) { throw "Required CUDA runtime DLL $name was not found in $bin" }
            if ($copied.Add($key)) {
                $destination = Join-Path $staging $name
                Copy-Item -LiteralPath $toolkitDlls[$key] -Destination $destination
                $pending.Enqueue($destination)
            }
        }
    }
    while ($pending.Count -gt 0) {
        $current = $pending.Dequeue()
        foreach ($import in Get-Imports $current) {
            $key = $import.ToLowerInvariant()
            if ($key -eq 'nvcuda.dll') { continue }
            if ($toolkitDlls.ContainsKey($key) -and $copied.Add($key)) {
                if ($key -notmatch $redistributable) {
                    throw "Review redistribution terms for Toolkit DLL before packaging: $import"
                }
                $destination = Join-Path $staging $import
                Copy-Item -LiteralPath $toolkitDlls[$key] -Destination $destination
                $pending.Enqueue($destination)
            } elseif ($key -match $cudaDependency -and
                      -not $copied.Contains($key)) {
                throw "Required CUDA DLL $import (imported by $(Split-Path -Leaf $current)) was not found in $bin"
            }
        }
    }
    if (-not ($copied | Where-Object { $_ -match '^cublas64_' })) { throw 'cuBLAS DLL was not packaged.' }

    $notice = @'
Jet bundles CUDA Toolkit runtime libraries required by this executable.
The NVIDIA CUDA Toolkit license and redistributable-library terms apply:
https://docs.nvidia.com/cuda/eula/
The NVIDIA GPU driver (nvcuda.dll) is supplied by the user's driver installation.
'@
    Set-Content -LiteralPath (Join-Path $staging 'CUDA-NOTICE.txt') -Value $notice
    $manifest = Get-ChildItem -LiteralPath $staging -File | Sort-Object Name | ForEach-Object {
        [pscustomobject]@{ name = $_.Name; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    ConvertTo-Json -InputObject @($manifest) | Set-Content -LiteralPath (Join-Path $staging 'SHA256.json')

    Assert-SafeOutputPath $OutputDirectory
    if (Test-Path -LiteralPath $OutputDirectory) { Remove-Item -LiteralPath $OutputDirectory -Recurse -Force }
    Move-Item -LiteralPath $staging -Destination $OutputDirectory
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $archive
    Write-Host "Packaged $archive"
    Get-ChildItem -LiteralPath $OutputDirectory -File | Select-Object Name, Length
} finally {
    if (Test-Path -LiteralPath $staging) {
        Assert-SafeOutputPath $staging
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
