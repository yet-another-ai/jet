#Requires -Version 7.0
param(
    [ValidateSet('windows-x64', 'linux-x64', 'linux-arm64', 'macos-arm64')]
    [string]$Platform
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
$os = if ($IsWindows) { 'windows' } elseif ($IsMacOS) { 'macos' } elseif ($IsLinux) { 'linux' } else { throw 'Unsupported OS' }
$hostPlatform = "$os-$arch"
if (-not $Platform) { $Platform = $hostPlatform }
if ($Platform -ne $hostPlatform) { throw "Native packaging requires $Platform; current host is $hostPlatform" }

Push-Location $repo
$previousPortable = $env:JET_CPU_PORTABLE
$previousRustFlags = $env:RUSTFLAGS
try {
    $env:JET_CPU_PORTABLE = '1'
    if ($IsWindows) {
        if ($env:CARGO_ENCODED_RUSTFLAGS) { throw 'Unset CARGO_ENCODED_RUSTFLAGS for static Windows CPU packaging' }
        if ($previousRustFlags -notmatch 'target-feature=\+crt-static(?:\s|$)') {
            $env:RUSTFLAGS = "$previousRustFlags -C target-feature=+crt-static".Trim()
        }
    }
    # No optional GPU or vision feature, no model/dataset downloads.
    & cargo build --locked --release -p jet-cli --no-default-features
    if ($LASTEXITCODE -ne 0) { throw "CPU build failed ($LASTEXITCODE)" }
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { 'target' }
    $binaryName = if ($IsWindows) { 'jet.exe' } else { 'jet' }
    $binary = Join-Path $target "release/$binaryName"
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) { throw "Missing CLI: $binary" }
    $name = "jet-cpu-$Platform"
    $dist = Join-Path $repo 'dist'
    New-Item -ItemType Directory -Path $dist -Force | Out-Null
    $extension = if ($IsWindows) { '.zip' } else { '.tar.gz' }
    $archive = Join-Path $dist "$name$extension"
    if ((Test-Path -LiteralPath $archive) -or (Test-Path -LiteralPath "$archive.sha256")) {
        throw "Output already exists: $archive"
    }
    $stage = Join-Path $dist ('.cpu-stage-' + [guid]::NewGuid().ToString('N'))
    $payload = Join-Path $stage $name
    New-Item -ItemType Directory -Path $payload -Force | Out-Null
    Copy-Item -LiteralPath $binary -Destination (Join-Path $payload $binaryName)
    Copy-Item -LiteralPath 'LICENSE', 'README.md' -Destination $payload
    Copy-Item -LiteralPath 'vendor/llama.cpp/LICENSE' -Destination (Join-Path $payload 'LLAMA-LICENSE')
    Copy-Item -LiteralPath 'tests/fixtures/decisions.jsonl' -Destination $payload
    $commit = & git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot identify source commit' }
    $llama = & git rev-parse HEAD:vendor/llama.cpp
    if ($LASTEXITCODE -ne 0) { throw 'Cannot identify pinned llama.cpp' }
    [ordered]@{ platform = $Platform; backend = 'cpu'; features = @(); portable_cpu = $true; commit = $commit; llama_cpp = $llama } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $payload 'BUILD.json') -Encoding utf8
    if ($IsWindows) {
        Compress-Archive -LiteralPath $payload -DestinationPath $archive
    } else {
        & tar -C $stage -czf $archive $name
        if ($LASTEXITCODE -ne 0) { throw 'Archive creation failed' }
    }
    $digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    "$digest  $(Split-Path -Leaf $archive)" | Set-Content -LiteralPath "$archive.sha256" -Encoding ascii

    # Exercise the actual extracted artifact, with no model and outside the repo cwd.
    $extracted = Join-Path $stage 'extracted'
    New-Item -ItemType Directory -Path $extracted | Out-Null
    if ($IsWindows) {
        Expand-Archive -LiteralPath $archive -DestinationPath $extracted
    } else {
        & tar -C $extracted -xzf $archive
        if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }
    }
    Push-Location (Join-Path $extracted $name)
    try {
        $cli = Join-Path $PWD $binaryName
        & $cli --version
        if ($LASTEXITCODE -ne 0) { throw 'Packaged CLI version smoke failed' }
        & $cli --help | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Packaged CLI help smoke failed' }
        $responses = @(Get-Content -LiteralPath decisions.jsonl | & $cli export-prompts)
        if ($LASTEXITCODE -ne 0) { throw 'Packaged prompt exporter smoke failed' }
        $requests = @(Get-Content -LiteralPath decisions.jsonl | Where-Object { $_.Trim() })
        if ($responses.Count -ne $requests.Count) { throw 'Packaged exporter lost JSONL requests' }
        foreach ($response in $responses) {
            $parsed = $response | ConvertFrom-Json
            if (-not $parsed.questions -or $parsed.PSObject.Properties.Name -contains 'error') {
                throw 'Packaged exporter produced an invalid prompt response'
            }
        }
    } finally { Pop-Location }
    Write-Host "Packaged and verified $archive ($digest)"
} finally {
    $env:JET_CPU_PORTABLE = $previousPortable
    $env:RUSTFLAGS = $previousRustFlags
    Pop-Location
}
