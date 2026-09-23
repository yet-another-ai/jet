#!/bin/sh
set -eu

model_dir=${JET_MODEL_DIR:-models}
selection=qwen35
if [ "$#" -gt 0 ]; then
    if [ "$#" -ne 1 ] || { [ "$1" != "--qwen36" ] && [ "$1" != "--qwen36-vision" ]; }; then
        printf 'Usage: %s [--qwen36|--qwen36-vision]\n' "$0" >&2
        exit 2
    fi
    selection=${1#--}
fi

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    repo=$1
    revision=$2
    remote_filename=$3
    local_filename=$4
    expected=$5
    destination="$model_dir/$local_filename"
    partial="$destination.partial"
    url="https://huggingface.co/$repo/resolve/$revision/$remote_filename"

    if [ -f "$destination" ] && [ "$(checksum "$destination")" = "$expected" ]; then
        printf '%s\n' "$destination"
        return
    fi

    curl --fail --location --retry 5 --retry-all-errors --continue-at - --output "$partial" "$url"
    actual=$(checksum "$partial")
    if [ "$actual" != "$expected" ]; then
        printf '%s: SHA-256 mismatch: expected %s, got %s\n' \
            "$local_filename" "$expected" "$actual" >&2
        exit 1
    fi
    mv "$partial" "$destination"
    printf '%s\n' "$destination"
}

mkdir -p "$model_dir"

if [ "$selection" = qwen36 ] || [ "$selection" = qwen36-vision ]; then
    download \
        ggml-org/Qwen3.6-35B-A3B-GGUF \
        baec3ebee244827cda0f4557eafa8b28f7545fa6 \
        Qwen3.6-35B-A3B-Q4_K_M.gguf \
        Qwen3.6-35B-A3B-Q4_K_M.gguf \
        671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7
    if [ "$selection" = qwen36-vision ]; then
        download \
            ggml-org/Qwen3.6-35B-A3B-GGUF \
            baec3ebee244827cda0f4557eafa8b28f7545fa6 \
            mmproj-Qwen3.6-35B-A3B-Q8_0.gguf \
            mmproj-Qwen3.6-35B-A3B-Q8_0.gguf \
            904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d
    fi
    exit 0
fi

download \
    bartowski/Qwen_Qwen3.5-0.8B-GGUF \
    f36b1ea49a332ede8fe5f389bbf5b3575ef71f48 \
    Qwen_Qwen3.5-0.8B-Q8_0.gguf \
    Qwen3.5-0.8B-Q8_0.gguf \
    7182e2362766bb9569209bbc24cf1a4cdfbb8ab161babdb2080c84fa62c08c2f

download \
    bartowski/Qwen_Qwen3.5-2B-GGUF \
    7d26695454df6de5fbcce2e58681e62dae06ce43 \
    Qwen_Qwen3.5-2B-Q8_0.gguf \
    Qwen3.5-2B-Q8_0.gguf \
    be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2
