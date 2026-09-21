#!/bin/sh
set -eu

revision=23749fefcc72300e3a2ad315e1317431b06b590a
filename=Qwen3-0.6B-Q8_0.gguf
expected=9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
destination="${JET_MODEL_DIR:-models}/$filename"
partial="$destination.partial"
url="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/$revision/$filename"

mkdir -p "$(dirname "$destination")"

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [ -f "$destination" ] && [ "$(checksum "$destination")" = "$expected" ]; then
    printf '%s\n' "$destination"
    exit 0
fi

curl --fail --location --retry 5 --continue-at - --output "$partial" "$url"
actual=$(checksum "$partial")
if [ "$actual" != "$expected" ]; then
    printf 'SHA-256 mismatch: expected %s, got %s\n' "$expected" "$actual" >&2
    exit 1
fi
mv "$partial" "$destination"
printf '%s\n' "$destination"
