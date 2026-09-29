"""Merge a JET LoRA into its pinned BF16 text base and optionally export GGUF."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .train import load_config, load_text_model, package_versions, sha256, write_json


def adapter_provenance(adapter: Path, config: dict) -> dict:
    marker = adapter / "jet_training.json"
    if not marker.is_file():
        # Standard Trainer checkpoint directories retain their run-level provenance.
        marker = adapter.parent / "run.json"
    if not marker.is_file():
        raise ValueError("adapter requires jet_training.json or parent run.json provenance")
    provenance = json.loads(marker.read_text(encoding="utf-8"))
    if provenance.get("identity", {}).get("model") != config["model"]:
        raise ValueError("adapter provenance and configured pinned base model differ")
    adapter_config_path = adapter / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != config["model"]["id"]:
        raise ValueError("adapter base model does not match configured base")
    if adapter_config.get("revision") != config["model"]["revision"]:
        raise ValueError("adapter base revision is missing or does not match configured revision")
    for filename in ("adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json"):
        if not (adapter / filename).is_file():
            raise ValueError(f"adapter is incomplete: missing {filename}")
    return provenance


def conversion_command(output: Path, gguf: Path, outtype: str, converter: Path) -> list[str]:
    return [
        sys.executable,
        str(converter),
        str(output),
        "--outfile",
        str(gguf),
        "--outtype",
        outtype,
        # The text-only HF model excludes MTP weights even though its config
        # retains mtp_num_hidden_layers=1. Keep GGUF block_count at 32.
        "--no-nextn",
    ]


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    adapter, output = args.adapter.resolve(), args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"export output directory already exists: {output}")
    gguf = args.gguf_output.resolve() if args.gguf_output else None
    if gguf and gguf.exists():
        raise ValueError(f"GGUF output already exists: {gguf}")
    converter = Path(__file__).resolve().parents[2] / "vendor/llama.cpp/convert_hf_to_gguf.py"
    if not converter.is_file():
        raise ValueError("initialize the pinned vendor/llama.cpp submodule before export")
    provenance = adapter_provenance(adapter, config)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if args.device == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("CUDA merge requires a GPU supporting BF16")
    base = load_text_model(config)
    if args.device == "cuda":
        base = base.to("cuda")
    trained = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    merged = trained.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
    merged.config.use_cache = True
    merged.config.architectures = ["Qwen3_5ForCausalLM"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter), local_files_only=True, trust_remote_code=False
    )
    output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    record = {
        "base_model": config["model"],
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "training_identity": provenance["identity"],
        "packages": package_versions(),
        "dtype": "bfloat16",
        "architecture": "Qwen3_5ForCausalLM",
        "text_only": True,
        "merged_directory": str(output),
        "converter_sha256": sha256(converter),
        "gguf_command": conversion_command(
            output, gguf or output.with_suffix(".Q8_0.gguf"), args.gguf_type, converter
        ),
        "gguf_completed": False,
    }
    write_json(output / "export.json", record)
    # Release the merged model before running conversion in a second process.
    del merged, trained, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if gguf:
        gguf.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(record["gguf_command"], check=True)
        if not gguf.is_file() or gguf.stat().st_size == 0:
            raise ValueError("converter completed without a nonempty GGUF file")
        record.update(
            {
                "gguf_completed": True,
                "gguf_path": str(gguf),
                "gguf_sha256": sha256(gguf),
                "gguf_bytes": gguf.stat().st_size,
            }
        )
        write_json(output / "export.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--gguf-output", type=Path, help="Optional new GGUF path; invokes pinned converter"
    )
    parser.add_argument("--gguf-type", choices=("q8_0", "bf16", "f16"), default="q8_0")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, ImportError, subprocess.CalledProcessError) as error:
        print(f"jet-export: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
