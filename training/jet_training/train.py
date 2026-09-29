"""Train a text-only BF16 Qwen3.5 LoRA using JET's answer tokens."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from .tokenization import AnswerOnlyCollator, TokenizedDataset, load_tokenized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    path = path.resolve()
    with path.open("rb") as source:
        config = tomllib.load(source)
    for section in ("model", "data", "lora", "training"):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"missing [{section}] configuration")
    if config["model"].get("id") != "Qwen/Qwen3.5-4B":
        raise ValueError("this recipe requires Qwen/Qwen3.5-4B")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model"].get("revision", "")):
        raise ValueError("model.revision must be an immutable 40-character commit SHA")
    for section, field in (
        ("data", "train_file"),
        ("data", "validation_file"),
        ("training", "output_dir"),
    ):
        value = config[section].get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{section}.{field} must be a path")
        config[section][field] = str((path.parent / value).resolve())
    for section, field in (
        ("data", "max_length"),
        ("data", "validation_max_samples"),
        ("lora", "r"),
        ("lora", "alpha"),
        ("training", "micro_batch_size"),
        ("training", "gradient_accumulation_steps"),
        ("training", "logging_steps"),
        ("training", "save_steps"),
        ("training", "eval_steps"),
        ("training", "save_total_limit"),
    ):
        value = config[section].get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{section}.{field} must be a positive integer")
    if not config["lora"].get("target_modules"):
        raise ValueError("lora.target_modules must not be empty")
    for section, field in (("training", "epochs"), ("training", "learning_rate")):
        value = config[section].get(field)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{section}.{field} must be positive")
    for section, field in (("lora", "dropout"), ("training", "warmup_steps")):
        value = config[section].get(field)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 <= value < 1:
            raise ValueError(f"{section}.{field} must be in [0, 1)")
    if not isinstance(config["training"].get("gradient_checkpointing"), bool):
        raise TypeError("training.gradient_checkpointing must be boolean")
    if not isinstance(config["training"].get("seed"), int):
        raise TypeError("training.seed must be an integer")
    if (
        not isinstance(config["training"].get("weight_decay"), (int, float))
        or config["training"]["weight_decay"] < 0
    ):
        raise ValueError("training.weight_decay must be nonnegative")
    if Path(config["data"]["train_file"]) == Path(config["data"]["validation_file"]):
        raise ValueError("training and validation must be different files")
    return config


def package_versions() -> dict:
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_text_model(config: dict):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    model_id, revision = config["model"]["id"], config["model"]["revision"]
    # The HF text config extracts text_config from the multimodal checkpoint.
    # Transformers' qwen3_5_text mapping removes model.language_model's prefix.
    text_config = Qwen3_5TextConfig.from_pretrained(model_id, revision=revision)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        config=text_config,
        dtype=torch.bfloat16,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=False,
        output_loading_info=True,
    )
    unexpected = [
        key
        for key in info.get("unexpected_keys", [])
        if not key.startswith(("model.visual.", "mtp.", "model.mtp."))
    ]
    if (
        info.get("missing_keys")
        or info.get("mismatched_keys")
        or unexpected
        or info.get("error_msgs")
    ):
        raise ValueError(f"text checkpoint did not load completely: {info}")
    if any("visual" in name for name, _ in model.named_parameters()):
        raise ValueError("text-only model unexpectedly includes vision parameters")
    model.config.use_cache = False
    return model


def add_lora(model, config: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    selected = config["lora"]["target_modules"]
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(selected) - names
    if missing:
        raise ValueError(f"LoRA targets absent from model: {sorted(missing)}")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=selected,
            bias="none",
            revision=config["model"]["revision"],
        ),
    )
    if any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in model.named_parameters()
    ):
        raise ValueError("unexpected trainable base parameter")
    model.print_trainable_parameters()
    return model


def validate_resume(output: Path, checkpoint: Path | None, identity: dict) -> None:
    if checkpoint is None:
        if output.exists():
            raise ValueError(
                f"output directory already exists: {output}; use a new run or explicit resume"
            )
        return
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != output.resolve():
        raise ValueError("resume checkpoint must be a direct child of configured output_dir")
    for filename in (
        "trainer_state.json",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"incomplete resume checkpoint: missing {filename}")
    manifest_path = output / "run.json"
    if not manifest_path.is_file():
        raise ValueError("resume requires original run.json provenance")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("identity") != identity:
        raise ValueError("resume config/model/data differ from original run")


def run(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir.resolve())
    if args.validation_max_samples is not None:
        config["data"]["validation_max_samples"] = args.validation_max_samples
    if args.micro_batch_size is not None:
        config["training"]["micro_batch_size"] = args.micro_batch_size
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    output = Path(config["training"]["output_dir"])
    identity = {
        "config_sha256": sha256(config_path),
        "model": config["model"],
        "output_dir": str(output),
        "train_sha256": sha256(Path(config["data"]["train_file"])),
        "validation_sha256": sha256(Path(config["data"]["validation_file"])),
        "max_steps": args.max_steps,
        "validation_max_samples": config["data"]["validation_max_samples"],
        "micro_batch_size": config["training"]["micro_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
    }
    if not args.dry_run:
        validate_resume(output, args.resume_from_checkpoint, identity)
    tokenizer = load_tokenizer(config)
    datasets, reports = {}, {}
    for split, field in (("train", "train_file"), ("validation", "validation_file")):
        examples, reports[split] = load_tokenized(
            Path(config["data"][field]), tokenizer, config["data"]["max_length"]
        )
        if not examples:
            raise ValueError(f"{split}: no examples survived tokenization: {reports[split]}")
        ids = reports[split].pop("kept_ids")
        if split == "validation":
            count = min(len(examples), config["data"]["validation_max_samples"])
            indices = sorted(
                random.Random(config["training"]["seed"]).sample(range(len(examples)), count)
            )
            examples = [examples[index] for index in indices]
            selected_ids = [ids[index] for index in indices]
            reports[split].update(
                {
                    "eval_selected": count,
                    "eval_selected_ids": selected_ids,
                    "eval_selected_ids_sha256": hashlib.sha256(
                        json.dumps(selected_ids).encode()
                    ).hexdigest(),
                }
            )
        datasets[split] = TokenizedDataset(examples)
    report = {
        "dry_run": args.dry_run,
        "identity": identity,
        "tokenization": reports,
        "packages": package_versions(),
        "max_steps": args.max_steps,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
    }
    if args.dry_run:
        if args.report:
            if args.report.exists():
                raise ValueError(f"report exists: {args.report}")
            write_json(args.report, report)
        return report

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 training requires a CUDA GPU with native BF16 support")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("this recipe uses one GPU; set CUDA_VISIBLE_DEVICES to one device")
    settings = config["training"]
    set_seed(settings["seed"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = add_lora(load_text_model(config), config)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=settings["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=settings["micro_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        tf32=False,
        seed=settings["seed"],
        data_seed=settings["seed"],
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=settings["logging_steps"],
        save_steps=settings["save_steps"],
        eval_strategy="steps",
        eval_steps=settings["eval_steps"],
        save_strategy="steps",
        save_total_limit=settings["save_total_limit"],
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        prediction_loss_only=True,
        logging_nan_inf_filter=False,
    )
    output.mkdir(parents=True, exist_ok=args.resume_from_checkpoint is not None)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    report.update(
        {
            "config": config,
            "started_at": datetime.now(UTC).isoformat(),
            "git_commit": git.stdout.strip(),
            "gpu": torch.cuda.get_device_name(0),
            "resume_from_checkpoint": str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None,
        }
    )
    if args.resume_from_checkpoint is None:
        write_json(output / "run.json", report)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(output / f"resume-{stamp}.json", report)
    write_json(output / "tokenization.json", reports)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=AnswerOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    if not math.isfinite(result.training_loss):
        raise ValueError("training produced non-finite loss; final adapter was not saved")
    adapter_dir = output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    metrics = {"train": result.metrics, "validation": trainer.evaluate()}
    if not math.isfinite(metrics["validation"]["eval_loss"]):
        raise ValueError("validation produced non-finite loss; run is not complete")
    write_json(output / "metrics.json", metrics)
    write_json(
        adapter_dir / "jet_training.json",
        {
            "identity": identity,
            "config": config,
            "global_step": trainer.state.global_step,
            "packages": package_versions(),
        },
    )
    report.update(
        {
            "adapter_dir": str(adapter_dir),
            "metrics": metrics,
            "global_step": trainer.state.global_step,
            "completed": True,
            "runtime_seconds_including_model_load": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    write_json(output / "completed.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenize data using real pinned tokenizer; do not load weights",
    )
    parser.add_argument("--report", type=Path, help="New JSON path for dry-run report")
    parser.add_argument(
        "--max-steps", type=int, help="Stop after this total number of optimizer steps"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Use a new run directory, e.g. for a short smoke test"
    )
    parser.add_argument(
        "--validation-max-samples",
        type=int,
        help="Override validation subset size for a short smoke test",
    )
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.validation_max_samples is not None and args.validation_max_samples <= 0:
        parser.error("--validation-max-samples must be positive")
    if args.micro_batch_size is not None and args.micro_batch_size <= 0:
        parser.error("--micro-batch-size must be positive")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.dry_run and args.resume_from_checkpoint:
        parser.error("--dry-run and --resume-from-checkpoint cannot be combined")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, ImportError) as error:
        print(f"jet-train: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
