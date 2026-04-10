"""
Stage 3 experiment runner for Assignment 1.

Run this from the project root in Colab after data preprocessing:

    from stage3_experiments import run_stage3_experiments
    summary = run_stage3_experiments()

The default plan is intentionally small enough to run on Colab Pro:
one shared baseline plus one controlled variant for each hypothesis.
Increase num_steps in COMMON_CONFIG if you have more GPU budget.
"""

from __future__ import annotations

import csv
import json
import os
import traceback
from copy import deepcopy
from typing import Any

from EvaluateTools.evaluate import evaluate
from TrainTools.train import train


COMMON_CONFIG: dict[str, Any] = {
    # Data paths produced by the notebook preprocessing cell.
    "train_npz": "_data/train.npz",
    "dev_npz": "_data/dev.npz",
    "word_emb_json": "_data/word_emb.json",
    "char_emb_json": "_data/char_emb.json",
    "train_eval_json": "_data/train_eval.json",
    "dev_eval_json": "_data/dev_eval.json",
    # Use shorter controlled runs for Stage 3 comparisons.
    "num_steps": 3000,
    "checkpoint": 500,
    "val_num_batches": 25,
    "test_num_batches": 25,
    "batch_size": 8,
    "seed": 42,
    # Baseline mechanism choices.
    "optimizer_name": "adam",
    "scheduler_name": "lambda",
    "loss_name": "qa_nll",
    "norm_name": "layer_norm",
    "norm_groups": 8,
    "dropout": 0.1,
    "dropout_char": 0.05,
    "activation": "relu",
    "init_name": "kaiming",
}


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "baseline_adam_lambda_layernorm_dropout010",
        "hypothesis": "baseline",
        "overrides": {},
    },
    {
        "name": "optimizer_sgdmomentum_step",
        "hypothesis": "optimizer_scheduler",
        "overrides": {
            "optimizer_name": "sgd_momentum",
            "scheduler_name": "step",
            "lr_step_size": 1000,
            "lr_gamma": 0.5,
        },
    },
    {
        "name": "normalization_groupnorm",
        "hypothesis": "normalization",
        "overrides": {
            "norm_name": "group_norm",
            "norm_groups": 8,
        },
    },
    {
        "name": "dropout_zero",
        "hypothesis": "regularization",
        "overrides": {
            "dropout": 0.0,
            "dropout_char": 0.0,
        },
    },
    {
        "name": "dropout_high",
        "hypothesis": "regularization",
        "overrides": {
            "dropout": 0.2,
            "dropout_char": 0.1,
        },
    },
]


def _summarize_result(name: str, hypothesis: str, status: str, result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("history") or []
    last = history[-1] if history else {}
    return {
        "name": name,
        "hypothesis": hypothesis,
        "status": status,
        "best_f1": result.get("best_f1"),
        "best_em": result.get("best_em"),
        "final_step": last.get("step"),
        "final_train_loss": last.get("train_loss"),
        "final_train_f1": last.get("train_f1"),
        "final_train_em": last.get("train_em"),
        "final_dev_loss": last.get("dev_loss"),
        "final_dev_f1": last.get("dev_f1"),
        "final_dev_em": last.get("dev_em"),
        "final_lr": last.get("lr"),
    }


def _write_summary_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_checkpoint(cfg: dict[str, Any], final_eval_batches: int) -> dict[str, Any]:
    return evaluate(
        dev_npz=cfg["dev_npz"],
        word_emb_json=cfg["word_emb_json"],
        char_emb_json=cfg["char_emb_json"],
        dev_eval_json=cfg["dev_eval_json"],
        save_dir=cfg["save_dir"],
        log_dir=cfg["log_dir"],
        ckpt_name=cfg["ckpt_name"],
        batch_size=cfg["batch_size"],
        test_num_batches=final_eval_batches,
        loss_name=cfg["loss_name"],
        para_limit=cfg.get("para_limit", 400),
        ques_limit=cfg.get("ques_limit", 50),
        char_limit=cfg.get("char_limit", 16),
        d_model=cfg.get("d_model", 96),
        num_heads=cfg.get("num_heads", 8),
        glove_dim=cfg.get("glove_dim", 300),
        char_dim=cfg.get("char_dim", 64),
        dropout=cfg.get("dropout", 0.1),
        dropout_char=cfg.get("dropout_char", 0.05),
        pretrained_char=cfg.get("pretrained_char", False),
        norm_name=cfg.get("norm_name", "layer_norm"),
        norm_groups=cfg.get("norm_groups", 8),
        activation=cfg.get("activation", "relu"),
        init_name=cfg.get("init_name", "kaiming"),
    )


def run_stage3_experiments(
    experiments: list[dict[str, Any]] | None = None,
    common_overrides: dict[str, Any] | None = None,
    output_dir: str = "_stage3_results",
    final_eval_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Run controlled Stage 3 experiments and save machine-readable results.

    Args:
        experiments: Optional subset/replacement for EXPERIMENTS.
        common_overrides: Optional shared config overrides, e.g.
            {"batch_size": 16, "num_steps": 5000}.
        output_dir: Directory for per-run logs, checkpoints, and summaries.
        final_eval_batches: If set, run evaluate() after each training run.
            Use 25 for a quick extra evaluation or -1 for full dev evaluation.

    Returns:
        A list of summary rows, also written to output_dir/summary.csv.
    """
    selected = experiments or EXPERIMENTS
    base_config = deepcopy(COMMON_CONFIG)
    if common_overrides:
        base_config.update(common_overrides)

    os.makedirs(output_dir, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for spec in selected:
        name = spec["name"]
        hypothesis = spec.get("hypothesis", "")
        cfg = deepcopy(base_config)
        cfg.update(spec.get("overrides", {}))
        cfg["save_dir"] = os.path.join(output_dir, "models", name)
        cfg["log_dir"] = os.path.join(output_dir, "logs", name)
        cfg["ckpt_name"] = "model.pt"

        print("=" * 80)
        print(f"Running Stage 3 experiment: {name}")
        print(json.dumps({"hypothesis": hypothesis, "config": cfg}, indent=2))
        print("=" * 80)

        result_path = os.path.join(output_dir, f"{name}.json")
        try:
            result = train(**cfg)
            if final_eval_batches is not None:
                result["final_eval"] = _evaluate_checkpoint(cfg, final_eval_batches)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            row = _summarize_result(name, hypothesis, "ok", result)
            if "final_eval" in result:
                row["full_eval_f1"] = result["final_eval"]["f1"]
                row["full_eval_em"] = result["final_eval"]["exact_match"]
                row["full_eval_loss"] = result["final_eval"]["loss"]
        except Exception as exc:
            row = {
                "name": name,
                "hypothesis": hypothesis,
                "status": "error",
                "error": repr(exc),
            }
            with open(result_path, "w") as f:
                json.dump(
                    {
                        "name": name,
                        "hypothesis": hypothesis,
                        "config": cfg,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                    f,
                    indent=2,
                )
            print(f"Experiment failed: {name}")
            traceback.print_exc()
        rows.append(row)
        _write_summary_csv(rows, os.path.join(output_dir, "summary.csv"))

    print("Stage 3 summary:")
    for row in rows:
        print(row)
    return rows


if __name__ == "__main__":
    run_stage3_experiments()
