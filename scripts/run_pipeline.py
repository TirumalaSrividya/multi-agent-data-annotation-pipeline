#!/usr/bin/env python3
"""
Entry point for the annotation and training pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT, load_settings_with_overrides
from src.data.dataset import load_unlabelled_pool
from src.logging_setup import get_logger, setup_logging
from src.orchestrator import PipelineOrchestrator


def parse_overrides(raw_overrides: list[str]) -> dict:
    overrides = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}', expected key=value")
        key, value = item.split("=", 1)
        # best-effort type coercion
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent annotation + training pipeline.")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config/config.yaml)")
    parser.add_argument("--provider", default=None, choices=["anthropic", "mock"], help="Override llm.provider")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config, e.g. --set trainer.target_accuracy=0.9")
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    if args.provider:
        overrides["llm.provider"] = args.provider

    settings = load_settings_with_overrides(args.config, overrides)
    setup_logging(settings.logging.level, settings.logging.log_dir, settings.logging.log_file)
    logger = get_logger("run_pipeline")

    logger.info("Loaded settings for task '%s' with labels %s", settings.task.name, settings.task.label_set)
    logger.info("LLM provider: %s (model=%s)", settings.llm.provider, settings.llm.model)

    dataset_path = settings.resolve(settings.data.dataset_path)
    unlabelled_pool = load_unlabelled_pool(
        dataset_path=dataset_path,
        text_column=settings.task.text_column,
        id_column=settings.task.id_column,
        label_column=settings.task.label_column,
        sample_size=settings.data.unlabelled_sample_size,
        random_seed=settings.data.random_seed,
    )

    orchestrator = PipelineOrchestrator(settings)

    logger.info("=== PIPELINE 1: ANNOTATION (active learning) ===")
    state = orchestrator.run_annotation_pipeline(unlabelled_pool)

    output_dir = settings.resolve(settings.artifacts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labelled_rows = []
    for sample_id, annotation in state.labelled.items():
        sample = orchestrator.samples_by_id.get(sample_id)
        labelled_rows.append({
            "id": sample_id,
            "text": sample.text if sample else "",
            "true_label": sample.true_label if sample else None,
            "predicted_label": annotation.label,
            "confidence": annotation.confidence,
            "source": annotation.source,
            "review_count": annotation.review_count,
        })
    labelled_df = pd.DataFrame(labelled_rows)
    labelled_csv_path = output_dir / settings.artifacts.labelled_pool_file
    labelled_df.to_csv(labelled_csv_path, index=False)
    logger.info("Wrote %d labelled samples to %s", len(labelled_df), labelled_csv_path)

    if settings.task.label_column and not labelled_df.empty and labelled_df["true_label"].notna().any():
        agreement = (labelled_df["predicted_label"] == labelled_df["true_label"]).mean()
        logger.info("Agent-vs-ground-truth agreement on labelled pool: %.3f (demo diagnostic only)", agreement)

    if len(set(labelled_df["predicted_label"])) < 2:
        logger.error("Fewer than 2 distinct labels were accepted; cannot proceed to training. Exiting.")
        return

    logger.info("=== PIPELINE 2: AUTOMATED TRAINING ===")
    report, best_model = orchestrator.run_training_pipeline()

    report_dict = {
        "candidate_results_eval": [asdict(r) for r in report.candidate_results_eval],
        "best_model_name": report.best_model_name,
        "test_result": asdict(report.test_result),
        "target_accuracy": report.target_accuracy,
        "target_reached": report.target_reached,
        "decision": report.decision,
        "train_size": report.train_size,
        "eval_size": report.eval_size,
        "test_size": report.test_size,
    }
    report_path = output_dir / settings.artifacts.training_report_file
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, default=str)
    logger.info("Wrote training report to %s", report_path)

    print("\n" + "=" * 70)
    print(f"Best model: {report.best_model_name}")
    print(f"Test accuracy: {report.test_result.accuracy:.3f} (target: {report.target_accuracy:.2f}, "
          f"reached: {report.target_reached})")
    print(f"Test macro-F1: {report.test_result.macro_f1:.3f}")
    print("Per-class test metrics:")
    for label, m in report.test_result.per_class.items():
        print(f"  {label:12s} precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} support={m.support}")
    print("=" * 70)


if __name__ == "__main__":
    main()
