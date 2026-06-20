from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from feature_cache import SEED, available_split_strategies, build_feature_dataset


@dataclass
class ModelRun:
    name: str
    estimator: object
    train_matrix: object
    all_matrix: object
    artifact: dict
    feasible: bool = True
    skip_reason: str = ""


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No metrics available._\n"
    columns = list(df.columns)
    widths = {
        col: max(len(str(col)), *(len(str(value)) for value in df[col].astype(str).tolist()))
        for col in columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in columns) + " |"
    divider = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in columns) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def aligned_proba(estimator: object, matrix: object, all_classes: np.ndarray) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(matrix)
    estimator_classes = getattr(estimator, "classes_", all_classes)
    aligned = np.zeros((proba.shape[0], len(all_classes)), dtype=np.float64)
    class_to_idx = {label: idx for idx, label in enumerate(all_classes)}
    for src_idx, label in enumerate(estimator_classes):
        dst_idx = class_to_idx.get(label)
        if dst_idx is not None:
            aligned[:, dst_idx] = proba[:, src_idx]
    return aligned


def score_split(
    benchmark_split: str,
    model_name: str,
    split_name: str,
    estimator: object,
    matrix: object,
    y_true: np.ndarray,
    labels: list[str],
    encoded_classes: np.ndarray,
) -> dict:
    y_pred = estimator.predict(matrix)
    row = {
        "benchmark_split": benchmark_split,
        "model": model_name,
        "split": split_name,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "log_loss": np.nan,
        "auroc_macro_ovr": np.nan,
    }
    proba = aligned_proba(estimator, matrix, encoded_classes)
    if proba is not None and len(np.unique(y_true)) > 1:
        try:
            row["log_loss"] = log_loss(y_true, proba, labels=encoded_classes)
        except Exception:
            row["log_loss"] = np.nan
        try:
            if len(encoded_classes) == 2:
                row["auroc_macro_ovr"] = roc_auc_score(y_true, proba[:, 1])
            else:
                row["auroc_macro_ovr"] = roc_auc_score(
                    y_true,
                    proba,
                    labels=encoded_classes,
                    multi_class="ovr",
                    average="macro",
                )
        except Exception:
            row["auroc_macro_ovr"] = np.nan
    row["labels"] = ",".join(labels)
    return row


def prepare_model_runs(
    embeddings: np.ndarray,
    mechanism: pd.DataFrame,
    train_mask: np.ndarray,
    seed: int,
    n_classes: int,
) -> tuple[list[ModelRun], dict]:
    categorical_cols = [col for col in mechanism.columns if mechanism[col].dtype == object or mechanism[col].dtype == bool]
    numeric_cols = [col for col in mechanism.columns if col not in categorical_cols]

    mechanism_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ],
        sparse_threshold=0.3,
    )
    X_mech_all = mechanism_preprocessor.fit_transform(mechanism.iloc[train_mask])
    X_mech_full = mechanism_preprocessor.transform(mechanism)
    X_mech_tree_train = X_mech_all.toarray() if sparse.issparse(X_mech_all) else X_mech_all
    X_mech_tree_full = X_mech_full.toarray() if sparse.issparse(X_mech_full) else X_mech_full

    seq_scaler = StandardScaler()
    X_seq_train = seq_scaler.fit_transform(embeddings[train_mask])
    X_seq_full = seq_scaler.transform(embeddings)

    X_combined_train = sparse.hstack([sparse.csr_matrix(X_seq_train), sparse.csr_matrix(X_mech_all)], format="csr")
    X_combined_full = sparse.hstack([sparse.csr_matrix(X_seq_full), sparse.csr_matrix(X_mech_full)], format="csr")

    runs = [
        ModelRun(
            name="majority",
            estimator=DummyClassifier(strategy="most_frequent", random_state=seed),
            train_matrix=np.zeros((int(train_mask.sum()), 1), dtype=np.float32),
            all_matrix=np.zeros((len(train_mask), 1), dtype=np.float32),
            artifact={"feature_set": "none"},
        ),
        ModelRun(
            name="logistic_regression_mechanism",
            estimator=LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            ),
            train_matrix=X_mech_all,
            all_matrix=X_mech_full,
            artifact={"feature_set": "mechanism", "preprocessor": mechanism_preprocessor},
        ),
        ModelRun(
            name="tree_baseline_mechanism",
            estimator=RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            train_matrix=X_mech_tree_train,
            all_matrix=X_mech_tree_full,
            artifact={"feature_set": "mechanism", "preprocessor": mechanism_preprocessor},
        ),
        ModelRun(
            name="sequence_embedding_logistic",
            estimator=LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            ),
            train_matrix=X_seq_train,
            all_matrix=X_seq_full,
            artifact={"feature_set": "sequence_embeddings", "scaler": seq_scaler},
        ),
        ModelRun(
            name="combined_mechanism_aware",
            estimator=LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            ),
            train_matrix=X_combined_train,
            all_matrix=X_combined_full,
            artifact={
                "feature_set": "sequence_embeddings+mechanism",
                "sequence_scaler": seq_scaler,
                "mechanism_preprocessor": mechanism_preprocessor,
            },
        ),
    ]

    if int(train_mask.sum()) >= 80 and n_classes >= 2 and X_seq_train.shape[1] <= 5000:
        runs.append(
            ModelRun(
                name="mlp_sequence_embedding",
                estimator=MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=min(256, int(train_mask.sum())),
                    early_stopping=True,
                    learning_rate_init=1e-3,
                    max_iter=250,
                    random_state=seed,
                    validation_fraction=0.15,
                ),
                train_matrix=X_seq_train,
                all_matrix=X_seq_full,
                artifact={"feature_set": "sequence_embeddings", "scaler": seq_scaler},
            )
        )
    else:
        runs.append(
            ModelRun(
                name="mlp_sequence_embedding",
                estimator=DummyClassifier(strategy="most_frequent", random_state=seed),
                train_matrix=np.zeros((int(train_mask.sum()), 1), dtype=np.float32),
                all_matrix=np.zeros((len(train_mask), 1), dtype=np.float32),
                artifact={"feature_set": "sequence_embeddings"},
                feasible=False,
                skip_reason="MLP skipped: need at least 80 train rows, 2 classes, and <=5000 embedding dimensions.",
            )
        )

    feature_meta = {
        "mechanism_categorical_columns": categorical_cols,
        "mechanism_numeric_columns": numeric_cols,
        "sequence_embedding_dim": int(embeddings.shape[1]),
        "mechanism_dim": int(X_mech_full.shape[1]),
        "combined_dim": int(X_combined_full.shape[1]),
    }
    return runs, feature_meta


def train_and_evaluate_strategy(args: argparse.Namespace, split_strategy: str) -> tuple[pd.DataFrame, list[str], dict]:
    set_seeds(args.seed)
    start = time.time()
    notes = [f"Training seed: {args.seed}.", f"Benchmark split strategy: {split_strategy}."]

    dataset = build_feature_dataset(
        project_dir=args.project_dir,
        split_strategy=split_strategy,
        backend=args.backend,
        allow_cpu_esm=args.allow_cpu_esm,
        allow_sequence_downloads=not args.no_sequence_downloads,
        force=args.force_features,
        seed=args.seed,
    )
    notes.extend(dataset.notes)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset.labels)
    labels = label_encoder.classes_.tolist()
    train_mask = dataset.splits == "train"
    eval_splits = ["train"] + sorted(split for split in set(dataset.splits) if split != "train")

    if len(np.unique(y[train_mask])) < 2:
        raise RuntimeError("Training split has fewer than two classes; cannot train classifiers.")

    runs, feature_meta = prepare_model_runs(
        embeddings=dataset.embeddings,
        mechanism=dataset.mechanism,
        train_mask=train_mask,
        seed=args.seed,
        n_classes=len(labels),
    )

    models_dir = args.project_dir / "models" / split_strategy
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    for run in runs:
        if not run.feasible:
            notes.append(run.skip_reason)
            continue
        fit_start = time.time()
        run.estimator.fit(run.train_matrix, y[train_mask])
        fit_seconds = time.time() - fit_start
        notes.append(f"Fit {run.name} in {fit_seconds:.2f}s.")

        for split_name in eval_splits:
            mask = dataset.splits == split_name
            if not mask.any():
                continue
            split_matrix = run.all_matrix[mask]
            metrics_rows.append(
                score_split(
                    benchmark_split=split_strategy,
                    model_name=run.name,
                    split_name=split_name,
                    estimator=run.estimator,
                    matrix=split_matrix,
                    y_true=y[mask],
                    labels=labels,
                    encoded_classes=np.arange(len(labels)),
                )
            )

        artifact = {
            "model": run.estimator,
            "labels": labels,
            "backend": dataset.backend,
            "seed": args.seed,
            **run.artifact,
        }
        joblib.dump(artifact, models_dir / f"{run.name}.joblib")

    metrics = pd.DataFrame(metrics_rows)
    if not metrics.empty:
        numeric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "auroc_macro_ovr"]
        metrics[numeric_cols] = metrics[numeric_cols].round(4)

    elapsed = time.time() - start
    split_counts = pd.Series(dataset.splits).value_counts().to_dict()
    label_counts = pd.Series(dataset.labels).value_counts().to_dict()
    run_meta = {
        "backend": dataset.backend,
        "benchmark_split": split_strategy,
        "rows": int(len(dataset.rows)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "elapsed_seconds": round(elapsed, 2),
        "feature_meta": feature_meta,
        "feature_files": dataset.metadata,
    }
    notes.append(f"Training/evaluation elapsed: {elapsed:.2f}s.")
    return metrics, notes, run_meta


def train_and_evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict]:
    if args.split_strategy == "all":
        strategies = available_split_strategies(args.project_dir)
        if not strategies:
            strategies = ["random"]
    else:
        strategies = [args.split_strategy]

    all_metrics: list[pd.DataFrame] = []
    all_notes: list[str] = [f"Requested split strategy: {args.split_strategy}."]
    strategy_meta: dict[str, dict] = {}

    for strategy in strategies:
        metrics, notes, meta = train_and_evaluate_strategy(args, strategy)
        all_metrics.append(metrics)
        all_notes.extend(notes)
        strategy_meta[strategy] = meta

    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    if len(strategy_meta) == 1:
        run_meta = next(iter(strategy_meta.values()))
    else:
        elapsed = sum(meta.get("elapsed_seconds", 0) for meta in strategy_meta.values())
        run_meta = {
            "backend": {name: meta.get("backend") for name, meta in strategy_meta.items()},
            "benchmark_split": "all",
            "rows": {name: meta.get("rows") for name, meta in strategy_meta.items()},
            "split_counts": {name: meta.get("split_counts") for name, meta in strategy_meta.items()},
            "label_counts": {name: meta.get("label_counts") for name, meta in strategy_meta.items()},
            "elapsed_seconds": round(elapsed, 2),
            "strategies": strategy_meta,
        }
    return combined_metrics, all_notes, run_meta


def write_outputs(project_dir: Path, metrics: pd.DataFrame, notes: list[str], run_meta: dict) -> None:
    tables_dir = project_dir / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = tables_dir / "model_metrics.csv"
    metrics_md = tables_dir / "model_metrics.md"
    notes_md = tables_dir / "runtime_notes.md"
    metadata_json = tables_dir / "training_run_metadata.json"

    metrics.to_csv(metrics_csv, index=False)
    metrics_md.write_text(markdown_table(metrics), encoding="utf-8")
    metadata_json.write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Runtime Notes",
        "",
        f"- Backend: {run_meta.get('backend', 'unknown')}",
        f"- Benchmark split: {run_meta.get('benchmark_split', 'unknown')}",
        f"- Rows: {run_meta.get('rows', 'unknown')}",
        f"- Split counts: `{json.dumps(run_meta.get('split_counts', {}), sort_keys=True)}`",
        f"- Label counts: `{json.dumps(run_meta.get('label_counts', {}), sort_keys=True)}`",
        f"- Elapsed seconds: {run_meta.get('elapsed_seconds', 'unknown')}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    notes_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SwitchPPI feature/model baselines from auditor splits.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split-strategy", default="all")
    parser.add_argument("--backend", choices=["auto", "esm2", "deterministic"], default="auto")
    parser.add_argument("--allow-cpu-esm", action="store_true")
    parser.add_argument("--no-sequence-downloads", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    try:
        metrics, notes, run_meta = train_and_evaluate(args)
    except Exception as exc:
        tables_dir = args.project_dir / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {"backend": args.backend, "rows": 0, "split_counts": {}, "label_counts": {}, "elapsed_seconds": 0}
        notes = [f"Training did not run: {exc}"]
        write_outputs(args.project_dir, pd.DataFrame(), notes, run_meta)
        raise

    write_outputs(args.project_dir, metrics, notes, run_meta)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
