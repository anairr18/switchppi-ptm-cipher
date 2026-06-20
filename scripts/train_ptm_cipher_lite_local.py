from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from ptm_cipher_model import NO_STRUCTURE_ID, PTMCipher, PTMCipherConfig, PTM_STATES, RESIDUES, ptm_cipher_loss


ROOT = Path(__file__).resolve().parents[1]
TABLES_V3 = ROOT / "results_v3" / "tables"
MODELS_V3 = ROOT / "models_v3"
SEED = 4242


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_pairs(value: object) -> list[tuple[int, int]]:
    pairs = []
    for item in str(value or "").split(";"):
        if not item or ":" not in item:
            continue
        left, right = item.split(":", 1)
        try:
            pairs.append((int(left), int(right)))
        except ValueError:
            continue
    return pairs


def crop_mod(row: pd.Series, max_len: int) -> tuple[str, int, int]:
    seq = str(row["mod_seq_crop"])
    ptm_index = int(row["ptm_index_crop_0based"])
    if len(seq) <= max_len:
        return seq, 0, ptm_index
    start = max(0, min(ptm_index - max_len // 2, len(seq) - max_len))
    return seq[start : start + max_len], start, ptm_index - start


def crop_partner(row: pd.Series, max_len: int) -> tuple[str, int]:
    seq = str(row["partner_seq_crop"])
    pairs = parse_pairs(row["contact_pairs_crop"])
    if len(seq) <= max_len:
        return seq, 0
    if pairs:
        partner_positions = sorted(j for _, j in pairs)
        center = partner_positions[len(partner_positions) // 2]
    else:
        center = len(seq) // 2
    start = max(0, min(center - max_len // 2, len(seq) - max_len))
    return seq[start : start + max_len], start


class CipherLiteDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_mod_len: int, max_partner_len: int):
        self.frame = frame.reset_index(drop=True)
        self.max_mod_len = max_mod_len
        self.max_partner_len = max_partner_len
        self.residue_to_id = {aa: i for i, aa in enumerate(RESIDUES)}
        self.ptm_to_id = {name: i for i, name in enumerate(PTM_STATES)}

    def __len__(self) -> int:
        return len(self.frame)

    def encode_sequence(self, seq: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        ids = np.full(max_len, self.residue_to_id["X"], dtype=np.int64)
        mask = np.zeros(max_len, dtype=np.bool_)
        for i, aa in enumerate(str(seq)[:max_len]):
            ids[i] = self.residue_to_id.get(aa, self.residue_to_id["X"])
            mask[i] = True
        return ids, mask

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | np.int64]:
        row = self.frame.iloc[idx]
        mod_seq, mod_start, ptm_index = crop_mod(row, self.max_mod_len)
        partner_seq, partner_start = crop_partner(row, self.max_partner_len)
        mod_ids, mod_mask = self.encode_sequence(mod_seq, self.max_mod_len)
        partner_ids, partner_mask = self.encode_sequence(partner_seq, self.max_partner_len)
        mod_structure = np.full(self.max_mod_len, NO_STRUCTURE_ID, dtype=np.int64)
        partner_structure = np.full(self.max_partner_len, NO_STRUCTURE_ID, dtype=np.int64)
        unmod_ptm = np.zeros(self.max_mod_len, dtype=np.int64)
        mod_ptm = np.zeros(self.max_mod_len, dtype=np.int64)
        if 0 <= ptm_index < self.max_mod_len:
            mod_ptm[ptm_index] = self.ptm_to_id.get(str(row["ptm_state"]), self.ptm_to_id["other"])
        contact = np.zeros((self.max_mod_len, self.max_partner_len), dtype=np.bool_)
        adjacency = np.eye(self.max_mod_len, dtype=np.bool_)
        for mod_pos, partner_pos in parse_pairs(row["contact_pairs_crop"]):
            i = mod_pos - mod_start
            j = partner_pos - partner_start
            if 0 <= i < self.max_mod_len and 0 <= j < self.max_partner_len:
                contact[i, j] = True
                adjacency[ptm_index, i] = True
                adjacency[i, ptm_index] = True
        return {
            "mod_residue_ids": mod_ids,
            "mod_structure_ids": mod_structure,
            "mod_ptm_state_ids": mod_ptm,
            "unmod_ptm_state_ids": unmod_ptm,
            "mod_mask": mod_mask,
            "partner_residue_ids": partner_ids,
            "partner_structure_ids": partner_structure,
            "partner_mask": partner_mask,
            "ptm_index": np.int64(max(0, min(ptm_index, self.max_mod_len - 1))),
            "contact_mask": contact,
            "residue_adjacency": adjacency,
            "label": np.int64(row["label_binary"]),
            "assay_family": np.int64(row["assay_family_id"]),
            "topology_pair_community": np.int64(row["topology_pair_community_id"]),
        }


def collate(batch: list[dict[str, np.ndarray | np.int64]]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(np.stack([item[key] for item in batch])) for key in batch[0]}


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: PTMCipher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    labels: list[int] = []
    probs: list[float] = []
    for batch in loader:
        batch = move(batch, device)
        with torch.set_grad_enabled(train):
            out = model(
                batch["mod_residue_ids"],
                batch["mod_structure_ids"],
                batch["mod_ptm_state_ids"],
                batch["unmod_ptm_state_ids"],
                batch["mod_mask"],
                batch["partner_residue_ids"],
                batch["partner_structure_ids"],
                batch["partner_mask"],
                batch["ptm_index"],
                batch["contact_mask"],
                batch["residue_adjacency"],
                adversary_alpha=min(1.0, epoch / 3.0),
            )
            loss_dict = ptm_cipher_loss(
                out,
                batch["label"],
                {
                    "assay_family": batch["assay_family"],
                    "topology_pair_community": batch["topology_pair_community"],
                },
                lambda_brier=0.05,
                lambda_adversary=0.05,
            )
            loss = loss_dict["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.extend(batch["label"].detach().cpu().numpy().tolist())
        probs.extend(out["logits"].softmax(dim=-1)[:, 1].detach().cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(probs), float(np.mean(losses))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    thresholds = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    best_t = 0.5
    best_mcc = -2.0
    for t in thresholds:
        mcc = matthews_corrcoef(y, (p >= t).astype(int))
        if mcc > best_mcc:
            best_t = float(t)
            best_mcc = float(mcc)
    return best_t


def score(split: str, y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict[str, object]:
    threshold = best_threshold(y, p) if threshold is None else threshold
    pred = (p >= threshold).astype(int)
    return {
        "split_col": "S2b_cold_interface_split",
        "model": "ptm_cipher_lite_local",
        "split": split,
        "n": int(len(y)),
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def build_comparison(ptm_test: dict[str, object]) -> pd.DataFrame:
    baselines = pd.read_csv(TABLES_V3 / "shield_model_metrics_v3.tsv", sep="\t")
    s2b = baselines[baselines["split_col"].eq("S2b_cold_interface_split")].copy()
    keep = ["split_col", "model", "test_n", "auprc", "auroc", "mcc", "balanced_accuracy", "ece"]
    s2b = s2b[keep]
    ptm_row = {
        "split_col": ptm_test["split_col"],
        "model": ptm_test["model"],
        "test_n": ptm_test["n"],
        "auprc": ptm_test["auprc"],
        "auroc": ptm_test["auroc"],
        "mcc": ptm_test["mcc"],
        "balanced_accuracy": ptm_test["balanced_accuracy"],
        "ece": np.nan,
    }
    out = pd.concat([s2b, pd.DataFrame([ptm_row])], ignore_index=True)
    out["auprc_rank"] = out["auprc"].rank(ascending=False, method="min").astype(int)
    out["mcc_rank"] = out["mcc"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("auprc", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-mod-len", type=int, default=128)
    parser.add_argument("--max-partner-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--graph-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()

    set_seed(SEED)
    TABLES_V3.mkdir(parents=True, exist_ok=True)
    MODELS_V3.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(TABLES_V3 / "ptm_cipher_input_manifest.tsv", sep="\t").fillna("")
    for col in ["assay_family", "topology_pair_community"]:
        encoder = LabelEncoder()
        frame[col + "_id"] = encoder.fit_transform(frame[col].astype(str))
    train_df = frame[frame["S2b_cold_interface_split"].eq("train")]
    valid_df = frame[frame["S2b_cold_interface_split"].eq("valid")]
    test_df = frame[frame["S2b_cold_interface_split"].eq("test")]
    train_loader = DataLoader(
        CipherLiteDataset(train_df, args.max_mod_len, args.max_partner_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        CipherLiteDataset(valid_df, args.max_mod_len, args.max_partner_len),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        CipherLiteDataset(test_df, args.max_mod_len, args.max_partner_len),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
    )
    config = PTMCipherConfig(
        dim=args.dim,
        heads=4,
        layers=args.layers,
        ff_dim=args.dim * 3,
        dropout=0.20,
        graph_layers=args.graph_layers,
        classes=2,
        adversary_dims={
            "assay_family": int(frame["assay_family_id"].nunique()),
            "topology_pair_community": int(frame["topology_pair_community_id"].nunique()),
        },
    )
    model = PTMCipher(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_state = None
    best_valid_auprc = -1.0
    best_valid_threshold_value = 0.5
    for epoch in range(1, args.epochs + 1):
        _, _, train_loss = run_epoch(model, train_loader, optimizer, device, epoch)
        y_valid, p_valid, valid_loss = run_epoch(model, valid_loader, None, device, epoch)
        valid_row = score("valid", y_valid, p_valid)
        valid_row["epoch"] = epoch
        valid_row["train_loss"] = train_loss
        valid_row["valid_loss"] = valid_loss
        history.append(valid_row)
        print(valid_row, flush=True)
        if float(valid_row["auprc"]) > best_valid_auprc:
            best_valid_auprc = float(valid_row["auprc"])
            best_valid_threshold_value = float(valid_row["threshold"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    y_test, p_test, test_loss = run_epoch(model, test_loader, None, device, args.epochs)
    test_row = score("test", y_test, p_test, threshold=best_valid_threshold_value)
    test_row["epoch"] = "best_valid"
    test_row["test_loss"] = test_loss
    metrics = pd.DataFrame(history + [test_row])
    metrics.to_csv(TABLES_V3 / "ptm_cipher_lite_local_metrics.tsv", sep="\t", index=False)
    comparison = build_comparison(test_row)
    comparison.to_csv(TABLES_V3 / "architecture_comparison_s2b_v3.tsv", sep="\t", index=False)
    torch.save(best_state, MODELS_V3 / "ptm_cipher_lite_local_best.pt")
    print("TEST", test_row, flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
