# %% [markdown]
# # PTM-CIPHER Prototype
#
# This notebook trains a small PTM-CIPHER prototype on the v3 Shield manifest.
#
# It is **not** the full NCS-scale model because it uses `no_structure` 3Di fallback
# and a compact encoder. It does exercise the novelty-bearing path:
#
# - proteoform-state residue x structure x PTM embeddings
# - weight-shared unmodified/modified dual encoder
# - interface-conditioned cross-attention
# - PTM-site perturbation propagation to interface residues
# - representation-level modified-minus-unmodified delta head
# - gradient-reversal shortcut adversaries
#
# Upload `ptmppi_shield_colab_inputs.zip` when prompted.

# %%
!nvidia-smi
!pip -q install pandas numpy scikit-learn tqdm

# %%
from google.colab import files
import json, os, random, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED = 4242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/content/ptm_cipher_outputs")
OUT.mkdir(exist_ok=True)

# %%
uploaded = files.upload()
zip_name = next(name for name in uploaded if name.endswith(".zip"))
with zipfile.ZipFile(zip_name) as zf:
    zf.extractall("/content/ptmppi_inputs")
INPUT = Path("/content/ptmppi_inputs")
print(sorted(p.name for p in INPUT.iterdir()))

# %%
import sys
sys.path.insert(0, str(INPUT))
from ptm_cipher_model import PTMCipher, PTMCipherConfig, RESIDUES, PTM_STATES, NO_STRUCTURE_ID, ptm_cipher_loss

manifest = pd.read_csv(INPUT / "ptm_cipher_input_manifest.tsv", sep="\t").fillna("")
manifest["label_binary"] = manifest["label_binary"].astype(int)
print(manifest.shape)
print(manifest["S2b_cold_interface_split"].value_counts())

# %%
residue_to_id = {aa: i for i, aa in enumerate(RESIDUES)}
ptm_to_id = {name: i for i, name in enumerate(PTM_STATES)}
MAX_MOD = int(manifest["mod_seq_crop"].map(len).max())
MAX_PARTNER = int(manifest["partner_seq_crop"].map(len).max())
MAX_MOD, MAX_PARTNER

for col in ["assay_family", "topology_pair_community"]:
    enc = LabelEncoder()
    manifest[col + "_id"] = enc.fit_transform(manifest[col].astype(str))
    print(col, len(enc.classes_))

class CipherDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def encode_seq(self, seq, max_len):
        ids = np.full(max_len, residue_to_id["X"], dtype=np.int64)
        mask = np.zeros(max_len, dtype=np.bool_)
        for i, aa in enumerate(str(seq)[:max_len]):
            ids[i] = residue_to_id.get(aa, residue_to_id["X"])
            mask[i] = True
        return ids, mask

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        mod_ids, mod_mask = self.encode_seq(row.mod_seq_crop, MAX_MOD)
        partner_ids, partner_mask = self.encode_seq(row.partner_seq_crop, MAX_PARTNER)
        mod_structure = np.full(MAX_MOD, NO_STRUCTURE_ID, dtype=np.int64)
        partner_structure = np.full(MAX_PARTNER, NO_STRUCTURE_ID, dtype=np.int64)
        unmod_ptm = np.zeros(MAX_MOD, dtype=np.int64)
        mod_ptm = np.zeros(MAX_MOD, dtype=np.int64)
        ptm_index = int(row.ptm_index_crop_0based)
        if 0 <= ptm_index < MAX_MOD:
            mod_ptm[ptm_index] = ptm_to_id.get(str(row.ptm_state), ptm_to_id["other"])
        contact = np.zeros((MAX_MOD, MAX_PARTNER), dtype=np.bool_)
        adjacency = np.eye(MAX_MOD, dtype=np.bool_)
        for pair in str(row.contact_pairs_crop).split(";"):
            if not pair or ":" not in pair:
                continue
            i, j = [int(x) for x in pair.split(":")]
            if 0 <= i < MAX_MOD and 0 <= j < MAX_PARTNER:
                contact[i, j] = True
                if 0 <= ptm_index < MAX_MOD:
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
            "ptm_index": np.int64(max(0, min(ptm_index, MAX_MOD - 1))),
            "contact_mask": contact,
            "residue_adjacency": adjacency,
            "label": np.int64(row.label_binary),
            "assay_family": np.int64(row.assay_family_id),
            "topology_pair_community": np.int64(row.topology_pair_community_id),
        }

def collate(batch):
    out = {}
    for key in batch[0]:
        arr = np.stack([item[key] for item in batch])
        out[key] = torch.as_tensor(arr)
    return out

# %%
train_df = manifest[manifest["S2b_cold_interface_split"] == "train"]
valid_df = manifest[manifest["S2b_cold_interface_split"] == "valid"]
test_df = manifest[manifest["S2b_cold_interface_split"] == "test"]
train_loader = DataLoader(CipherDataset(train_df), batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)
valid_loader = DataLoader(CipherDataset(valid_df), batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)
test_loader = DataLoader(CipherDataset(test_df), batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)

config = PTMCipherConfig(
    dim=128,
    heads=4,
    layers=2,
    ff_dim=384,
    dropout=0.20,
    graph_layers=2,
    classes=2,
    adversary_dims={
        "assay_family": int(manifest["assay_family_id"].nunique()),
        "topology_pair_community": int(manifest["topology_pair_community_id"].nunique()),
    },
)
model = PTMCipher(config).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sum(p.numel() for p in model.parameters()) / 1e6

# %%
def move(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}

def run_epoch(loader, train=True, epoch=0):
    model.train(train)
    losses = []
    ys, ps = [], []
    for batch in tqdm(loader, leave=False):
        batch = move(batch)
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
                adversary_alpha=min(1.0, epoch / 3),
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
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(batch["label"].detach().cpu().numpy().tolist())
        ps.extend(out["logits"].softmax(dim=-1)[:, 1].detach().cpu().numpy().tolist())
    return np.array(ys), np.array(ps), float(np.mean(losses))

def metrics(name, y, p):
    threshold = np.quantile(p, 1 - y.mean()) if len(np.unique(y)) > 1 else 0.5
    pred = (p >= threshold).astype(int)
    return {
        "split_col": "S2b_cold_interface_split",
        "model": "ptm_cipher_lite_no3di",
        "split": name,
        "n": len(y),
        "auprc": average_precision_score(y, p),
        "auroc": roc_auc_score(y, p),
        "mcc": matthews_corrcoef(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }

# %%
best_valid = -1
best_state = None
history = []
for epoch in range(1, 8):
    y_tr, p_tr, loss_tr = run_epoch(train_loader, train=True, epoch=epoch)
    y_va, p_va, loss_va = run_epoch(valid_loader, train=False, epoch=epoch)
    row = metrics("valid", y_va, p_va)
    row["epoch"] = epoch
    row["train_loss"] = loss_tr
    row["valid_loss"] = loss_va
    history.append(row)
    print(row)
    if row["auprc"] > best_valid:
        best_valid = row["auprc"]
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
y_te, p_te, loss_te = run_epoch(test_loader, train=False)
test_metrics = metrics("test", y_te, p_te)
test_metrics["test_loss"] = loss_te
print(test_metrics)

pd.DataFrame(history + [test_metrics]).to_csv(OUT / "ptm_cipher_lite_metrics_colab.tsv", sep="\t", index=False)
torch.save(best_state, OUT / "ptm_cipher_lite_best_state.pt")

# %%
with zipfile.ZipFile("/content/ptm_cipher_colab_outputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUT.glob("*"):
        zf.write(path, path.name)
files.download("/content/ptm_cipher_colab_outputs.zip")
