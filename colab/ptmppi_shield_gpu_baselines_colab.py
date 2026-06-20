# %% [markdown]
# # PTM-PPI Shield GPU Baselines
#
# Runtime: Google Colab with GPU enabled (`Runtime -> Change runtime type -> T4/A100/L4 GPU`).
#
# Upload `ptmppi_shield_colab_inputs.zip` from the local `colab/` folder when prompted.
# This notebook trains GPU-feasible ESM2 baselines under the exact Shield split columns,
# including `S2b_cold_interface_split`.
#
# Outputs:
# - `shield_gpu_baseline_metrics_colab.tsv`
# - `shield_gpu_baseline_reproducibility_notes.tsv`
# - `ptmppi_shield_colab_outputs.zip`

# %%
!nvidia-smi
!pip -q install transformers accelerate biopython scikit-learn pandas numpy scipy joblib tqdm

# %%
from google.colab import files
import json, math, os, random, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, EsmModel

SEED = 4242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/content/ptmppi_colab_outputs")
OUT.mkdir(exist_ok=True)

# %%
uploaded = files.upload()
zip_name = next(name for name in uploaded if name.endswith(".zip"))
with zipfile.ZipFile(zip_name) as zf:
    zf.extractall("/content/ptmppi_inputs")
INPUT = Path("/content/ptmppi_inputs")
event = pd.read_csv(INPUT / "event_table_v3.tsv", sep="\t")
seqs = json.loads((INPUT / "uniprot_sequences.json").read_text())
event["label_binary"] = event["effect_label"].map({"inhibit": 0, "enhance": 1}).astype(int)
event["modified_sequence"] = event["modified_uniprot"].map(seqs)
event["partner_sequence"] = event["partner_uniprot"].map(seqs)
assert event["modified_sequence"].notna().all()
assert event["partner_sequence"].notna().all()
print(event.shape)
print([c for c in event.columns if c.startswith("S") and c.endswith("_split")])

# %%
AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq):
    return "".join(a if a in AA else "X" for a in str(seq).upper())

def window_around_site(seq, position, flank=15):
    seq = clean_seq(seq)
    idx = int(position) - 1
    return seq[max(0, idx - flank): min(len(seq), idx + flank + 1)]

def clip_seq(seq, max_len=1022):
    seq = clean_seq(seq)
    if len(seq) <= max_len:
        return seq
    half = max_len // 2
    return seq[:half] + seq[-(max_len - half):]

event["site_window_31"] = [window_around_site(s, p) for s, p in zip(event["modified_sequence"], event["position"])]
event["modified_clip"] = event["modified_sequence"].map(clip_seq)
event["partner_clip"] = event["partner_sequence"].map(clip_seq)

# %%
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

@torch.no_grad()
def embed_sequences(texts, batch_size=16, cache_path=None):
    texts = list(texts)
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)
    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {cache_path or ''}"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        out = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        chunks.append(pooled.detach().cpu().numpy().astype("float32"))
    arr = np.vstack(chunks)
    if cache_path:
        np.save(cache_path, arr)
    return arr

# Embed unique long protein clips once; site windows are per-event.
unique_mod = pd.Series(event["modified_clip"].unique())
unique_partner = pd.Series(event["partner_clip"].unique())
mod_emb_unique = embed_sequences(unique_mod, batch_size=8, cache_path=OUT / "esm2_modified_unique.npy")
partner_emb_unique = embed_sequences(unique_partner, batch_size=8, cache_path=OUT / "esm2_partner_unique.npy")
site_emb = embed_sequences(event["site_window_31"], batch_size=64, cache_path=OUT / "esm2_site_windows.npy")
mod_lookup = dict(zip(unique_mod, mod_emb_unique))
partner_lookup = dict(zip(unique_partner, partner_emb_unique))
mod_emb = np.vstack([mod_lookup[x] for x in event["modified_clip"]])
partner_emb = np.vstack([partner_lookup[x] for x in event["partner_clip"]])
X_esm = np.hstack([site_emb, mod_emb, partner_emb]).astype("float32")
print(X_esm.shape)

# %%
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

categorical = ["ptm_type", "residue", "assay_family", "motif_family", "kinase_proxy"]
numeric = ["position", "interface_signature_size", "interface_contact_count"]
mechanism = event[categorical + numeric].copy()
mechanism[numeric] = mechanism[numeric].fillna(0)
pre = ColumnTransformer(
    [("cat", make_one_hot_encoder(), categorical), ("num", StandardScaler(), numeric)],
    sparse_threshold=0.3,
)
X_mech = pre.fit_transform(mechanism)
X_combined = sparse.hstack([sparse.csr_matrix(X_esm), sparse.csr_matrix(X_mech)], format="csr")
y = event["label_binary"].to_numpy(int)

# %%
def ece_score(y_true, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(total)

def best_threshold(y_valid, p_valid):
    if len(np.unique(y_valid)) < 2:
        return 0.5
    best_t, best_mcc = 0.5, -2
    for t in np.unique(np.quantile(p_valid, np.linspace(0.02, 0.98, 97))):
        mcc = matthews_corrcoef(y_valid, (p_valid >= t).astype(int))
        if mcc > best_mcc:
            best_t, best_mcc = float(t), float(mcc)
    return best_t

def score(split_col, model_name, y_train, y_valid, y_test, p_valid, p_test):
    threshold = best_threshold(y_valid, p_valid)
    pred = (p_test >= threshold).astype(int)
    return {
        "split_col": split_col,
        "model": model_name,
        "train_n": len(y_train),
        "valid_n": len(y_valid),
        "test_n": len(y_test),
        "train_pos_rate": float(np.mean(y_train)),
        "valid_pos_rate": float(np.mean(y_valid)),
        "test_pos_rate": float(np.mean(y_test)),
        "valid_threshold_mcc": threshold,
        "auprc": float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else np.nan,
        "auroc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else np.nan,
        "mcc": float(matthews_corrcoef(y_test, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "ece": ece_score(y_test, p_test),
    }

# %%
split_cols = [c for c in event.columns if c.startswith("S") and c.endswith("_split")]
models = {
    "esm2_logistic": lambda: make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED)),
    "esm2_mechanism_logistic": lambda: make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED)),
    "esm2_pair_mlp": lambda: make_pipeline(StandardScaler(with_mean=False), MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, early_stopping=True, max_iter=120, random_state=SEED)),
    "esm2_random_forest": lambda: RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
}

rows = []
for split_col in split_cols:
    train_idx = np.where(event[split_col].eq("train"))[0]
    valid_idx = np.where(event[split_col].eq("valid"))[0]
    test_idx = np.where(event[split_col].eq("test"))[0]
    if len(test_idx) < 25 or len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        continue
    matrices = {
        "esm2_logistic": sparse.csr_matrix(X_esm),
        "esm2_pair_mlp": sparse.csr_matrix(X_esm),
        "esm2_random_forest": X_esm,
        "esm2_mechanism_logistic": X_combined,
    }
    for model_name, factory in models.items():
        X = matrices[model_name]
        clf = factory()
        clf.fit(X[train_idx], y[train_idx])
        p_valid = clf.predict_proba(X[valid_idx])[:, 1]
        p_test = clf.predict_proba(X[test_idx])[:, 1]
        rows.append(score(split_col, model_name, y[train_idx], y[valid_idx], y[test_idx], p_valid, p_test))
        print(split_col, model_name, rows[-1]["auprc"], rows[-1]["mcc"])

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "shield_gpu_baseline_metrics_colab.tsv", sep="\t", index=False)
metrics.sort_values(["split_col", "auprc"], ascending=[True, False]).head(30)

# %% [markdown]
# ## Optional PTM-Mamba checkpoint run
#
# This is intentionally gated. Put the official checkpoint in Google Drive and set
# `PTM_MAMBA_CKPT`; otherwise the notebook records the blocker instead of pretending
# the baseline ran.

# %%
notes = []
PTM_MAMBA_CKPT = Path("/content/drive/MyDrive/ptm_mamba/best.ckpt")
if PTM_MAMBA_CKPT.exists():
    notes.append({"method": "PTM-Mamba", "status": "checkpoint_present", "note": str(PTM_MAMBA_CKPT)})
    print("Checkpoint present. Install/run PTM-Mamba inference in a separate high-RAM GPU session if Mamba dependencies build successfully.")
else:
    notes.append({
        "method": "PTM-Mamba",
        "status": "blocked",
        "note": "Official code is public, but checkpoint is not present in this Colab runtime. Add it at /content/drive/MyDrive/ptm_mamba/best.ckpt.",
    })

notes.append({
    "method": "DeepPhosPPI",
    "status": "approximated_by_esm2_pair_mlp",
    "note": "Public repo lacks encoded feature caches/pretrained weights; fair rerun requires rebuilding embeddings under S1-S9/S2b.",
})
notes.append({
    "method": "PhosPPI",
    "status": "blocked",
    "note": "Public web-app repo lacks model1.dat/model2.dat and PSSM/NetSurfP batch artifacts.",
})
pd.DataFrame(notes).to_csv(OUT / "shield_gpu_baseline_reproducibility_notes.tsv", sep="\t", index=False)

# %%
with zipfile.ZipFile("/content/ptmppi_shield_colab_outputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUT.glob("*"):
        zf.write(path, path.name)
files.download("/content/ptmppi_shield_colab_outputs.zip")
