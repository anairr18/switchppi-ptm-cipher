# %% [markdown]
# # PTM-CIPHER Strict Real-3Di Ablation
#
# This notebook is the GPU path for testing whether PTM-CIPHER is measurably
# better because it uses real Foldseek-style 3Di structure tokens.
#
# Required inputs:
#
# 1. `ptmppi_shield_colab_inputs.zip`
# 2. `ptmint_protein_structure_information.zip`
#
# Strict no-fallback policy:
#
# - Rows without complete real 3Di coverage for both cropped proteins are
#   excluded and reported.
# - `ptm_cipher_full_3di` uses only real 3Di states `0-19` on unmasked residues.
# - `NO_STRUCTURE_ID` is used only for masked padding, or for the explicit
#   `no_3di` ablation after the strict real-3Di cohort has been built.

# %%
!nvidia-smi
!pip -q install pandas numpy scikit-learn tqdm biopython mini3di

# %%
from google.colab import drive, files
import json
import os
import pickle
import random
import urllib.request
import zipfile
from pathlib import Path

import mini3di
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

SEED = 4242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError("This notebook is intended for a GPU runtime. Switch Colab to Runtime > Change runtime type > GPU.")

INPUT = Path("/content/ptmppi_inputs")
STRUCT_ROOT = Path("/content/ptmppi_structures")
OUT = Path("/content/ptm_cipher_3di_outputs")
for path in [INPUT, STRUCT_ROOT, OUT]:
    path.mkdir(parents=True, exist_ok=True)

GITHUB_REPO = "anairr18/switchppi-ptm-cipher"
GITHUB_BRANCH = "main"
RELEASE_TAG = "colab-data-v1"
USE_GITHUB_INPUTS = True

print("device:", DEVICE)

# %% [markdown]
# ## Fetch the small metadata/code zip
#
# By default this notebook downloads the input package directly from GitHub.
# Set `USE_GITHUB_INPUTS = False` only if you need the manual upload backup.

# %%
if USE_GITHUB_INPUTS:
    small_zip = Path("/content/ptmppi_shield_colab_inputs.zip")
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/colab/ptmppi_shield_colab_inputs.zip"
    print("downloading:", url)
    urllib.request.urlretrieve(url, small_zip)
else:
    uploaded = files.upload()
    small_zips = [name for name in uploaded if name.endswith(".zip") and "ptmppi_shield_colab_inputs" in name]
    if not small_zips:
        raise FileNotFoundError("Upload ptmppi_shield_colab_inputs.zip in this cell.")
    small_zip = Path(small_zips[0])

with zipfile.ZipFile(small_zip) as zf:
    zf.extractall(INPUT)

print("input files:")
for path in sorted(INPUT.iterdir()):
    print(" -", path.name)

# %%
import sys
sys.path.insert(0, str(INPUT))
from ptm_cipher_model import (
    NO_STRUCTURE_ID,
    PTMCipher,
    PTMCipherConfig,
    PTM_STATES,
    RESIDUES,
    ptm_cipher_loss,
)

assert NO_STRUCTURE_ID == 20, "The strict 3Di notebook expects 3Di states 0-19 and NO_STRUCTURE_ID=20."

# %% [markdown]
# ## Provide the PTMint structure archive
#
# Default: download `ptmint_protein_structure_information.zip` from the GitHub Release.
#
# Backup option A: upload `ptmint_protein_structure_information.zip` in the next cell.
#
# Backup option B: set `USE_DRIVE_FOR_STRUCTURE_ZIP = True` and point
# `STRUCTURE_ZIP_DRIVE_PATH` to a copy in Google Drive.

# %%
USE_DRIVE_FOR_STRUCTURE_ZIP = False
STRUCTURE_ZIP_DRIVE_PATH = "/content/drive/MyDrive/ptmint_protein_structure_information.zip"

if USE_GITHUB_INPUTS:
    structure_zip = Path("/content/ptmint_protein_structure_information.zip")
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/ptmint_protein_structure_information.zip"
    print("downloading:", url)
    urllib.request.urlretrieve(url, structure_zip)
elif USE_DRIVE_FOR_STRUCTURE_ZIP:
    drive.mount("/content/drive")
    structure_zip = Path(STRUCTURE_ZIP_DRIVE_PATH)
    if not structure_zip.exists():
        raise FileNotFoundError(f"Structure zip not found: {structure_zip}")
else:
    uploaded = files.upload()
    structure_zips = [name for name in uploaded if name.endswith(".zip") and "structure" in name.lower()]
    if not structure_zips:
        raise FileNotFoundError("Upload ptmint_protein_structure_information.zip in this cell.")
    structure_zip = Path(structure_zips[0])

print("structure archive:", structure_zip, f"{structure_zip.stat().st_size:,} bytes")

# %%
marker = STRUCT_ROOT / ".extracted"
if not marker.exists():
    with zipfile.ZipFile(structure_zip) as zf:
        zf.extractall(STRUCT_ROOT)
    marker.write_text("ok\n")

pdb_files = list(STRUCT_ROOT.rglob("*.pdb"))
if not pdb_files:
    raise FileNotFoundError("No PDB files found after extracting the PTMint structure archive.")

PDB_BY_NAME = {p.name: p for p in pdb_files}
print("PDB files:", len(PDB_BY_NAME))
print("example:", next(iter(PDB_BY_NAME.values())))

# %% [markdown]
# ## Build the strict real-3Di event manifest

# %%
manifest = pd.read_csv(INPUT / "ptm_cipher_input_manifest.tsv", sep="\t").fillna("")
chain_map = pd.read_csv(INPUT / "structure_chain_uniprot_mapping_v2.tsv", sep="\t").fillna("")
event_iface = pd.read_csv(INPUT / "structure_event_interface_mapping_v2.tsv", sep="\t").fillna("")

manifest["label_binary"] = manifest["label_binary"].astype(int)
for frame, cols in [
    (chain_map, ["uniprot_start", "uniprot_end", "chain_length"]),
    (manifest, ["mod_crop_start_0based", "partner_crop_start_0based", "ptm_index_crop_0based"]),
]:
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

def as_bool(value):
    return str(value).strip().lower() == "true"

def basename_from_any_path(value):
    return str(value).replace("\\", "/").split("/")[-1]

chain_map["mapped_bool"] = chain_map["mapped"].map(as_bool)
chain_map["mapping_unique_bool"] = chain_map["mapping_unique"].map(as_bool)
chain_map["pdb_basename"] = chain_map["pdb_file"].map(basename_from_any_path)
chain_map = chain_map[
    chain_map["mapped_bool"]
    & chain_map["mapping_unique_bool"]
    & chain_map["uniprot"].astype(str).ne("")
].copy()

chain_lookup = {}
for row in chain_map.to_dict("records"):
    key = (str(row["complex_id"]), str(row["chain"]), str(row["uniprot"]))
    chain_lookup.setdefault(key, row)

event_iface["site_has_recorded_contact_bool"] = event_iface["site_has_recorded_contact"].map(as_bool)
event_iface["site_at_interface_bool"] = event_iface["site_at_interface"].map(as_bool)
event_iface["distance_to_nearest_interface_site_num"] = pd.to_numeric(
    event_iface["distance_to_nearest_interface_site"], errors="coerce"
).fillna(1e9)
event_iface = event_iface.sort_values(
    ["site_has_recorded_contact_bool", "site_at_interface_bool", "distance_to_nearest_interface_site_num"],
    ascending=[False, False, True],
)

event_records = []
used_events = set()
for row in event_iface.to_dict("records"):
    event_id = str(row["event_id"])
    if event_id in used_events:
        continue
    complex_id = str(row["complex_id"])
    mod_chain = str(row["modified_chain"])
    mod_key = (complex_id, mod_chain, str(row["modified_uniprot"]))
    if mod_key not in chain_lookup:
        continue

    partner_chains = [c.strip() for c in str(row["partner_chains"]).split(",") if c.strip()]
    partner_keys = [(complex_id, c, str(row["partner_uniprot"])) for c in partner_chains]
    partner_keys = [key for key in partner_keys if key in chain_lookup]
    if not partner_keys:
        continue
    partner_keys = sorted(partner_keys, key=lambda key: key[1] == mod_chain)
    partner_key = partner_keys[0]

    event_records.append(
        {
            "event_id": event_id,
            "complex_id": complex_id,
            "modified_chain_3di": mod_chain,
            "partner_chain_3di": partner_key[1],
            "modified_chain_key": json.dumps(mod_key),
            "partner_chain_key": json.dumps(partner_key),
        }
    )
    used_events.add(event_id)

event_struct = pd.DataFrame(event_records)
if event_struct.empty:
    raise RuntimeError("No event-to-structure mappings survived strict chain mapping.")

manifest_by_event = manifest.set_index("event_id").to_dict("index")
prefilter_keep = []
prefilter_drops = []

def crop_within_mapping(manifest_row, chain_row, crop_start_col, seq_col):
    start = int(manifest_row[crop_start_col]) + 1
    end = start + len(str(manifest_row[seq_col])) - 1
    return start >= int(chain_row["uniprot_start"]) and end <= int(chain_row["uniprot_end"])

for row in event_struct.to_dict("records"):
    event_id = row["event_id"]
    manifest_row = manifest_by_event.get(event_id)
    if manifest_row is None:
        prefilter_drops.append({"event_id": event_id, "reason": "event_not_in_manifest"})
        continue
    mod_key = tuple(json.loads(row["modified_chain_key"]))
    partner_key = tuple(json.loads(row["partner_chain_key"]))
    if not crop_within_mapping(manifest_row, chain_lookup[mod_key], "mod_crop_start_0based", "mod_seq_crop"):
        prefilter_drops.append({"event_id": event_id, "reason": "modified_crop_outside_mapped_range"})
        continue
    if not crop_within_mapping(manifest_row, chain_lookup[partner_key], "partner_crop_start_0based", "partner_seq_crop"):
        prefilter_drops.append({"event_id": event_id, "reason": "partner_crop_outside_mapped_range"})
        continue
    prefilter_keep.append(row)

prefilter_drop_by_event = {row["event_id"]: row["reason"] for row in prefilter_drops}
event_struct = pd.DataFrame(prefilter_keep)
pd.DataFrame(prefilter_drops).to_csv(OUT / "ptm_cipher_3di_prefilter_drop_report.tsv", sep="\t", index=False)

if event_struct.empty:
    raise RuntimeError("No event-to-structure mappings retained after strict crop-range prefilter.")

print("events with strict chain candidates after crop-range prefilter:", len(event_struct))
if prefilter_drops:
    print("prefilter drops:", pd.Series([row["reason"] for row in prefilter_drops]).value_counts().to_dict())

# %%
parser = PDBParser(QUIET=True)
encoder = mini3di.Encoder()

def get_chain_residues(chain):
    residues = []
    for residue in chain.get_residues():
        if is_aa(residue, standard=False) and "N" in residue and "CA" in residue and "C" in residue:
            residues.append(residue)
    return residues

def encode_chain_strict(chain_key):
    complex_id, chain_id, uniprot = chain_key
    row = chain_lookup[chain_key]
    pdb_name = row["pdb_basename"]
    pdb_path = PDB_BY_NAME.get(pdb_name)
    if pdb_path is None:
        return None, f"missing_pdb:{pdb_name}"

    try:
        structure = parser.get_structure(f"{complex_id}_{chain_id}", str(pdb_path))
        chain = structure[0][chain_id]
        residues = get_chain_residues(chain)
        states = np.asarray(encoder.encode_chain(chain), dtype=np.int64)
    except Exception as exc:
        return None, f"encode_error:{type(exc).__name__}:{exc}"

    expected_length = int(row["chain_length"])
    if len(states) != len(residues):
        return None, f"state_residue_length_mismatch:{len(states)}!={len(residues)}"
    if len(states) != expected_length:
        return None, f"state_mapping_length_mismatch:{len(states)}!={expected_length}"
    if len(states) == 0 or states.min() < 0 or states.max() >= NO_STRUCTURE_ID:
        return None, "invalid_3di_state_range"

    start = int(row["uniprot_start"])
    return {start + i: int(state) for i, state in enumerate(states)}, ""

CACHE_EVERY = 25
CHAIN_TOKEN_CACHE = OUT / "ptm_cipher_3di_chain_token_cache.pkl"
needed_chain_keys = set()
for row in event_struct.to_dict("records"):
    needed_chain_keys.add(tuple(json.loads(row["modified_chain_key"])))
    needed_chain_keys.add(tuple(json.loads(row["partner_chain_key"])))

if CHAIN_TOKEN_CACHE.exists():
    with CHAIN_TOKEN_CACHE.open("rb") as fh:
        chain_token_maps = pickle.load(fh)
    print("loaded chain token cache:", len(chain_token_maps))
else:
    chain_token_maps = {}

chain_failures = []
keys_to_encode = [key for key in sorted(needed_chain_keys) if key not in chain_token_maps]
print("needed chains:", len(needed_chain_keys), "to encode:", len(keys_to_encode))

for idx, key in enumerate(tqdm(keys_to_encode, desc="Encoding real 3Di chains"), start=1):
    token_map, reason = encode_chain_strict(key)
    if token_map is None:
        chain_failures.append({"complex_id": key[0], "chain": key[1], "uniprot": key[2], "reason": reason})
    else:
        chain_token_maps[key] = token_map
    if idx % CACHE_EVERY == 0:
        with CHAIN_TOKEN_CACHE.open("wb") as fh:
            pickle.dump(chain_token_maps, fh)

with CHAIN_TOKEN_CACHE.open("wb") as fh:
    pickle.dump(chain_token_maps, fh)

chain_failure_df = pd.DataFrame(chain_failures)
chain_failure_df.to_csv(OUT / "ptm_cipher_3di_chain_failures.tsv", sep="\t", index=False)
print("encoded chains:", len(chain_token_maps), "failed chains:", len(chain_failure_df))

# %%
event_lookup = event_struct.set_index("event_id").to_dict("index")
strict_rows = []
drop_rows = []

def ids_for_crop(token_map, crop_start_0based, seq):
    start_pos = int(crop_start_0based) + 1
    positions = range(start_pos, start_pos + len(seq))
    ids = [token_map.get(pos) for pos in positions]
    missing = sum(value is None for value in ids)
    return ids, missing

for row in manifest.to_dict("records"):
    event_id = str(row["event_id"])
    if event_id not in event_lookup:
        drop_rows.append({"event_id": event_id, "reason": prefilter_drop_by_event.get(event_id, "no_strict_event_chain_mapping")})
        continue

    struct_row = event_lookup[event_id]
    mod_key = tuple(json.loads(struct_row["modified_chain_key"]))
    partner_key = tuple(json.loads(struct_row["partner_chain_key"]))
    mod_map = chain_token_maps.get(mod_key)
    partner_map = chain_token_maps.get(partner_key)
    if mod_map is None or partner_map is None:
        drop_rows.append({"event_id": event_id, "reason": "chain_3di_encoding_failed"})
        continue

    mod_seq = str(row["mod_seq_crop"])
    partner_seq = str(row["partner_seq_crop"])
    mod_ids, mod_missing = ids_for_crop(mod_map, row["mod_crop_start_0based"], mod_seq)
    partner_ids, partner_missing = ids_for_crop(partner_map, row["partner_crop_start_0based"], partner_seq)
    if mod_missing or partner_missing:
        drop_rows.append(
            {
                "event_id": event_id,
                "reason": "incomplete_crop_3di_coverage",
                "missing_modified_positions": mod_missing,
                "missing_partner_positions": partner_missing,
            }
        )
        continue

    if any(value is None or value < 0 or value >= NO_STRUCTURE_ID for value in mod_ids + partner_ids):
        drop_rows.append({"event_id": event_id, "reason": "invalid_real_3di_ids"})
        continue

    out = dict(row)
    out.update(
        {
            "structure_complex_id": struct_row["complex_id"],
            "modified_chain_3di": struct_row["modified_chain_3di"],
            "partner_chain_3di": struct_row["partner_chain_3di"],
            "mod_3di_ids": ";".join(str(int(x)) for x in mod_ids),
            "partner_3di_ids": ";".join(str(int(x)) for x in partner_ids),
        }
    )
    strict_rows.append(out)

strict_manifest = pd.DataFrame(strict_rows)
drop_report = pd.DataFrame(drop_rows)

strict_manifest.to_csv(OUT / "ptm_cipher_3di_strict_manifest.tsv", sep="\t", index=False)
drop_report.to_csv(OUT / "ptm_cipher_3di_drop_report.tsv", sep="\t", index=False)

print("original rows:", len(manifest))
print("strict real-3Di rows:", len(strict_manifest))
print("drops by reason:")
print(drop_report["reason"].value_counts(dropna=False) if not drop_report.empty else "none")

if strict_manifest.empty:
    raise RuntimeError("Strict real-3Di cohort is empty. Do not train a fallback model.")

print(strict_manifest.groupby(["S2b_cold_interface_split", "label_binary"]).size())

for split_name in ["train", "valid", "test"]:
    sub = strict_manifest[strict_manifest["S2b_cold_interface_split"] == split_name]
    if sub.empty or sub["label_binary"].nunique() < 2:
        raise RuntimeError(
            f"Strict real-3Di cohort lacks both classes in {split_name}. "
            "Stop here; do not use fallback structure tokens."
        )

# %%
def assert_real_3di_manifest(frame):
    bad_rows = []
    for row in frame.to_dict("records"):
        for col, seq_col in [("mod_3di_ids", "mod_seq_crop"), ("partner_3di_ids", "partner_seq_crop")]:
            ids = [int(x) for x in str(row[col]).split(";") if x != ""]
            if len(ids) != len(str(row[seq_col])) or any(x < 0 or x >= NO_STRUCTURE_ID for x in ids):
                bad_rows.append((row["event_id"], col))
    if bad_rows:
        raise AssertionError(f"Found non-real 3Di tokens in strict manifest: {bad_rows[:5]}")

assert_real_3di_manifest(strict_manifest)
print("Strict real-3Di assertion passed.")

# %% [markdown]
# ## Train full model and ablations

# %%
for col in ["assay_family", "topology_pair_community"]:
    enc = LabelEncoder()
    strict_manifest[col + "_id"] = enc.fit_transform(strict_manifest[col].astype(str))
    print(col, len(enc.classes_))

residue_to_id = {aa: i for i, aa in enumerate(RESIDUES)}
ptm_to_id = {name: i for i, name in enumerate(PTM_STATES)}
MAX_MOD = int(strict_manifest["mod_seq_crop"].map(len).max())
MAX_PARTNER = int(strict_manifest["partner_seq_crop"].map(len).max())

BATCH_SIZE = 8
EPOCHS = 8
ABLATIONS_TO_RUN = [
    "ptm_cipher_full_3di",
    "no_3di",
    "no_ptm_state",
    "no_contacts",
    "no_adversary",
    "no_delta_head",
]

print("max lengths:", MAX_MOD, MAX_PARTNER)
print("ablations:", ABLATIONS_TO_RUN)

# %%
class Strict3DiCipherDataset(Dataset):
    def __init__(self, frame, ablation):
        self.frame = frame.reset_index(drop=True)
        self.ablation = ablation

    def __len__(self):
        return len(self.frame)

    @staticmethod
    def parse_ids(value):
        return [int(x) for x in str(value).split(";") if x != ""]

    def encode_seq(self, seq, max_len):
        ids = np.full(max_len, residue_to_id["X"], dtype=np.int64)
        mask = np.zeros(max_len, dtype=np.bool_)
        for i, aa in enumerate(str(seq)[:max_len]):
            ids[i] = residue_to_id.get(aa, residue_to_id["X"])
            mask[i] = True
        return ids, mask

    def encode_3di(self, token_text, seq, max_len):
        ids = self.parse_ids(token_text)
        if len(ids) != len(str(seq)):
            raise ValueError(f"3Di length mismatch: {len(ids)} != {len(str(seq))}")
        if any(x < 0 or x >= NO_STRUCTURE_ID for x in ids):
            raise ValueError("Full 3Di model received a non-real 3Di state.")
        out = np.full(max_len, NO_STRUCTURE_ID, dtype=np.int64)
        out[: len(ids)] = np.asarray(ids, dtype=np.int64)
        if self.ablation == "no_3di":
            out[: len(ids)] = NO_STRUCTURE_ID
        return out

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        mod_ids, mod_mask = self.encode_seq(row.mod_seq_crop, MAX_MOD)
        partner_ids, partner_mask = self.encode_seq(row.partner_seq_crop, MAX_PARTNER)
        mod_structure = self.encode_3di(row.mod_3di_ids, row.mod_seq_crop, MAX_MOD)
        partner_structure = self.encode_3di(row.partner_3di_ids, row.partner_seq_crop, MAX_PARTNER)

        unmod_ptm = np.zeros(MAX_MOD, dtype=np.int64)
        mod_ptm = np.zeros(MAX_MOD, dtype=np.int64)
        ptm_index = int(row.ptm_index_crop_0based)
        if self.ablation != "no_ptm_state" and 0 <= ptm_index < MAX_MOD:
            mod_ptm[ptm_index] = ptm_to_id.get(str(row.ptm_state), ptm_to_id["other"])

        contact = np.zeros((MAX_MOD, MAX_PARTNER), dtype=np.bool_)
        adjacency = np.eye(MAX_MOD, dtype=np.bool_)
        if self.ablation != "no_contacts":
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
    return {key: torch.as_tensor(np.stack([item[key] for item in batch])) for key in batch[0]}

def move(batch):
    return {key: value.to(DEVICE) for key, value in batch.items()}

train_df = strict_manifest[strict_manifest["S2b_cold_interface_split"] == "train"].copy()
valid_df = strict_manifest[strict_manifest["S2b_cold_interface_split"] == "valid"].copy()
test_df = strict_manifest[strict_manifest["S2b_cold_interface_split"] == "test"].copy()

# %%
def metric_row(model_name, split_name, y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "split_col": "S2b_cold_interface_split",
        "model": model_name,
        "split": split_name,
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }

def choose_threshold(y, p):
    thresholds = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    best_threshold = 0.5
    best_score = -1e9
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        score = matthews_corrcoef(y, pred) + balanced_accuracy_score(y, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold

def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch)
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
                adversary_alpha=0.0,
            )
            loss = ptm_cipher_loss(out, batch["label"], lambda_brier=0.05, lambda_adversary=0.0)["loss"]
            losses.append(float(loss.detach().cpu()))
            ys.extend(batch["label"].detach().cpu().numpy().tolist())
            ps.extend(out["logits"].softmax(dim=-1)[:, 1].detach().cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(ps), float(np.mean(losses))

def train_one(ablation):
    train_loader = DataLoader(
        Strict3DiCipherDataset(train_df, ablation),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        Strict3DiCipherDataset(valid_df, ablation),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        Strict3DiCipherDataset(test_df, ablation),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    config = PTMCipherConfig(
        dim=128,
        heads=4,
        layers=2,
        ff_dim=384,
        dropout=0.20,
        graph_layers=2,
        classes=2,
        head_input="modified" if ablation == "no_delta_head" else "delta",
        adversary_dims={
            "assay_family": int(strict_manifest["assay_family_id"].nunique()),
            "topology_pair_community": int(strict_manifest["topology_pair_community_id"].nunique()),
        },
    )
    model = PTMCipher(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lambda_adversary = 0.0 if ablation == "no_adversary" else 0.05

    best_score = -1
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"{ablation} epoch {epoch}", leave=False):
            batch = move(batch)
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
                adversary_alpha=0.0 if ablation == "no_adversary" else min(1.0, epoch / 3),
            )
            loss_dict = ptm_cipher_loss(
                out,
                batch["label"],
                {
                    "assay_family": batch["assay_family"],
                    "topology_pair_community": batch["topology_pair_community"],
                },
                lambda_brier=0.05,
                lambda_adversary=lambda_adversary,
            )
            loss = loss_dict["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        y_valid, p_valid, valid_loss = evaluate(model, valid_loader)
        threshold = choose_threshold(y_valid, p_valid)
        row = metric_row(ablation, "valid", y_valid, p_valid, threshold)
        row.update({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "valid_loss": valid_loss})
        history.append(row)
        print(row)

        if row["auprc"] > best_score:
            best_score = row["auprc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_valid, p_valid, valid_loss = evaluate(model, valid_loader)
    threshold = choose_threshold(y_valid, p_valid)
    y_test, p_test, test_loss = evaluate(model, test_loader)
    valid_metrics = metric_row(ablation, "valid_best", y_valid, p_valid, threshold)
    valid_metrics["loss"] = valid_loss
    test_metrics = metric_row(ablation, "test", y_test, p_test, threshold)
    test_metrics["loss"] = test_loss
    torch.save(best_state, OUT / f"{ablation}_best_state.pt")
    return history, [valid_metrics, test_metrics]

# %%
all_history = []
all_metrics = []
for ablation in ABLATIONS_TO_RUN:
    history, metrics = train_one(ablation)
    all_history.extend(history)
    all_metrics.extend(metrics)
    pd.DataFrame(all_history).to_csv(OUT / "ptm_cipher_3di_ablation_history_colab.tsv", sep="\t", index=False)
    pd.DataFrame(all_metrics).to_csv(OUT / "ptm_cipher_3di_ablation_metrics_colab.tsv", sep="\t", index=False)

metrics_df = pd.DataFrame(all_metrics)
display(metrics_df.sort_values(["split", "auprc"], ascending=[True, False]))

# %%
full = metrics_df[(metrics_df["model"] == "ptm_cipher_full_3di") & (metrics_df["split"] == "test")]
no_3di = metrics_df[(metrics_df["model"] == "no_3di") & (metrics_df["split"] == "test")]
if not full.empty and not no_3di.empty:
    delta = float(full.iloc[0]["auprc"] - no_3di.iloc[0]["auprc"])
    print(f"Test AUPRC delta, full real-3Di minus no_3di: {delta:.4f}")
    if delta <= 0:
        print("Interpretation guardrail: real 3Di did not improve AUPRC in this run. Do not claim architecture superiority.")
    else:
        print("Interpretation guardrail: real 3Di improved AUPRC; verify with repeated seeds before a strong claim.")

# %% [markdown]
# ## Literature-Level AUROC/AUPRC Comparison
#
# This is **not** a same-split reproduction. It answers only:
# "Is my strict 3Di model numerically above or below reported AUROC/AUPRC
# values in related papers?"
#
# Use this table for sanity checking, not as a final SOTA claim.

# %%
if "metrics_df" not in globals():
    metrics_df = pd.read_csv(OUT / "ptm_cipher_3di_ablation_metrics_colab.tsv", sep="\t")

own = metrics_df[(metrics_df["model"] == "ptm_cipher_full_3di") & (metrics_df["split"] == "test")].copy()
if own.empty:
    raise RuntimeError("No ptm_cipher_full_3di test row found. Run the ablation cell first.")
own = own.iloc[0]

reported = pd.DataFrame(
    [
        {
            "reported_method": "DeepPhosPPI-2",
            "reported_dataset": "Betts independent benchmark",
            "reported_auroc": 0.820,
            "reported_auprc": 0.921,
            "value_type": "exact_table",
            "source": "Gong et al. 2025 Briefings in Bioinformatics Table 2",
            "url": "https://academic.oup.com/bib/article/26/5/bbaf462/8248860",
            "comparison_note": "Phosphorylation-only enhance/inhibit task; not the strict S2b cold-interface split.",
        },
        {
            "reported_method": "AttCNN-PhosPPI",
            "reported_dataset": "Betts independent benchmark",
            "reported_auroc": 0.744,
            "reported_auprc": 0.865,
            "value_type": "exact_table",
            "source": "Gong et al. 2025 Briefings in Bioinformatics Table 2",
            "url": "https://academic.oup.com/bib/article/26/5/bbaf462/8248860",
            "comparison_note": "DeepPhosPPI component model; not the strict S2b cold-interface split.",
        },
        {
            "reported_method": "Transformer-PhosPPI",
            "reported_dataset": "Betts independent benchmark",
            "reported_auroc": 0.748,
            "reported_auprc": 0.858,
            "value_type": "exact_table",
            "source": "Gong et al. 2025 Briefings in Bioinformatics Table 2",
            "url": "https://academic.oup.com/bib/article/26/5/bbaf462/8248860",
            "comparison_note": "DeepPhosPPI component model; not the strict S2b cold-interface split.",
        },
        {
            "reported_method": "PTM-Mamba",
            "reported_dataset": "PTMint PTM effect on PPI task",
            "reported_auroc": 0.63,
            "reported_auprc": 0.79,
            "value_type": "approx_from_figure",
            "source": "Peng et al. 2025 Nature Methods Figure 2c",
            "url": "https://www.nature.com/articles/s41592-025-02656-9",
            "comparison_note": "Approximate from plotted bars; official exact source-data table was not exposed in the paper text.",
        },
        {
            "reported_method": "PTM-SaProt",
            "reported_dataset": "PTMint PTM effect on PPI task",
            "reported_auroc": 0.61,
            "reported_auprc": 0.77,
            "value_type": "approx_from_figure",
            "source": "Peng et al. 2025 Nature Methods Figure 2c",
            "url": "https://www.nature.com/articles/s41592-025-02656-9",
            "comparison_note": "Approximate structure-aware PLM baseline from plotted bars.",
        },
        {
            "reported_method": "PTM-Transformer",
            "reported_dataset": "PTMint PTM effect on PPI task",
            "reported_auroc": 0.61,
            "reported_auprc": 0.72,
            "value_type": "approx_from_figure",
            "source": "Peng et al. 2025 Nature Methods Figure 2c",
            "url": "https://www.nature.com/articles/s41592-025-02656-9",
            "comparison_note": "Approximate from plotted bars.",
        },
        {
            "reported_method": "ESM-2-3B",
            "reported_dataset": "PTMint PTM effect on PPI task",
            "reported_auroc": 0.57,
            "reported_auprc": 0.76,
            "value_type": "approx_from_figure",
            "source": "Peng et al. 2025 Nature Methods Figure 2c",
            "url": "https://www.nature.com/articles/s41592-025-02656-9",
            "comparison_note": "Approximate from plotted bars.",
        },
        {
            "reported_method": "ESM-2-650M",
            "reported_dataset": "PTMint PTM effect on PPI task",
            "reported_auroc": 0.50,
            "reported_auprc": 0.70,
            "value_type": "approx_from_figure",
            "source": "Peng et al. 2025 Nature Methods Figure 2c",
            "url": "https://www.nature.com/articles/s41592-025-02656-9",
            "comparison_note": "Approximate from plotted bars.",
        },
        {
            "reported_method": "OneHot(+PTM)",
            "reported_dataset": "PTMint PTM effect on PPI task",
            "reported_auroc": 0.50,
            "reported_auprc": 0.69,
            "value_type": "approx_from_figure",
            "source": "Peng et al. 2025 Nature Methods Figure 2c",
            "url": "https://www.nature.com/articles/s41592-025-02656-9",
            "comparison_note": "Approximate from plotted bars.",
        },
    ]
)

reported["your_model"] = "ptm_cipher_full_3di"
reported["your_dataset"] = "strict S2b cold-interface test"
reported["your_auroc"] = float(own["auroc"])
reported["your_auprc"] = float(own["auprc"])
reported["delta_auroc_yours_minus_reported"] = reported["your_auroc"] - reported["reported_auroc"]
reported["delta_auprc_yours_minus_reported"] = reported["your_auprc"] - reported["reported_auprc"]
reported["beats_reported_auroc"] = reported["delta_auroc_yours_minus_reported"] > 0
reported["beats_reported_auprc"] = reported["delta_auprc_yours_minus_reported"] > 0
reported["sota_claim_safe"] = False
reported["why_not_sota_claim"] = (
    "Cross-study metric comparison only: different datasets/splits/class priors. "
    "Use same-split reruns or official predictions before claiming SOTA."
)

reported = reported.sort_values(["reported_auprc", "reported_auroc"], ascending=False)
reported.to_csv(OUT / "literature_metric_comparison_colab.tsv", sep="\t", index=False)

display_cols = [
    "reported_method",
    "reported_dataset",
    "reported_auroc",
    "reported_auprc",
    "your_auroc",
    "your_auprc",
    "delta_auroc_yours_minus_reported",
    "delta_auprc_yours_minus_reported",
    "beats_reported_auroc",
    "beats_reported_auprc",
    "value_type",
]
display(reported[display_cols])

best_reported_auprc = reported["reported_auprc"].max()
best_reported_auroc = reported["reported_auroc"].max()
print(f"Your strict S2b full-3Di AUROC: {own['auroc']:.4f}")
print(f"Your strict S2b full-3Di AUPRC: {own['auprc']:.4f}")
print(f"Best reported literature AUROC in this table: {best_reported_auroc:.4f}")
print(f"Best reported literature AUPRC in this table: {best_reported_auprc:.4f}")

if own["auprc"] > best_reported_auprc and own["auroc"] > best_reported_auroc:
    print("Numerically above the listed literature metrics, but still cross-study. Claim: promising; same-split validation still needed.")
elif own["auprc"] > best_reported_auprc:
    print("AUPRC is numerically above the listed literature metrics, but AUROC is not. Be careful with SOTA language.")
else:
    print("Not numerically above the strongest listed AUPRC. Better claim: stricter cold-interface evaluation plus real-3Di contribution.")

# %%
with zipfile.ZipFile("/content/ptm_cipher_3di_colab_outputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUT.glob("*"):
        zf.write(path, path.name)

files.download("/content/ptm_cipher_3di_colab_outputs.zip")
