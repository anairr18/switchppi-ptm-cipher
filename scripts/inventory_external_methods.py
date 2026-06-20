from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external_methods"
TABLES = ROOT / "results_v2" / "tables"


def exists(path: Path) -> bool:
    return path.exists()


def main() -> None:
    phosppi = EXT / "PhosPPI"
    deep = EXT / "DeepPhosPPI"
    ptm_mamba = EXT / "ptm-mamba"
    rows = [
        {
            "method": "PhosPPI",
            "local_path": str(phosppi),
            "retrieved": exists(phosppi),
            "code_available": exists(phosppi / "application.py"),
            "data_available": False,
            "weights_available": exists(phosppi / "model1.dat") and exists(phosppi / "model2.dat"),
            "rerunnable_now": False,
            "blocker": "Public GitHub web app references model1.dat/model2.dat, NetSurfP, PSI-BLAST, and PSSM pipeline, but weights/data are not included in cloned repository.",
            "paper_role": "Report as contacted/blocked unless authors provide weights or a reproducible batch predictor; approximate with sequence/motif baselines meanwhile.",
        },
        {
            "method": "DeepPhosPPI",
            "local_path": str(deep),
            "retrieved": exists(deep),
            "code_available": exists(deep / "DeepPhosPPI" / "TASK2" / "TASK2_train_CNN.py"),
            "data_available": exists(deep / "Datasets" / "DatasetB"),
            "weights_available": False,
            "rerunnable_now": False,
            "blocker": "Code and public protein-list/split pickles are available, but no pretrained weights detected and the expected encoded feature caches are absent/empty .pkl.txt placeholders; fair Shield rerun requires rebuilding embeddings and adapting training code to S1-S9 splits.",
            "paper_role": "Priority-1 baseline/data-lineage comparator. Current audit maps its public rows to PTMint-derived events; next step is a re-embedding rerun under Shield splits.",
        },
        {
            "method": "PTM-Mamba",
            "local_path": str(ptm_mamba),
            "retrieved": exists(ptm_mamba),
            "code_available": exists(ptm_mamba / "protein_lm" / "modeling" / "scripts" / "infer.py"),
            "data_available": False,
            "weights_available": False,
            "rerunnable_now": False,
            "blocker": "Official repository is cloned and provides Docker/inference code, but model checkpoints are external Google Drive artifacts not present locally; inference also expects CUDA/Mamba dependencies while this sprint has no detected GPU.",
            "paper_role": "Priority-1 representation baseline because PTM-aware PLM benchmarking on PTMint is a direct preemption risk.",
        },
        {
            "method": "Betts/Mechismo-style interface rules",
            "local_path": str(ROOT / "data" / "raw" / "ptmint_structure_information"),
            "retrieved": exists(ROOT / "data" / "raw" / "ptmint_structure_information"),
            "code_available": False,
            "data_available": exists(ROOT / "data" / "raw" / "ptmint_structure_information" / "Protein structure information" / "complex_interface.csv"),
            "weights_available": False,
            "rerunnable_now": True,
            "blocker": "Exact chain-to-UniProt event mapping and interface-localization audit are now available for a structure-supported subset, but no Mechismo implementation or Foldseek/contact-cluster similarity split is implemented yet.",
            "paper_role": "Partial structure baseline now present via interface localization; priority-1 next step is interface-similarity shielding and a formal Mechismo/PINDER-style comparator.",
        },
        {
            "method": "ELM motif rules",
            "local_path": "",
            "retrieved": False,
            "code_available": False,
            "data_available": False,
            "weights_available": False,
            "rerunnable_now": True,
            "blocker": "Current implementation uses heuristic motif proxies; official ELM class/instance files should replace heuristics for publication.",
            "paper_role": "Already approximated by motif_only; replace with official ELM rules before submission.",
        },
    ]
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "external_method_reproducibility_inventory.tsv", sep="\t", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
