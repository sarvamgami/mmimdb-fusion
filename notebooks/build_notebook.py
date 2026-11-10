"""Builds run_all.ipynb from the cells below (keeps the notebook diff-friendly in git)."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Multimodal Movie Genre Classification (MM-IMDb): early vs late vs intermediate fusion

Runs the whole project end to end: download → data-quality checks → frozen feature extraction
(EfficientNet-B0 + BERT) → 7 models × 3 seeds → evaluation, ablations, bias analysis → every figure
and table used in the report (saved to `results/`).

**Colab:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
Total time ≈ 60–80 min (download ~15–25, feature extraction ~25, training + ablations ~15).
Features are cached in Google Drive, so a re-run after a disconnect skips extraction.
""")

md("## 0. Setup")
code("""
import os, sys, subprocess, time
IN_COLAB = "google.colab" in sys.modules

# >>> EDIT: your public GitHub repo (leave as-is if you uploaded the repo folder to /content yourself)
REPO_URL = "https://github.com/sarvamgami/mmimdb-fusion.git"
REPO_DIR = "/content/mmimdb-fusion" if IN_COLAB else os.path.abspath("..")

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    if not os.path.exists(REPO_DIR):
        if os.path.exists("/content/mmimdb-fusion.zip"):
            subprocess.run(["unzip", "-q", "/content/mmimdb-fusion.zip", "-d", "/content"], check=True)
        else:
            subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

import torch
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to T4 GPU!")
""")

code("""
from src.config import Config, GENRES
cfg = Config(
    data_dir="/content/mmimdb" if IN_COLAB else "data/mmimdb",
    cache_dir="/content/drive/MyDrive/mmimdb_cache" if IN_COLAB else "cache",
    results_dir=os.path.join(REPO_DIR, "results"),
)
FIG, TAB = cfg.results_dir / "figures", cfg.results_dir / "tables"
cfg
""")

md("""
## 1. Download MM-IMDb (Arevalo et al., 2017)
~8 GB archive from archive.org, extracted to the Colab local disk (much faster to read than Drive).
""")
code("""
from pathlib import Path
t0 = time.time()
if not (Path(cfg.data_dir) / "split.json").exists() and not list(Path(cfg.data_dir).glob("*/split.json")):
    os.makedirs("/content", exist_ok=True)
    !wget -c -q --show-progress -O /content/mmimdb.tar.gz https://archive.org/download/mmimdb/mmimdb.tar.gz
    !tar -xzf /content/mmimdb.tar.gz -C /content && rm /content/mmimdb.tar.gz
print(f"data ready in {time.time() - t0:.0f}s")
print(os.listdir(cfg.data_dir)[:10])
""")

md("## 2. Load data and assess quality")
code("""
import numpy as np, pandas as pd
from src.data import load_mmimdb, deduplicate, quality_summary, labels_matrix

raw, problems = load_mmimdb(cfg.data_dir, GENRES)
print("Rows dropped while loading:", problems)
q_before = quality_summary(raw, GENRES)
df, dedup = deduplicate(raw)
print("De-duplication:", dedup)
q_after = quality_summary(df, GENRES)
quality = pd.DataFrame({"before de-dup": q_before, "after de-dup": q_after})
quality.to_csv(TAB / "data_quality.csv"); quality
""")

md("## 3. Frozen feature extraction (cached)")
code("""
from src.features import extract_image_features, extract_text_features
img_res = extract_image_features(df, cfg)
txt_res = extract_text_features(df, cfg)
assert img_res["ids"] == list(df["id"]) and txt_res["ids"] == list(df["id"]), "cache does not match the data - delete the cache folder"
info = img_res["image_info"]
img_quality = {
    "unreadable_images": int((info["readable"] == 0).sum()),
    "non_RGB_images (greyscale/CMYK/palette, converted)": int(info["non_rgb"].sum()),
    "width_min/median/max": f"{info['width'].min()}/{int(info['width'].median())}/{info['width'].max()}",
    "height_min/median/max": f"{info['height'].min()}/{int(info['height'].median())}/{info['height'].max()}",
    "plots_truncated_at_512_tokens": f"{txt_res['truncated_share']:.2%}",
    "image_extraction_seconds": round(img_res["seconds"]), "text_extraction_seconds": round(txt_res["seconds"]),
    "peak_gpu_GB_images": round(img_res["peak_gpu_gb"], 2),
}
pd.Series(img_quality).to_csv(TAB / "feature_quality.csv"); pd.Series(img_quality)
""")

md("### Data-quality figures (Fig. 1–2, Appendix)")
code("""
from src import plots
Y = labels_matrix(df, GENRES)
split = df["split"].values
train_counts = Y[split == "train"].sum(0)
plots.genre_distribution(GENRES, train_counts, FIG / "fig1_genre_distribution.png")
aspect = (info["height"] / info["width"].replace(0, np.nan)).fillna(1).values
plots.data_quality(Y.sum(1), df["plot"].str.split().str.len().values, txt_res["n_tokens"], cfg.max_tokens, aspect, FIG / "fig2_data_quality.png")
plots.cooccurrence(Y[split == "train"], GENRES, FIG / "figA1_cooccurrence.png")
""")

md("""
## 4. Prepare features
Each feature dimension is z-scored with **training-set statistics only** (Week 6), so validation/test never leak into preprocessing.
""")
code("""
from src.train import standardise
tr = split == "train"
IMG = standardise(img_res["feats"].numpy(), tr).astype(np.float32)
TXT = standardise(txt_res["feats"].numpy(), tr).astype(np.float32)
TXT_CLS = standardise(txt_res["cls"].numpy(), tr).astype(np.float32)
print(IMG.shape, TXT.shape, Y.shape, {s: int((split == s).sum()) for s in ("train", "val", "test")})
""")

md("""
## 5. Main comparison: 7 approaches × 3 seeds
Early stopping on validation loss; per-genre thresholds tuned on validation; test used once for reporting (Week 7).
""")
code("""
from src.experiment import run_all, summarise, run_ablation
ORDER = ["Image only", "Text only", "Early fusion", "Late fusion", "Intermediate (concat)", "Modality attention", "GMU"]
runs, art = run_all(IMG, TXT, Y, split, cfg)
runs.to_csv(TAB / "all_runs.csv", index=False)

MAIN = ["test_micro_F1", "test_macro_F1", "test_weighted_F1", "test_samples_F1", "test_mAP", "test_hamming"]
EFF = ["params", "train_seconds", "infer_ms_per_1k", "best_epoch"]
table, mean, std = summarise(runs, MAIN + ["val_macro_F1", "train_micro_F1"], ORDER)
_, eff_mean, _ = summarise(runs, EFF, ORDER)
table.to_csv(TAB / "table_main_results.csv")
eff_mean.round(1).to_csv(TAB / "table_efficiency.csv")

# Model selection uses VALIDATION macro-F1 only (never the test set)
fusion = [m for m in ORDER if m not in ("Image only", "Text only")]
BEST = mean.loc[fusion, "val_macro_F1"].idxmax()
# Ablations need a trainable network, so they use the best fusion model that has one
ABL_MODEL = mean.loc[[m for m in fusion if m != "Late fusion"], "val_macro_F1"].idxmax()
print("Selected model (best validation macro-F1):", BEST, "| ablation model:", ABL_MODEL)
display(table[MAIN]); display(eff_mean.round(2))
""")

code("""
# Published reference (Arevalo et al., 2017, Table 2 - GMU with VGG + word2vec features). Verify against the paper.
reference = pd.DataFrame({"micro_F1": [0.630], "macro_F1": [0.541], "weighted_F1": [0.617], "samples_F1": [0.630]},
                         index=["GMU (Arevalo et al., 2017, reported)"])
reference.to_csv(TAB / "table_published_reference.csv"); reference
""")

code("""
plots.results_bars(mean[MAIN[:5]], std[MAIN[:5]], MAIN[:5], FIG / "fig3_main_results.png",
                   "Test performance of unimodal baselines and five fusion strategies")
plots.learning_curves({m: art["hist"][m] for m in ORDER if m in art["hist"]}, FIG / "fig4_learning_curves.png")
plots.overfit_gap(mean, FIG / "fig5_overfitting_gap.png")
_, em, _ = summarise(runs, ["params", "test_macro_F1"], ORDER)
plots.efficiency(em, FIG / "figA2_efficiency.png")
""")

md("### Threshold tuning: fixed 0.5 vs per-genre thresholds tuned on validation")
code("""
t05, m05, s05 = summarise(runs, ["test_micro_F1@0.5", "test_macro_F1@0.5"], ORDER)
thr_table = pd.concat([t05, table[["test_micro_F1", "test_macro_F1"]].rename(columns=lambda c: c + " (tuned)")], axis=1)
thr_table.to_csv(TAB / "table_threshold_tuning.csv"); thr_table
""")

md("## 6. Per-genre analysis and what each modality contributes")
code("""
from src.evaluate import per_genre_f1, subgroup_table
te = split == "test"
Yte, df_te = Y[te], df[te].reset_index(drop=True)
pg = {m: per_genre_f1(Yte, art["probs"][m], art["thr"][m], GENRES, train_counts) for m in ["Image only", "Text only", BEST]}
pg_table = pd.DataFrame({"train_count": pg[BEST].set_index("genre")["train_count"],
                         **{f"F1 {m}": t.set_index("genre")["F1"] for m, t in pg.items()}})
pg_table.round(3).to_csv(TAB / "table_per_genre_f1.csv")
plots.per_genre(pg, FIG / "fig6_per_genre_f1.png")
pg_table.round(3)
""")

code("""
# How much each genre relies on the poster vs the plot
gmu = art["models"]["GMU"]
_ = gmu(torch.tensor(IMG[te]).to(next(gmu.parameters()).device), torch.tensor(TXT[te]).to(next(gmu.parameters()).device))
gate = gmu.last_gate.mean(1).cpu().numpy()                      # mean image share per test movie
gate_share = np.array([gate[Yte[:, c] > 0].mean() for c in range(len(GENRES))])
plots.modality_share(GENRES, gate_share, FIG / "fig7_gmu_gate.png",
                     "GMU gate: average weight given to the POSTER, by true genre (test set)", "Mean image gate z")
plots.modality_share(GENRES, art["late_weights"], FIG / "figA3_late_weights.png",
                     "Late fusion: poster weight per genre learned on validation", "Image weight w")
att = art["models"]["Modality attention"]
_ = att(torch.tensor(IMG[te]).to(next(att.parameters()).device), torch.tensor(TXT[te]).to(next(att.parameters()).device))
print("Modality attention - mean weight on poster:", round(float(att.last_weights[:, 0].mean()), 3))
pd.DataFrame({"genre": GENRES, "gmu_image_gate": gate_share, "late_fusion_image_weight": art["late_weights"]}).round(3).to_csv(TAB / "table_modality_share.csv", index=False)
""")

md("## 7. Failure cases")
code("""
worst = plots.failures(df_te, Yte, art["probs"][BEST], art["thr"][BEST], GENRES, FIG / "fig8_failures.png",
                       title=f"Worst test predictions of {BEST} (true genres vs top-3 predicted probabilities)")
df_te.iloc[worst][["title", "year", "genres", "plot"]].assign(plot=lambda d: d["plot"].str[:200]).to_csv(TAB / "failure_cases.csv", index=False)
df_te.iloc[worst][["title", "year", "genres"]]
""")

md("""
## 8. Ablations (selected model, 3 seeds each)
Text pooling (mean vs `[CLS]`), class weighting, regularisation (overfitting check), hidden size.
""")
code("""
variants = {
    "Default (mean-pool, pos_weight, dropout 0.3)": {},
    "BERT [CLS] instead of mean-pooling": {"txt": TXT_CLS},
    "No class weighting (plain BCE)": {"cfg": {"use_pos_weight": False}},
    "No regularisation (dropout 0, weight decay 0)": {"cfg": {"dropout": 0.0, "weight_decay": 0.0}},
    "Hidden 128": {"cfg": {"hidden": 128}},
    "Hidden 512": {"cfg": {"hidden": 512}},
}
abl = run_ablation(IMG, TXT, Y, split, cfg, ABL_MODEL, variants)
abl["model"] = abl["tag"]
abl_t, abl_m, abl_s = summarise(abl, ["test_micro_F1", "test_macro_F1", "train_micro_F1", "best_epoch"], list(variants))
abl_t["train-test gap"] = (abl_m["train_micro_F1"] - abl_m["test_micro_F1"]).round(3)
abl_t.to_csv(TAB / "table_ablations.csv")
plots.ablation_bars(abl_m, abl_s, FIG / "fig9_ablations.png", f"Ablations of {ABL_MODEL} (test set)")
abl_t
""")

md("## 9. Bias analysis: does performance differ by country, language and era?")
code("""
groups = {
    "USA": df_te["countries"].map(lambda c: "USA" in c).values,
    "Non-USA": df_te["countries"].map(lambda c: "USA" not in c).values,
    "English": df_te["languages"].map(lambda l: "English" in l).values,
    "Non-English": df_te["languages"].map(lambda l: "English" not in l).values,
    "Before 1990": (df_te["year"] < 1990).values,
    "1990 or later": (df_te["year"] >= 1990).values,
}
bias = subgroup_table(df_te, Yte, art["probs"][BEST], art["thr"][BEST], groups)
bias.round(3).to_csv(TAB / "table_bias_subgroups.csv", index=False)
plots.subgroups(bias, FIG / "fig10_bias_subgroups.png", f"{BEST}: test performance by subgroup")
# Share of training movies in each group, for context
tr_df = df[split == "train"]
print({"train USA share": round(tr_df["countries"].map(lambda c: "USA" in c).mean(), 3),
       "train English share": round(tr_df["languages"].map(lambda l: "English" in l).mean(), 3),
       "train pre-1990 share": round((tr_df["year"] < 1990).mean(), 3)})
bias.round(3)
""")

md("## 10. Summary for the report (copy everything below)")
code("""
pd.set_option("display.width", 250)
print("=== DATA QUALITY ===");      print(quality.to_string()); print(pd.Series(img_quality).to_string())
print("\\n=== MAIN RESULTS (test, mean ± std over 3 seeds, tuned thresholds) ==="); print(table[MAIN].to_string())
print("\\n=== EFFICIENCY ===");       print(eff_mean.round(2).to_string())
print("\\n=== THRESHOLD TUNING ==="); print(thr_table.to_string())
print("\\n=== OVERFITTING (micro-F1) ==="); print((mean[["train_micro_F1", "test_micro_F1"]].assign(gap=lambda d: d.iloc[:, 0] - d.iloc[:, 1])).round(3).to_string())
print("\\nSelected model:", BEST, "| ablation model:", ABL_MODEL)
print("\\n=== PER-GENRE F1 ===");     print(pg_table.round(3).to_string())
print("\\n=== MODALITY SHARE ===");   print(pd.read_csv(TAB / 'table_modality_share.csv').to_string())
print("\\n=== ABLATIONS ===");        print(abl_t.to_string())
print("\\n=== BIAS ===");             print(bias.round(3).to_string())
print("\\n=== FAILURES ===");         print(df_te.iloc[worst][["title", "year", "genres"]].to_string())
""")

code("""
# Download all figures and tables in one zip (also commit results/ to GitHub)
!cd {REPO_DIR} && zip -qr /content/results.zip results
if IN_COLAB:
    from google.colab import files; files.download("/content/results.zip")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"provenance": []},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, Path(__file__).with_name("run_all.ipynb"))
print("wrote run_all.ipynb with", len(cells), "cells")
