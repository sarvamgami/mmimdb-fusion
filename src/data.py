"""Loading MM-IMDb, data-quality checks and leakage-free de-duplication."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GENRES


def find_root(data_dir):
    """Return the folder containing split.json (the archive sometimes nests one level deeper)."""
    data_dir = Path(data_dir)
    for cand in [data_dir, data_dir / "mmimdb", *data_dir.glob("*/")]:
        if (cand / "split.json").exists():
            return cand
    raise FileNotFoundError(f"split.json not found under {data_dir}")


def image_path(root, movie_id):
    for ext in (".jpeg", ".jpg", ".png"):
        p = Path(root) / "dataset" / f"{movie_id}{ext}"
        if p.exists():
            return p
    return None


def load_mmimdb(data_dir, genres=GENRES):
    """One row per movie: id, split, plot, genre list, metadata. Uses the official split.json."""
    root = find_root(data_dir)
    split = json.loads((root / "split.json").read_text())
    split_of = {}
    for name, ids in split.items():
        name = {"dev": "val"}.get(name, name)
        for i in ids:
            split_of[str(i)] = name

    rows, problems = [], {"missing_json": 0, "missing_image": 0, "empty_plot": 0, "no_valid_genre": 0}
    for movie_id, sp in split_of.items():
        jp = root / "dataset" / f"{movie_id}.json"
        if not jp.exists():
            problems["missing_json"] += 1
            continue
        meta = json.loads(jp.read_text())
        img = image_path(root, movie_id)
        if img is None:
            problems["missing_image"] += 1
            continue
        plots = meta.get("plot") or []
        plot = plots[0].strip() if plots and isinstance(plots[0], str) else ""
        if not plot:
            problems["empty_plot"] += 1
            continue
        g = [x for x in (meta.get("genres") or []) if x in genres]
        if not g:
            problems["no_valid_genre"] += 1
            continue
        year = meta.get("year")
        rows.append({
            "id": movie_id, "split": sp, "image": str(img), "plot": plot, "genres": g,
            "title": meta.get("title", ""),
            "year": int(year) if isinstance(year, (int, float)) or (isinstance(year, str) and year.isdigit()) else np.nan,
            "countries": meta.get("countries") or [], "languages": meta.get("languages") or [],
            "n_plots": len(plots),
        })
    df = pd.DataFrame(rows).sort_values(["split", "id"]).reset_index(drop=True)
    return df, problems


def labels_matrix(df, genres=GENRES):
    Y = np.zeros((len(df), len(genres)), dtype=np.float32)
    idx = {g: k for k, g in enumerate(genres)}
    for r, gs in enumerate(df["genres"]):
        for g in gs:
            Y[r, idx[g]] = 1
    return Y


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def deduplicate(df):
    """Remove duplicated plots. Train copies are kept; a val/test copy of a train plot is
    dropped (it would leak the answer), and repeats within a split keep one copy."""
    order = {"train": 0, "val": 1, "test": 2}
    d = df.assign(_key=df["plot"].map(_norm), _o=df["split"].map(order)).sort_values(["_o", "id"])
    dup_mask = d.duplicated("_key", keep="first")
    removed = d[dup_mask]
    n_groups = d[d["_key"].duplicated(keep=False)]["_key"].nunique()
    report = {
        "duplicate_plot_groups": int(n_groups),
        "rows_removed": int(dup_mask.sum()),
        "removed_by_split": removed["split"].value_counts().to_dict(),
    }
    kept = d[~dup_mask].drop(columns=["_key", "_o"]).sort_values(["split", "id"]).reset_index(drop=True)
    return kept, report


def quality_summary(df, genres=GENRES):
    """Numbers for the data-quality table of the report."""
    Y = labels_matrix(df, genres)
    words = df["plot"].str.split().str.len()
    tr = df["split"] == "train"
    freq = Y[tr.values].sum(0)
    return {
        "movies": len(df),
        **{f"{s}_movies": int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "labels_per_movie_mean": float(Y.sum(1).mean()),
        "labels_per_movie_max": int(Y.sum(1).max()),
        "plot_words_mean": float(words.mean()),
        "plot_words_median": float(words.median()),
        "plot_words_p95": float(words.quantile(0.95)),
        "plot_words_min": int(words.min()),
        "most_common_genre": genres[int(freq.argmax())],
        "rarest_genre": genres[int(freq.argmin())],
        "imbalance_ratio_train": float(freq.max() / max(freq.min(), 1)),
        "share_us": float(df["countries"].map(lambda c: "USA" in c).mean()),
        "share_english": float(df["languages"].map(lambda l: "English" in l).mean()),
        "share_pre_1990": float((df["year"] < 1990).mean()),
    }
