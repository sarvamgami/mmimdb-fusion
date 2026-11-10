"""Multi-label metrics, threshold tuning and late fusion."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, precision_score, recall_score

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def metrics(Y, P, thr=0.5):
    """Y: (N, C) 0/1, P: (N, C) probabilities, thr: scalar or (C,) per-genre thresholds."""
    pred = (P >= np.asarray(thr)).astype(int)
    present = Y.sum(0) > 0                    # mAP is undefined for genres absent from a split
    return {
        "micro_F1": f1_score(Y, pred, average="micro", zero_division=0),
        "macro_F1": f1_score(Y, pred, average="macro", zero_division=0),
        "weighted_F1": f1_score(Y, pred, average="weighted", zero_division=0),
        "samples_F1": f1_score(Y, pred, average="samples", zero_division=0),
        "micro_P": precision_score(Y, pred, average="micro", zero_division=0),
        "micro_R": recall_score(Y, pred, average="micro", zero_division=0),
        "mAP": average_precision_score(Y[:, present], P[:, present], average="macro"),
        "hamming": hamming_loss(Y, pred),
    }


def tune_thresholds(Yv, Pv):
    """One threshold per genre, chosen to maximise that genre's F1 on the VALIDATION set."""
    thr = np.full(Yv.shape[1], 0.5)
    for c in range(Yv.shape[1]):
        if Yv[:, c].sum() == 0:
            continue
        scores = [f1_score(Yv[:, c], Pv[:, c] >= t, zero_division=0) for t in GRID]
        thr[c] = GRID[int(np.argmax(scores))]
    return thr


def late_fusion_weights(Yv, Pv_img, Pv_txt):
    """Per-genre weight w_c for p = w_c * p_img + (1 - w_c) * p_txt, chosen on validation AP."""
    W = np.round(np.arange(0, 1.01, 0.05), 2)
    w = np.full(Yv.shape[1], 0.5)
    for c in range(Yv.shape[1]):
        if Yv[:, c].sum() == 0:
            continue
        aps = [average_precision_score(Yv[:, c], a * Pv_img[:, c] + (1 - a) * Pv_txt[:, c]) for a in W]
        w[c] = W[int(np.argmax(aps))]
    return w


def per_genre_f1(Y, P, thr, genres, train_counts):
    pred = (P >= np.asarray(thr)).astype(int)
    return pd.DataFrame({
        "genre": genres,
        "train_count": train_counts.astype(int),
        "test_count": Y.sum(0).astype(int),
        "F1": f1_score(Y, pred, average=None, zero_division=0),
        "AP": [average_precision_score(Y[:, c], P[:, c]) if Y[:, c].sum() else np.nan for c in range(Y.shape[1])],
    }).sort_values("train_count", ascending=False)


def subgroup_table(df_test, Y, P, thr, groups):
    """groups: {name: boolean mask over test rows}. Returns metrics per subgroup."""
    rows = []
    for name, mask in groups.items():
        mask = np.asarray(mask)
        if mask.sum() < 20:
            continue
        m = metrics(Y[mask], P[mask], thr)
        rows.append({"group": name, "n": int(mask.sum()), **{k: m[k] for k in ("micro_F1", "macro_F1", "samples_F1", "mAP")}})
    return pd.DataFrame(rows)


def mean_std(rows, keys):
    """rows: list of metric dicts (one per seed) -> 'mean ± std' strings and raw means."""
    df = pd.DataFrame(rows)
    return {k: (df[k].mean(), df[k].std(ddof=0)) for k in keys}
