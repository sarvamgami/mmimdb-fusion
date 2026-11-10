"""Runs every model over several seeds and collects metrics, predictions and histories."""
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .evaluate import late_fusion_weights, metrics, tune_thresholds
from .models import MODELS, n_params
from .train import pos_weights, predict, train_model

KEYS = ["micro_F1", "macro_F1", "weighted_F1", "samples_F1", "mAP", "hamming", "micro_P", "micro_R"]


def split_tensors(img, txt, Y, split):
    out = {}
    for s in ("train", "val", "test"):
        m = (split == s)
        out[s] = (torch.tensor(img[m]), torch.tensor(txt[m]), torch.tensor(Y[m]))
    return out


def time_inference(model, img, txt, repeats=3):
    """Milliseconds to score 1,000 test movies (after one warm-up)."""
    x_i, x_t = img[:1000], txt[:1000]
    predict(model, x_i, x_t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        predict(model, x_i, x_t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000


def run_all(img, txt, Y, split, cfg, model_names=None, seeds=None, tag="main", verbose=True):
    """Trains each model with each seed. Returns (per-run rows, artefacts for seed 0)."""
    data = split_tensors(img, txt, Y, split)
    Ytr, Yv, Yte = (data[s][2].numpy() for s in ("train", "val", "test"))
    pw = pos_weights(Ytr) if cfg.use_pos_weight else None
    seeds = seeds or cfg.seeds
    model_names = model_names or list(MODELS)
    rows, art = [], {"probs": {}, "hist": {}, "models": {}, "thr": {}, "train_probs": {}}

    for seed in seeds:
        probs_seed = {}
        for name in model_names:
            model = MODELS[name](img.shape[1], txt.shape[1], Y.shape[1], cfg)
            model, hist, best_epoch, secs = train_model(model, data, cfg, seed, pw)
            P = {s: predict(model, data[s][0], data[s][1]) for s in ("train", "val", "test")}
            probs_seed[name] = P
            thr = tune_thresholds(Yv, P["val"])
            row = {"tag": tag, "model": name, "seed": seed, "best_epoch": best_epoch,
                   "train_seconds": secs, "params": n_params(model),
                   "infer_ms_per_1k": time_inference(model, data["test"][0], data["test"][1])}
            for split_name, Ys in (("train", Ytr), ("val", Yv), ("test", Yte)):
                for k, v in metrics(Ys, P[split_name], 0.5).items():
                    row[f"{split_name}_{k}@0.5"] = v
                for k, v in metrics(Ys, P[split_name], thr).items():
                    row[f"{split_name}_{k}"] = v
            rows.append(row)
            if seed == seeds[0]:
                art["probs"][name], art["hist"][name] = P["test"], hist
                art["train_probs"][name] = P["train"]
                art["models"][name], art["thr"][name] = model, thr
            if verbose:
                print(f"  [{tag}] seed {seed} {name:<22} epochs {best_epoch + 1:3d}  "
                      f"test micro-F1 {row['test_micro_F1']:.3f}  macro-F1 {row['test_macro_F1']:.3f}  ({secs:.0f}s)")

        # ---- Late fusion: combine the two unimodal models of this seed ----
        if {"Image only", "Text only"} <= set(probs_seed):
            pi, pt = probs_seed["Image only"], probs_seed["Text only"]
            w = late_fusion_weights(Yv, pi["val"], pt["val"])
            P = {s: w * pi[s] + (1 - w) * pt[s] for s in ("train", "val", "test")}
            thr = tune_thresholds(Yv, P["val"])
            pr = [r for r in rows if r["seed"] == seed and r["model"] in ("Image only", "Text only")]
            row = {"tag": tag, "model": "Late fusion", "seed": seed, "best_epoch": np.nan,
                   "train_seconds": sum(r["train_seconds"] for r in pr),
                   "params": sum(r["params"] for r in pr),
                   "infer_ms_per_1k": sum(r["infer_ms_per_1k"] for r in pr)}
            for split_name, Ys in (("train", Ytr), ("val", Yv), ("test", Yte)):
                for k, v in metrics(Ys, P[split_name], 0.5).items():
                    row[f"{split_name}_{k}@0.5"] = v
                for k, v in metrics(Ys, P[split_name], thr).items():
                    row[f"{split_name}_{k}"] = v
            rows.append(row)
            if seed == seeds[0]:
                art["probs"]["Late fusion"], art["thr"]["Late fusion"] = P["test"], thr
                art["train_probs"]["Late fusion"] = P["train"]
                art["late_weights"] = w
            if verbose:
                print(f"  [{tag}] seed {seed} {'Late fusion':<22}               "
                      f"test micro-F1 {row['test_micro_F1']:.3f}  macro-F1 {row['test_macro_F1']:.3f}")
    return pd.DataFrame(rows), art


def summarise(runs, cols, order=None):
    """mean ± std over seeds, one row per model."""
    g = runs.groupby("model")[cols]
    mean, std = g.mean(), g.std(ddof=0).fillna(0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    if order:
        out = out.loc[[m for m in order if m in out.index]]
        mean = mean.loc[out.index]
        std = std.loc[out.index]
    return out, mean, std


def run_ablation(img, txt, Y, split, cfg, model_name, variants, seeds=None):
    """variants: {label: dict(cfg_overrides=..., txt=optional alternative text features)}."""
    rows = []
    for label, v in variants.items():
        c = replace(cfg, **v.get("cfg", {}))
        r, _ = run_all(img, v.get("txt", txt), Y, split, c, [model_name], seeds, tag=label, verbose=False)
        rows.append(r)
        print(f"  ablation {label:<36} test micro-F1 {r['test_micro_F1'].mean():.3f}  "
              f"macro-F1 {r['test_macro_F1'].mean():.3f}  train-test gap "
              f"{(r['train_micro_F1'] - r['test_micro_F1']).mean():.3f}")
    return pd.concat(rows, ignore_index=True)
