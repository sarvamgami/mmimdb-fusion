"""All report figures. Every figure has a title, labelled axes and (where needed) a legend."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PALETTE = {
    "Image only": "#9aa5b1", "Text only": "#5b6b7a", "Early fusion": "#e69f00",
    "Late fusion": "#56b4e9", "Intermediate (concat)": "#009e73",
    "Modality attention": "#cc79a7", "GMU": "#d55e00",
}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.show()


def genre_distribution(genres, train_counts, path):
    order = np.argsort(train_counts)[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar([genres[i] for i in order], train_counts[order], color="#5b6b7a")
    ax.set_yscale("log")
    ax.set_ylabel("Training movies (log scale)")
    ax.set_xlabel("Genre")
    ax.set_title("Genre frequency in the training split (strong class imbalance)")
    ax.tick_params(axis="x", rotation=60)
    for k, i in enumerate(order):
        ax.text(k, train_counts[i] * 1.08, str(int(train_counts[i])), ha="center", fontsize=7)
    _save(fig, path)


def data_quality(labels_per_movie, plot_words, n_tokens, max_tokens, aspect, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].hist(labels_per_movie, bins=np.arange(0.5, labels_per_movie.max() + 1.5), color="#5b6b7a")
    ax[0].set(title="(a) Genres per movie", xlabel="Number of genres", ylabel="Movies")
    ax[1].hist(np.clip(n_tokens, 0, 800), bins=60, color="#009e73")
    ax[1].axvline(max_tokens, color="#d55e00", ls="--", label=f"BERT limit ({max_tokens} tokens)")
    ax[1].set(title="(b) Plot length", xlabel="BERT tokens per plot (clipped at 800)", ylabel="Movies")
    ax[1].legend()
    ax[2].hist(np.clip(aspect, 0, 3), bins=60, color="#e69f00")
    ax[2].axvline(1.0, color="k", ls=":", label="square (model input)")
    ax[2].set(title="(c) Poster shape", xlabel="Height / width ratio", ylabel="Posters")
    ax[2].legend()
    _save(fig, path)


def cooccurrence(Y, genres, path):
    C = Y.T @ Y
    C = C / np.diag(C)[:, None]           # P(column genre | row genre)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(genres)), genres, rotation=90, fontsize=8)
    ax.set_yticks(range(len(genres)), genres, fontsize=8)
    ax.set(title="Genre co-occurrence: P(column genre | row genre), training split",
           xlabel="Co-occurring genre", ylabel="Given genre")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Conditional probability")
    _save(fig, path)


def results_bars(mean, std, metrics_, path, title):
    names = list(mean.index)
    x = np.arange(len(metrics_))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, n in enumerate(names):
        ax.bar(x + k * w - 0.4 + w / 2, mean.loc[n, metrics_], w, yerr=std.loc[n, metrics_],
               capsize=2, label=n, color=PALETTE.get(n, None))
    ax.set_xticks(x, [m.replace("test_", "").replace("_", " ") for m in metrics_])
    ax.set_ylabel("Score on test set (mean ± std, 3 seeds)")
    ax.set_xlabel("Metric")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    _save(fig, path)


def learning_curves(hists, path):
    names = list(hists)
    fig, axes = plt.subplots(2, len(names), figsize=(3 * len(names), 5.2), sharey="row")
    for k, n in enumerate(names):
        h = hists[n]
        e = np.arange(1, len(h["train_loss"]) + 1)
        best = int(np.argmin(h["val_loss"])) + 1
        axes[0, k].plot(e, h["train_loss"], label="train")
        axes[0, k].plot(e, h["val_loss"], label="validation")
        axes[0, k].axvline(best, color="grey", ls=":", label="early stop")
        axes[0, k].set_title(n, fontsize=9)
        axes[1, k].plot(e, h["train_f1"], label="train")
        axes[1, k].plot(e, h["val_f1"], label="validation")
        axes[1, k].axvline(best, color="grey", ls=":")
        axes[1, k].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Weighted BCE loss")
    axes[1, 0].set_ylabel("Micro-F1 (threshold 0.5)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Training vs validation curves (seed 0); dotted line = epoch kept by early stopping")
    _save(fig, path)


def overfit_gap(mean, path):
    names = list(mean.index)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.2, mean["train_micro_F1"], 0.4, label="train", color="#9aa5b1")
    ax.bar(x + 0.2, mean["test_micro_F1"], 0.4, label="test", color="#d55e00")
    for i, n in enumerate(names):
        gap = mean.loc[n, "train_micro_F1"] - mean.loc[n, "test_micro_F1"]
        ax.text(i, max(mean.loc[n, "train_micro_F1"], mean.loc[n, "test_micro_F1"]) + 0.02,
                f"gap {gap:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=20)
    ax.set(ylabel="Micro-F1 (tuned thresholds)", xlabel="Model", ylim=(0, 1.05),
           title="Generalisation gap: train vs test micro-F1 (mean of 3 seeds)")
    ax.legend()
    _save(fig, path)


def per_genre(tables, path):
    """tables: {model: per_genre_f1 DataFrame}; genres sorted by training frequency."""
    base = next(iter(tables.values()))
    genres = list(base["genre"])
    x = np.arange(len(genres))
    w = 0.8 / len(tables)
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, (n, t) in enumerate(tables.items()):
        t = t.set_index("genre").loc[genres]
        ax.bar(x + k * w - 0.4 + w / 2, t["F1"], w, label=n, color=PALETTE.get(n))
    ax.set_xticks(x, [f"{g}\n(n={c})" for g, c in zip(genres, base["train_count"])], rotation=90, fontsize=7)
    ax.set(ylabel="Test F1", xlabel="Genre (training count), most to least frequent",
           title="Per-genre test F1: rare genres are hardest; fusion helps most where one modality is weak", ylim=(0, 1))
    ax.legend(fontsize=8)
    _save(fig, path)


def modality_share(genres, share, path, title, ylabel):
    order = np.argsort(share)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar([genres[i] for i in order], share[order], color=["#d55e00" if share[i] > 0.5 else "#5b6b7a" for i in order])
    ax.axhline(0.5, color="k", ls=":", label="equal weight")
    ax.set(ylabel=ylabel, xlabel="Genre", title=title)
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    _save(fig, path)


def efficiency(mean, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in mean.index:
        ax.scatter(mean.loc[n, "params"] / 1e6, mean.loc[n, "test_macro_F1"], s=60, color=PALETTE.get(n))
        ax.annotate(n, (mean.loc[n, "params"] / 1e6, mean.loc[n, "test_macro_F1"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set(xlabel="Trainable parameters (millions)", ylabel="Test macro-F1",
           title="Effectiveness vs model size")
    _save(fig, path)


def failures(df_test, Y, P, thr, genres, path, n=6, title="Failure cases"):
    """The test movies where the model is most wrong (lowest per-movie F1, then most confident)."""
    pred = P >= thr
    tp = (pred & (Y > 0)).sum(1)
    f1 = 2 * tp / (pred.sum(1) + Y.sum(1) + 1e-9)
    worst = np.lexsort((-P.max(1), f1))[:n]
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 5.4))
    for ax, i in zip(axes, worst):
        r = df_test.iloc[i]
        try:
            ax.imshow(Image.open(r["image"]).convert("RGB"))
        except Exception:
            pass
        ax.axis("off")
        true = ", ".join(g for g, y in zip(genres, Y[i]) if y)
        top = np.argsort(P[i])[::-1][:3]
        predicted = ", ".join(f"{genres[c]} {P[i, c]:.2f}" for c in top)
        ax.set_title(textwrap.fill(str(r["title"])[:40], 22), fontsize=8)
        ax.text(0.5, -0.04, textwrap.fill("True: " + true, 30) + "\n" + textwrap.fill("Top-3: " + predicted, 30),
                transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.suptitle(title)
    _save(fig, path)
    return worst


def subgroups(table, path, title):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["micro_F1"], 0.4, label="micro-F1", color="#5b6b7a")
    ax.bar(x + 0.2, table["macro_F1"], 0.4, label="macro-F1", color="#e69f00")
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(table["group"], table["n"])], fontsize=8)
    ax.set(ylabel="Test score", xlabel="Subgroup", title=title, ylim=(0, 1))
    ax.legend()
    _save(fig, path)


def ablation_bars(summary_mean, summary_std, path, title):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(summary_mean))
    ax.bar(x - 0.2, summary_mean["test_micro_F1"], 0.4, yerr=summary_std["test_micro_F1"], capsize=3, label="micro-F1", color="#5b6b7a")
    ax.bar(x + 0.2, summary_mean["test_macro_F1"], 0.4, yerr=summary_std["test_macro_F1"], capsize=3, label="macro-F1", color="#e69f00")
    ax.set_xticks(x, [textwrap.fill(s, 18) for s in summary_mean.index], fontsize=8)
    ax.set(ylabel="Test score (mean ± std, 3 seeds)", xlabel="Variant", title=title, ylim=(0, 1))
    ax.legend()
    _save(fig, path)
