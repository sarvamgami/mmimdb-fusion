"""Training loop with class-weighted BCE and early stopping (Weeks 6-7)."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardise(X, train_mask):
    """z-score each feature with TRAINING statistics only, so held-out data never leaks in."""
    mu = X[train_mask].mean(0, keepdims=True)
    sd = X[train_mask].std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def pos_weights(Y_train):
    """sqrt(neg/pos) per genre: up-weights rare genres without letting Film-Noir dominate."""
    pos = Y_train.sum(0).clip(min=1)
    return torch.tensor(np.sqrt((len(Y_train) - pos) / pos), dtype=torch.float32)


@torch.no_grad()
def predict(model, img, txt, batch=4096):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(img), batch):
        out.append(torch.sigmoid(model(img[s:s + batch].to(dev), txt[s:s + batch].to(dev))).cpu())
    return torch.cat(out).numpy()


def train_model(model, data, cfg, seed, pos_weight=None, verbose=False):
    """data = dict(train=(img, txt, Y), val=(img, txt, Y)) of float tensors.
    Returns the weights of the epoch with the lowest validation loss, plus the history."""
    set_seed(seed)
    dev = device()
    model = model.to(dev)
    Xi, Xt, Y = (t.to(dev) for t in data["train"])
    Vi, Vt, VY = (t.to(dev) for t in data["val"])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev) if pos_weight is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)

    hist = {k: [] for k in ("train_loss", "val_loss", "train_f1", "val_f1")}
    best, best_state, best_epoch, wait = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(len(Y), generator=g).to(dev)
        total = 0.0
        for s in range(0, len(Y), cfg.batch_size):
            b = order[s:s + cfg.batch_size]
            opt.zero_grad()
            loss = crit(model(Xi[b], Xt[b]), Y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            tr_logits = model(Xi, Xt)
            va_logits = model(Vi, Vt)
            val_loss = crit(va_logits, VY).item()
        hist["train_loss"].append(total / len(Y))
        hist["val_loss"].append(val_loss)
        hist["train_f1"].append(f1_score(Y.cpu().numpy(), (tr_logits > 0).cpu().numpy(), average="micro", zero_division=0))
        hist["val_f1"].append(f1_score(VY.cpu().numpy(), (va_logits > 0).cpu().numpy(), average="micro", zero_division=0))
        if val_loss < best - 1e-4:
            best, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d} train {hist['train_loss'][-1]:.4f} val {val_loss:.4f} val-F1 {hist['val_f1'][-1]:.3f}")
    model.load_state_dict(best_state)
    return model, hist, best_epoch, time.time() - t0
