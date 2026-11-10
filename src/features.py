"""Frozen feature extraction (run once, cached to disk), as in the Week 4/6 practicals.

Images: EfficientNet-B0 (Tan & Le, 2019), ImageNet weights, global-average-pooled 1280-d vector.
Text:   BERT-base-uncased (Devlin et al., 2019), mean over non-padding tokens (768-d); [CLS] also kept.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PosterDataset(Dataset):
    def __init__(self, paths, size=224):
        from torchvision import transforms as T
        self.paths = list(paths)
        # Posters are ~2:3 portrait. We resize the whole poster to size x size instead of
        # centre-cropping so the title text and top/bottom artwork are not cut away.
        self.tf = T.Compose([
            T.Resize((size, size), antialias=True),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet statistics
        ])
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i])
            w, h, mode = im.width, im.height, im.mode
            im = im.convert("RGB")          # greyscale / CMYK / palette posters -> 3 channels
            return self.tf(im), torch.tensor([w, h, int(mode != "RGB"), 1])
        except Exception:
            return torch.zeros(3, self.size, self.size), torch.tensor([0, 0, 0, 0])


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_image_features(df, cfg, out_name="img_effb0.pt"):
    out = Path(cfg.cache_dir) / out_name
    if out.exists():
        print(f"[cache] {out}")
        return torch.load(out, weights_only=False)
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    dev = _device()
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()          # keep the 1280-d pooled embedding
    net.eval().to(dev)
    loader = DataLoader(PosterDataset(df["image"], cfg.image_size), batch_size=cfg.extract_batch,
                        num_workers=cfg.num_workers, pin_memory=dev.type == "cuda")
    feats, info, t0 = [], [], time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for k, (x, meta) in enumerate(loader):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            f = net(x.to(dev, non_blocking=True))
        feats.append(f.float().cpu())
        info.append(meta)
        if k % 50 == 0:
            print(f"  images {k * cfg.extract_batch:>6}/{len(df)}  {time.time() - t0:5.0f}s")
    info = torch.cat(info).numpy()
    res = {
        "ids": list(df["id"]), "feats": torch.cat(feats),
        "image_info": pd.DataFrame(info, columns=["width", "height", "non_rgb", "readable"]),
        "seconds": time.time() - t0,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0,
    }
    torch.save(res, out)
    print(f"Image features {tuple(res['feats'].shape)} in {res['seconds']:.0f}s -> {out}")
    return res


@torch.no_grad()
def extract_text_features(df, cfg, out_name="txt_bert.pt"):
    out = Path(cfg.cache_dir) / out_name
    if out.exists():
        print(f"[cache] {out}")
        return torch.load(out, weights_only=False)
    from transformers import AutoModel, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(cfg.text_model)
    bert = AutoModel.from_pretrained(cfg.text_model).eval().to(dev)
    texts = list(df["plot"])
    n_tokens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
    order = np.argsort(n_tokens)                  # sort by length -> little padding, much faster
    mean_f = torch.zeros(len(texts), bert.config.hidden_size)
    cls_f = torch.zeros(len(texts), bert.config.hidden_size)
    t0 = time.time()
    for s in range(0, len(texts), cfg.extract_batch):
        idx = order[s:s + cfg.extract_batch]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=cfg.max_tokens, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            h = bert(**enc).last_hidden_state.float()          # (B, T, 768)
        m = enc["attention_mask"].unsqueeze(-1).float()
        mean_f[idx] = ((h * m).sum(1) / m.sum(1)).cpu()         # mean over real tokens only
        cls_f[idx] = h[:, 0].cpu()
        if (s // cfg.extract_batch) % 50 == 0:
            print(f"  plots {s:>6}/{len(texts)}  {time.time() - t0:5.0f}s")
    res = {"ids": list(df["id"]), "feats": mean_f, "cls": cls_f, "n_tokens": n_tokens,
           "truncated_share": float((n_tokens > cfg.max_tokens).mean()), "seconds": time.time() - t0}
    torch.save(res, out)
    print(f"Text features {tuple(mean_f.shape)} in {res['seconds']:.0f}s -> {out}")
    return res
