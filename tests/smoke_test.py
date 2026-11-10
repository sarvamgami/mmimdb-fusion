"""Offline end-to-end check: fake data + random cached features -> executes notebooks/run_all.ipynb."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import deduplicate, load_mmimdb  # noqa: E402

subprocess.run([sys.executable, str(ROOT / "tests/make_fake_mmimdb.py"), str(ROOT / "data/mmimdb"), "300"], check=True)
df, _ = deduplicate(load_mmimdb(ROOT / "data/mmimdb")[0])
(ROOT / "cache").mkdir(exist_ok=True)
rng = np.random.default_rng(0)
n = len(df)
torch.save({"ids": list(df["id"]), "feats": torch.randn(n, 1280),
            "image_info": pd.DataFrame({"width": rng.integers(180, 300, n), "height": rng.integers(260, 450, n),
                                        "non_rgb": rng.integers(0, 2, n), "readable": np.ones(n, int)}),
            "seconds": 1.0, "peak_gpu_gb": 0.0}, ROOT / "cache/img_effb0.pt")
torch.save({"ids": list(df["id"]), "feats": torch.randn(n, 768), "cls": torch.randn(n, 768),
            "n_tokens": rng.integers(30, 600, n), "truncated_share": 0.01, "seconds": 1.0}, ROOT / "cache/txt_bert.pt")

subprocess.run([sys.executable, str(ROOT / "notebooks/build_notebook.py")], check=True)
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=1200",
                str(ROOT / "notebooks/run_all.ipynb"), "--output", "smoke_executed.ipynb"], check=True, cwd=ROOT / "notebooks")
print("SMOKE TEST PASSED")
