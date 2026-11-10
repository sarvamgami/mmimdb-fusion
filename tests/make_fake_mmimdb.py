"""Creates a tiny fake MM-IMDb (same folder layout) so the pipeline can be smoke-tested offline."""
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GENRES  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fake_mmimdb")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
(out / "dataset").mkdir(parents=True, exist_ok=True)
random.seed(0)
ids = [f"{i:07d}" for i in range(n)]
words = "a man woman city love war killer ship space family young detective music journey".split()
for k, i in enumerate(ids):
    g = random.sample(GENRES[:8], random.randint(1, 3)) + (["Film-Noir"] if k % 40 == 0 else []) + (["News"] if k % 7 == 0 else [])
    plot = " ".join(random.choices(words, k=random.randint(20, 120)))
    if k in (5, 205):
        plot = "an identical duplicated plot used to test leakage removal"
    meta = {"title": f"Movie {k}", "year": random.randint(1930, 2015), "genres": g, "plot": [plot],
            "countries": [random.choice(["USA", "USA", "UK", "France"])], "languages": [random.choice(["English", "English", "French"])]}
    (out / "dataset" / f"{i}.json").write_text(json.dumps(meta))
    mode = "L" if k % 25 == 0 else "RGB"
    Image.new(mode, (random.randint(180, 300), random.randint(260, 450)), color=(k * 3 % 255) if mode == "L" else (k % 255, 80, 120)).save(out / "dataset" / f"{i}.jpeg")
json.dump({"train": ids[:200], "dev": ids[200:240], "test": ids[240:]}, open(out / "split.json", "w"))
print("fake dataset at", out)
