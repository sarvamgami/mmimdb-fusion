"""Central configuration: every path and hyperparameter used in the project lives here."""
from dataclasses import dataclass, field
from pathlib import Path

# The 23 genres used by Arevalo et al. (2017); rarer IMDb tags (News, Adult, ...) are dropped.
GENRES = [
    "Drama", "Comedy", "Romance", "Thriller", "Crime", "Action", "Adventure", "Horror",
    "Documentary", "Mystery", "Sci-Fi", "Fantasy", "Family", "Biography", "War", "History",
    "Music", "Animation", "Musical", "Western", "Sport", "Short", "Film-Noir",
]


@dataclass
class Config:
    # ---- paths ----
    data_dir: Path = Path("/content/mmimdb")           # folder that contains split.json and dataset/
    cache_dir: Path = Path("/content/drive/MyDrive/mmimdb_cache")  # cached features (survive disconnects)
    results_dir: Path = Path("results")

    # ---- feature extraction ----
    image_model: str = "efficientnet_b0"   # torchvision, ImageNet-1k weights, 1280-d pooled output
    image_size: int = 224
    text_model: str = "bert-base-uncased"  # 768-d, mean-pooled over non-padding tokens (+ [CLS] kept for ablation)
    max_tokens: int = 512
    extract_batch: int = 64
    num_workers: int = 2

    # ---- training (shared by all trainable models) ----
    hidden: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10              # early stopping on validation loss (Week 7)
    use_pos_weight: bool = True     # class-weighted BCE for the imbalanced genres
    seeds: tuple = (0, 1, 2)        # every result is mean +/- std over these seeds (Week 7)

    genres: list = field(default_factory=lambda: list(GENRES))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.results_dir = Path(self.results_dir)
        (self.results_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "tables").mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
