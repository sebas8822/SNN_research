import json
from pathlib import Path
from typing import Dict, List, Optional
from . import config

class CocoLoader:
    def __init__(self, train_json_path: Path = config.TRAIN_JSON):
        self.train_json_path = train_json_path
        self.data = self._load_json(train_json_path)
        self.categories_map: Dict[int, str] = {c["id"]: c["name"] for c in self.data["categories"]}
        self.id2file_train: Dict[int, str] = {im["id"]: im["file_name"] for im in self.data["images"]}
        self.annotations = self.data["annotations"]

    def _load_json(self, path: Path):
        with open(path, "r") as f:
            return json.load(f)

    def resolve_image_path(self, file_name: str) -> Optional[Path]:
        p = Path(file_name)
        # Candidates for image location
        cands = [
            p,
            config.DATA_ROOT / p,
            config.TRAIN_DIR / p.name,
            config.TRAIN_DIR / p
        ]
        for c in cands:
            if c.exists():
                return c
        return None

    def get_annotations(self):
        return self.annotations

    def get_category_name(self, category_id: int) -> str:
        return self.categories_map.get(category_id, "Unknown")

    def get_filename(self, image_id: int) -> Optional[str]:
        return self.id2file_train.get(image_id)
