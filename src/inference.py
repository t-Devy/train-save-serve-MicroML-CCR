import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from src.constants import MODEL_PATH, META_PATH
from src.model import CCRNet


@dataclass
class Prediction:
    probability: float
    prediction: int
    model_version: str


class CCRPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
        # 1) Load meta (features, scalar, threshold)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feature_columns = self.meta['feature_columns']
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.model_version = self.meta.get("model_version", "0.1.0")

        # scaler params saved in training
        self.mean = np.array(self.meta['scaler_mean'], dtype=np.float32)
        self.scale = np.array(self.meta['scaler_scale'], dtype=np.float32)

        # 2) Load model weights
        self.model = CCRNet(input_dim=len(self.feature_columns), hidden_dim=16)
        state = torch.load(str(model_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def _vectorize(self, features: Dict[str, Any]) -> np.ndarray:
        # Enforce feature order (prevent wrong column bugs)
        x = np.array([features[col] for col in self.feature_columns], dtype=np.float32)
        # apply same scaling from training
        x = (x - self.mean) / self.scale
        return x

    def predict(self, features: Dict[str, Any]) -> Prediction:
        x = self._vectorize(features)
        x_t = torch.from_numpy(x).unsqueeze(0)      #shape (1, 11)

        with torch.no_grad():
            logits = self.model(x_t)        # shape (1, 1)
            prob = torch.sigmoid(logits).item()     # -> float in (0, 1)

        pred = 1 if prob >= self.threshold else 0
        return Prediction(probability=prob, prediction=pred, model_version=self.model_version)



