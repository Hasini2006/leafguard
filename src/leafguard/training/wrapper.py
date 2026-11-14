"""The MLflow pyfunc model that the model server runs.

It is logged as an MLflow "model from code": the registry stage passes this file's path to
log_model, MLflow stores the source itself, and the model server executes it on load. No
pickled class crosses from the training environment to the serving container, so the two
can run different Python versions safely.

That is also why the file must stay standalone: no imports from the leafguard package,
only torch, numpy, pandas and Pillow. Resizing and normalisation
live here rather than in the backend, so the exact transform used during training travels
with every model version and serving can never drift away from training.

Input : a DataFrame with one column, image_b64, holding base64-encoded image bytes.
Output: a DataFrame with one column per class, holding the softmax scores.
"""

from __future__ import annotations

import base64
import io
import json

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LeafModel(mlflow.pyfunc.PythonModel):
    """Decode, resize, normalise and classify one image at a time."""

    def load_context(self, context):
        meta = json.loads(open(context.artifacts["classes"], encoding="utf-8").read())
        self.classes = meta["classes"]
        self.image_size = int(meta["image_size"])
        self.net = torch.jit.load(context.artifacts["weights"], map_location="cpu")
        self.net.eval()
        torch.set_num_threads(1)  # one request at a time keeps latency steady

    def prepare(self, raw: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image = ImageOps.fit(image, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, context, model_input, params=None):
        scores = []
        for encoded in model_input["image_b64"]:
            tensor = self.prepare(base64.b64decode(encoded))
            with torch.no_grad():
                probabilities = torch.softmax(self.net(tensor), dim=1)[0].numpy()
            scores.append([round(float(value), 6) for value in probabilities])
        return pd.DataFrame(scores, columns=self.classes)


# Tells MLflow which object in this file is the model when it is loaded from code.
mlflow.models.set_model(LeafModel())
