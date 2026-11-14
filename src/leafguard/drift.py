"""Reference distributions and the Population Stability Index (PSI).

The preprocess stage stores quantile bins for each image feature of the training set.
The backend later compares the images people actually upload against those bins.

Reading PSI: on large samples, below 0.1 the inputs look like the training data and above
0.2 is a clear shift. The backend scores a small rolling window (50 uploads), where random
sampling alone gives PSI of about 0.25 and rarely above 0.7, so the alert fires at 1.0;
the darkened, blurred demo traffic scores above 3. A shift is a warning that the model may
now be seeing different data; on its own it does not prove the model became less accurate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FLOOR = 1e-6


def build_reference(values: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Turn training values into quantile bin edges plus the share falling in each bin."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("cannot build a reference from an empty set of values")
    cuts = np.quantile(clean, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(cuts)
    counts = np.bincount(np.digitize(clean, edges), minlength=edges.size + 1)
    return {
        "edges": [round(float(e), 6) for e in edges],
        "shares": [round(float(c), 6) for c in counts / counts.sum()],
        "mean": round(float(clean.mean()), 6),
        "std": round(float(clean.std()), 6),
        "count": int(clean.size),
    }


def psi(reference: dict[str, Any], values: np.ndarray) -> float:
    """Population Stability Index between a stored reference and a batch of new values."""
    current = np.asarray(values, dtype=float)
    current = current[np.isfinite(current)]
    if current.size == 0:
        return 0.0
    edges = np.asarray(reference["edges"], dtype=float)
    expected = np.asarray(reference["shares"], dtype=float)
    counts = np.bincount(np.digitize(current, edges), minlength=edges.size + 1)
    # Half a count per bin (Jeffreys smoothing): with a small window an empty bin is likely
    # by chance, and treating it as a share of ~0 would add about 1.1 to the score alone.
    seen = (counts + 0.5) / (counts.sum() + 0.5 * counts.size)
    expected = np.clip(expected, FLOOR, None)
    return round(float(np.sum((seen - expected) * np.log(seen / expected))), 6)
