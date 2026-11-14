"""Pure image helpers used by the Spark stages, the drift baseline and the backend.

Nothing here imports Spark or Torch, so the same code runs inside Spark workers and
inside the slim backend container, and every function is easy to unit-test.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from leafguard.common import ImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def local_path(uri: str) -> str:
    """Turn a Spark 'file:/...' URI into a plain local path."""
    for prefix in ("file://", "file:"):
        if uri.startswith(prefix):
            return unquote(uri[len(prefix) :])
    return uri


def split_and_label(path: str) -> tuple[str, str]:
    """Read '<source_split>/<label>/<file>' out of an image path."""
    parts = PurePosixPath(local_path(path))
    return parts.parent.parent.name, parts.parent.name


def file_hash(data: bytes) -> str:
    """md5 of the raw bytes: used only to spot exact duplicate files."""
    return hashlib.md5(data).hexdigest()  # noqa: S324


def decode(data: bytes) -> Image.Image:
    """Fully decode image bytes, raising ImageError for corrupt or truncated files."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as err:
        raise ImageError(f"cannot decode image: {err}") from err
    return image


def perceptual_hash(image: Image.Image, size: int = 8) -> str:
    """Difference hash. Near-identical photos of the same leaf share this value."""
    grey = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def blur_score(image: Image.Image) -> float:
    """Variance of the Laplacian. A low value means a blurry image.

    Written with numpy slicing instead of OpenCV so the backend container stays small.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    if min(grey.shape) < 3:
        return 0.0
    laplacian = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4.0 * grey[1:-1, 1:-1]
    )
    return float(laplacian.var())


def green_ratio(image: Image.Image) -> float:
    """Share of green-ish, reasonably saturated pixels. Low means 'probably not a leaf'."""
    hsv = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.uint8)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Pillow stores hue as 0-255 for 0-360 degrees, so 25..120 is about 35..170 degrees.
    leafy = (hue >= 25) & (hue <= 120) & (sat >= 40) & (val >= 40)
    return float(leafy.mean())


def brightness(image: Image.Image) -> float:
    """Mean luminance, scaled to 0..1."""
    return float(np.asarray(image.convert("L"), dtype=np.float32).mean() / 255.0)


def saturation(image: Image.Image) -> float:
    """Mean saturation, scaled to 0..1."""
    hsv = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255.0)


def image_stats(image: Image.Image) -> dict[str, float]:
    """The four features tracked for drift, for one image."""
    rgb = image.convert("RGB")
    return {
        "brightness": round(brightness(rgb), 6),
        "saturation": round(saturation(rgb), 6),
        "green_ratio": round(green_ratio(rgb), 6),
        "blur": round(blur_score(rgb), 4),
    }


def inspect(data: bytes, min_side: int, blur_limit: float) -> dict[str, Any]:
    """Decode an image and work out every quality attribute the validate stage needs."""
    result: dict[str, Any] = {
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "phash": None,
        "blur": None,
        "green_ratio": None,
        "brightness": None,
        "saturation": None,
        "corrupt": False,
        "too_small": False,
        "too_blurry": False,
        "bad_format": False,
        "error": None,
    }
    try:
        image = decode(data)
    except ImageError as err:
        result["corrupt"] = True
        result["error"] = str(err)[:200]
        return result

    width, height = image.size
    rgb = image.convert("RGB")
    stats = image_stats(rgb)
    result.update(stats)
    result.update(
        width=width,
        height=height,
        mode=image.mode,
        format=image.format,
        phash=perceptual_hash(rgb),
        too_small=min(width, height) < min_side,
        too_blurry=blur_limit > 0 and stats["blur"] < blur_limit,
        bad_format=image.format not in ALLOWED_FORMATS,
    )
    return result


def resize_square(data: bytes, size: int, quality: int = 90) -> bytes:
    """Centre-crop to a square, resize, and re-encode as an RGB JPEG."""
    image = decode(data).convert("RGB")
    image = ImageOps.fit(image, (size, size), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def to_model_array(image: Image.Image, size: int) -> np.ndarray:
    """Resize and normalise one image into the CHW float array the model expects.

    The serving wrapper repeats these two lines on purpose (it must stay import-free),
    so keep the two copies in step if you change the normalisation.
    """
    rgb = ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.BILINEAR)
    array = (np.asarray(rgb, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return array.transpose(2, 0, 1)
