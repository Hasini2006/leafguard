"""Send traffic to the running service, for the Grafana dashboards and the drift demo.

Normal mode sends unmodified test images, which fills in the request, latency, class and
confidence panels. Drift mode darkens and blurs the same images before sending them, which
makes the PSI drift panel climb and shows what a change in input conditions looks like.

Usage:
  python scripts/demo_traffic.py --count 60
  python scripts/demo_traffic.py --count 60 --drift
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter

SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(folder: Path) -> list[Path]:
    images = [p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES]
    if not images:
        raise SystemExit(f"no images found under {folder}")
    return images


def degrade(raw: bytes) -> bytes:
    """Make an image look like it came from a different camera or a darker day."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(0.45)
    image = ImageEnhance.Color(image).enhance(0.4)
    image = image.filter(ImageFilter.GaussianBlur(1.2))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--folder", default="data/raw/val", type=Path)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--pause", type=float, default=0.4, help="seconds between requests")
    parser.add_argument("--drift", action="store_true", help="send darkened, blurred images")
    args = parser.parse_args()

    images = collect_images(args.folder)
    endpoint = f"{args.url.rstrip('/')}/api/v1/predict"
    counts: dict[str, int] = {}
    failures = 0

    print(f"sending {args.count} {'degraded' if args.drift else 'normal'} images to {endpoint}")
    for number in range(1, args.count + 1):
        raw = random.choice(images).read_bytes()
        if args.drift:
            raw = degrade(raw)
        try:
            reply = requests.post(
                endpoint, files={"file": ("leaf.jpg", raw, "image/jpeg")}, timeout=30
            )
            if reply.status_code == 200:
                body = reply.json()
                key = f"{body['predicted_class']} ({body['status']})"
                counts[key] = counts.get(key, 0) + 1
            else:
                failures += 1
        except requests.RequestException:
            failures += 1
        if number % 10 == 0:
            print(f"  {number}/{args.count} sent")
        time.sleep(args.pause)

    print("\nresults")
    for key, number in sorted(counts.items(), key=lambda pair: -pair[1]):
        print(f"  {number:4d}  {key}")
    if failures:
        print(f"  {failures} requests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
