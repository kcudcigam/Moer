import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def read_hdr(path):
    image = iio.imread(Path(path))
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def mse(image, reference):
    diff = image.astype(np.float64) - reference.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(image, reference):
    value = mse(image, reference)
    if value <= 0:
        return float("inf")
    peak = float(max(np.max(reference), 1.0))
    return float(20.0 * math.log10(peak) - 10.0 * math.log10(value))


def mae(image, reference):
    return float(np.mean(np.abs(image.astype(np.float64) - reference.astype(np.float64))))


def metrics(image_path, reference_path):
    image = read_hdr(image_path)
    reference = read_hdr(reference_path)
    if image.shape != reference.shape:
        raise ValueError(f"shape mismatch: {image.shape} vs {reference.shape}")
    return {
        "mse": mse(image, reference),
        "mae": mae(image, reference),
        "psnr": psnr(image, reference),
    }
