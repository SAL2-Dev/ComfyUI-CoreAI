"""
image_utils.py — Convert between ComfyUI image tensors and PNG files.

ComfyUI images are CHW float32 tensors in [0, 1] range, batched as [B, C, H, W].
The Swift runner expects file paths (PNG). This module bridges the two:

    ComfyUI tensor [B,3,H,W] float32 → PNG file → Swift → result file → tensor
"""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def tensor_to_png(tensor) -> str:
    """
    Save a ComfyUI image tensor as a PNG file.
    Handles both single [C,H,W] and batched [B,C,H,W] tensors.

    For batched input, saves the first image in the batch.

    Returns: absolute path to the temp PNG file.
    """
    # Detach if torch tensor
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)

    # Handle batch dimension [B,C,H,W] → take first [C,H,W]
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {arr.ndim}D")

    # CHW → HWC
    if arr.shape[0] <= 4:  # channels-first
        arr = np.transpose(arr, (1, 2, 0))

    # [0, 1] float → [0, 255] uint8
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    # Handle grayscale (H,W,1 → H,W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # Handle RGBA → RGB (Core AI expects RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Write PNG
    img = Image.fromarray(arr)
    path = Path(tempfile.gettempdir()) / f"coreai_in_{uuid.uuid4().hex}.png"
    img.save(path, format="PNG")
    return str(path)


def png_to_tensor(png_path: str):
    """
    Load a PNG file as a ComfyUI image tensor [1, C, H, W] float32 in [0, 1].
    """
    img = Image.open(png_path)

    # Convert to RGB (handles RGBA, grayscale, palette)
    if img.mode == "L":
        # Grayscale depth map → keep as single channel but expand to 3 for ComfyUI
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=-1)
    elif img.mode == "RGBA":
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))

    # Add batch dimension [1, C, H, W]
    arr = np.expand_dims(arr, axis=0)

    if HAS_TORCH:
        return torch.from_numpy(arr)
    return arr


def load_output_image(output_path: str):
    """Load any output file from the runner as a ComfyUI tensor."""
    return png_to_tensor(output_path)


def cleanup_temp(path: str) -> None:
    """Remove a temp file, ignoring errors."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
