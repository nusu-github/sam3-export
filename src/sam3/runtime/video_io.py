"""Host-side JPEG-folder video input utilities.

Port of the JPEG path from ``sam3.model.utils.sam2_utils`` (sync only).
"""

from __future__ import annotations

from collections.abc import Sequence
import os

from jaxtyping import Float
import numpy as np
from PIL import Image
import torch
from torch import Tensor


def _load_img_as_tensor(
    img_path: str, image_size: int
) -> tuple[Float[Tensor, "3 h w"], int, int]:
    img_pil = Image.open(img_path)
    video_width, video_height = img_pil.size
    img_np = np.array(
        img_pil.convert("RGB").resize((image_size, image_size)), dtype=np.uint8
    )
    img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return img, video_height, video_width


def list_jpeg_frames(video_path: str) -> list[str]:
    """Sorted ``<int>.jpg`` paths under a directory."""
    if not os.path.isdir(video_path):
        raise NotADirectoryError(video_path)
    names = [
        p
        for p in os.listdir(video_path)
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg")
    ]
    if not names:
        raise RuntimeError(f"no JPEG frames in {video_path}")
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return [os.path.join(video_path, n) for n in names]


def load_video_frames_from_jpg(
    video_path: str,
    image_size: int = 1008,
    *,
    offload_video_to_cpu: bool = False,
    img_mean: Sequence[float] = (0.5, 0.5, 0.5),
    img_std: Sequence[float] = (0.5, 0.5, 0.5),
    max_frames: int | None = None,
    device: str | torch.device = "cuda",
) -> tuple[Float[Tensor, "t 3 h w"], int, int]:
    """Load JPEG folder → ``(T, 3, S, S)`` normalized float32 tensor.

    Returns ``(images, original_height, original_width)``.
    """
    paths = list_jpeg_frames(video_path)
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    t = len(paths)
    images = torch.zeros(t, 3, image_size, image_size, dtype=torch.float32)
    video_height = video_width = 0
    for n, path in enumerate(paths):
        images[n], video_height, video_width = _load_img_as_tensor(path, image_size)

    mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    images = (images - mean) / std

    if not offload_video_to_cpu:
        images = images.to(device)
    return images, video_height, video_width
