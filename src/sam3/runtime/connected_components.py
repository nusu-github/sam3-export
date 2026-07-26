"""Pure-torch connected components for runtime mask postprocessing.

Matches ``skimage.measure.label`` connectivity used by official SAM3 CPU path:
same non-zero value, 8-connected. Host postprocess (not on ``torch.export`` path).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _connected_components_cpu_single(values: Tensor) -> tuple[Tensor, Tensor]:
    """Label 8-connected equal-value components on one HxW plane."""
    assert values.dim() == 2, "values must be 2D"
    if values.numel() == 0:
        empty = torch.zeros_like(values, dtype=torch.int32)
        return empty, empty

    values_cpu = values.detach().cpu()
    height, width = values_cpu.shape
    labels = torch.zeros((height, width), dtype=torch.int32)
    counts = torch.zeros((height, width), dtype=torch.int32)
    if height == 0 or width == 0:
        return labels.to(values.device), counts.to(values.device)

    flat_vals = values_cpu.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_counts = counts.reshape(-1)
    # 8-connected neighborhood (matches skimage default connectivity=2).
    neighbors = (
        -width - 1,
        -width,
        -width + 1,
        -1,
        1,
        width - 1,
        width,
        width + 1,
    )

    for h in range(height):
        for w in range(width):
            start = h * width + w
            if flat_vals[start] == 0 or flat_labels[start] != 0:
                continue

            comp_label = start + 1
            current_value = flat_vals[start]
            stack = [start]
            component: list[int] = []

            while stack:
                idx = stack.pop()
                if idx < 0 or idx >= flat_labels.numel():
                    continue
                if flat_labels[idx] != 0:
                    continue
                if flat_vals[idx] != current_value or flat_vals[idx] == 0:
                    continue

                flat_labels[idx] = comp_label
                component.append(idx)

                r, c = divmod(idx, width)
                for d in neighbors:
                    n_idx = idx + d
                    if n_idx < 0 or n_idx >= flat_labels.numel():
                        continue
                    nr, nc = divmod(n_idx, width)
                    # Reject wrap-around on row boundaries from ±1 column steps.
                    if abs(nr - r) > 1 or abs(nc - c) > 1:
                        continue
                    stack.append(n_idx)

            comp_size = len(component)
            for idx in component:
                flat_counts[idx] = comp_size

    return labels.to(values.device), counts.to(values.device)


def _connected_components_cpu(mask: Tensor) -> tuple[Tensor, Tensor]:
    out_shape = mask.shape
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0, :, :]
    else:
        assert mask.dim() == 3, "Input tensor must be (B, H, W) or (B, 1, H, W)."

    if mask.dim() == 3 and mask.shape[0] == 0:
        return (
            torch.zeros(mask.shape, dtype=torch.int32, device=mask.device),
            torch.zeros(mask.shape, dtype=torch.int32, device=mask.device),
        )

    batch = mask.shape[0]
    label_list: list[Tensor] = []
    count_list: list[Tensor] = []
    for b in range(batch):
        labels, counts = _connected_components_cpu_single(mask[b])
        label_list.append(labels)
        count_list.append(counts)

    out_labels = torch.stack(label_list, dim=0).to(mask.device)
    out_counts = torch.stack(count_list, dim=0).to(mask.device)
    return out_labels.reshape(out_shape), out_counts.reshape(out_shape)


def connected_components(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compute connected-components labels and per-pixel component sizes."""
    if mask.dim() == 3:
        in_tensor = mask
        out_shape = mask.shape
    elif mask.dim() == 4 and mask.shape[1] == 1:
        in_tensor = mask[:, 0, :, :]
        out_shape = mask.shape
    else:
        raise ValueError("Input tensor must be (B, H, W) or (B, 1, H, W).")

    if in_tensor.numel() == 0:
        zeros = torch.zeros(out_shape, dtype=torch.int32, device=mask.device)
        return zeros, zeros.clone()
    labels, counts = _connected_components_cpu(in_tensor)
    if out_shape != labels.shape:
        return labels.reshape(out_shape), counts.reshape(out_shape)
    return labels, counts
