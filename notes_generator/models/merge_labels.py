import typing

import torch


def merge_labels(onset_label: torch.Tensor, batch: typing.Dict, scale: float) -> torch.Tensor:
    assert "other_conditions" in batch
    other_conditions = batch["other_conditions"]
    for condition, score in other_conditions.items():
        onset_label = torch.max(onset_label, score * scale)
    return onset_label
