import torch
import torch.nn.functional as F

from notes_generator.models.util import round_decimal


def shift(ar, size, med):
    # [0, 0, 0, 1, 0, 0...]
    # -> [0, 0, med - 1, 0, mid - 1, 0 ...]
    if size > 0:
        ar = F.pad(ar[size:], [0, size]) + F.pad(ar[:-size], [size, 0])
    ar = ar * (med - size)
    return ar


def gauss(ar, width=3, scale=1.0):
    # [0, 0, 0, 1, 0, 0...]
    # -> [0, 1, 2, 3, 2, 1...]
    t = torch.stack([shift(ar, i, width) for i in range(width)])
    t = torch.max(t, dim=0)[0]
    mask = t == 0
    # Heuristically define std value so that non-zero value appears
    # within a range of `width` parameter of the center when the values
    # are rounded at the second decimal place.
    std = width / 3
    var = std**2
    t2 = torch.exp(-((t - width) ** 2) / (2 * var)) * scale
    return torch.max(t2.masked_fill(mask, 0), ar.float())


def fuzzy_label(onset_label: torch.Tensor, width: int, scale: float) -> torch.Tensor:
    """Apply fuzzy label

    Parameters
    ----------
    onset_label : torch.Tensor  shape = [frame_len, 1]
    width : int
    scale : float

    Returns
    -------
    fuzzy_labeled_onset : torch.Tensor  shape = [frame_len, 1]

    Examples
    >>> onset = torch.tensor([[0.], [0.], [1.], [0.], [1.], [0.], [0.], [0.]])
    >>> fuzzy_label(onset, width=2)
    tensor([[0.0100], [0.3200], [1.0000], [0.3200], [1.0000], [0.3200],
           [0.0100], [0.0000]])
    """
    assert width > 0
    assert 0 <= scale <= 1
    return round_decimal(gauss(onset_label.view(-1), width, scale), 2).view(len(onset_label), 1)


def fuzzy_on_batch(batch: torch.Tensor, width: int, scale: float) -> torch.Tensor:
    """Apply fuzzy label to batch

    Parameters
    ----------
    batch : torch.Tensor  shape = [batch_size, frame_len, 1]
    width : int
    scale : float

    Returns
    -------
    fuzzy_labeled_onsets : torch.Tensor  shape = [batch_size, frame_len, 1]

    """
    fuzzy_labeled = []
    for x in range(batch.shape[0]):
        fuzzy_labeled.append(fuzzy_label(batch[x], width, scale))
    return torch.stack(fuzzy_labeled)
