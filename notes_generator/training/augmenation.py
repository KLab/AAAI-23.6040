import math
import typing
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
import yaml

from notes_generator.constants import FRAME, NMELS

Sample = typing.Dict[str, torch.Tensor]


class AugConfig(typing.NamedTuple):
    function: str
    probability: float
    params: typing.Dict[str, typing.Any]


def load_augmentations(file_path: Path) -> typing.List[AugConfig]:
    with file_path.open() as fp:
        da = yaml.safe_load(fp)
    return [AugConfig(**config) for config in da]


def apply_augmentation(augmentations: typing.List[AugConfig], data: Sample) -> Sample:
    for aug in augmentations:
        params = aug.params or dict()
        if np.random.rand() < aug.probability:
            data = globals()[aug.function](data, **params)
    return data


def freq_shift(data: Sample, shift_min: float = -0.1, shift_max: float = 0.1) -> Sample:
    audio = data["audio"]
    r_min = int(audio.shape[1] * shift_min)
    r_max = int(audio.shape[1] * shift_max)
    r = np.random.randint(r_min, r_max + 1)
    i = torch.arange(-r, audio.shape[1] - r) % audio.shape[1]
    audio = audio[:, i]
    data["audio"] = audio
    return data


def freq_mask(data: Sample, mask_max: int = 4, mask_count: int = 1) -> Sample:
    audio = data["audio"]
    width = np.random.randint(1, mask_max + 1)  # plus one for including mask_max
    count = np.random.randint(1, mask_count + 1)
    mean = torch.mean(audio)
    for _ in range(count):
        i = np.random.randint(audio.shape[1] - width)
        audio[:, i : (i + width)] = mean
    data["audio"] = audio
    return data


def time_mask(data: Sample, mask_max: int = 4, mask_count: int = 1) -> Sample:
    audio = data["audio"]
    width = np.random.randint(1, mask_max + 1)  # plus one for including mask_max
    count = np.random.randint(1, mask_count + 1)
    mean = torch.mean(audio)
    for _ in range(count):
        i = np.random.randint(audio.shape[1] - width)
        audio[i : (i + width), :] = mean
    data["audio"] = audio
    return data


def white_noise(data: Sample, sigma: float = 0.03) -> Sample:
    audio = data["audio"]
    r = sigma * torch.randn(audio.shape, device=audio.device)
    audio += r
    data["audio"] = audio
    return data


def freq_low_mask(data: Sample, min_mask_range: int = 10, max_mask_range: int = 19) -> Sample:
    audio = data["audio"]
    mean = torch.mean(audio)
    p = np.random.randint(
        min_mask_range, max_mask_range + 1
    )  # plus one for including max_mask_range
    audio[:, -p:] = mean
    data["audio"] = audio
    return data


def freq_flip_audio(data: Sample) -> Sample:
    audio = data["audio"]
    audio = torch.flip(audio, dims=(1,))
    data["audio"] = audio
    return data


def mask_beats(data: Sample, drop_rate: float = 0.2) -> Sample:
    if "beats" in data:
        beats = data["beats"]
        beats = F.dropout(beats, drop_rate)
        data["beats"] = beats
    return data


def _audio_stretch(array: torch.Tensor, rate: float):
    """Stretch the mel spectrogram tensor."""
    audio_stretch = T.TimeStretch(n_freq=NMELS)
    array_conv = array.transpose(-2, -1)  # convert shape to (freq, timestep)
    array_conv = torch.exp(array_conv)

    # Add a dimension containing 2 elems for representing complex number
    array_conv = array_conv.unsqueeze(-1)
    array_conv = torch.cat([array_conv, torch.zeros(array_conv.shape)], dim=-1)

    array_conv = audio_stretch(array_conv, rate)  # shape: (freq, timestep, 2)
    array_conv = AF.complex_norm(array_conv).type(array.type())  # shape: (freq, timestep)
    array_conv = array_conv.transpose(-2, -1)  # convert shape to (timestep, freq)
    array_conv = torch.log(array_conv)
    return array_conv


def _label_stretch(array: torch.Tensor, rate: float):
    """Stretch the tensor containing pulses without varying pulse width."""
    # array.shape: (timestep, feature)
    result_shape = (math.ceil(array.shape[0] / rate), array.shape[1])
    array_conv = torch.zeros(*result_shape, dtype=array.dtype)

    indices = torch.nonzero(array, as_tuple=True)
    time_indices = torch.clone(indices[0])
    times_conv = FRAME * time_indices + FRAME / 2  # calc the center time for each frames
    time_indices_conv = torch.floor(times_conv / rate / FRAME).long()
    indices_conv = (time_indices_conv, indices[1])

    array_conv[indices_conv] = array[indices]
    return array_conv


def time_stretch(data: Sample, min_rate: float, max_rate: float):
    """Apply time stretch to sample data.

    Notes
    -----
    rate = 「倍速」
    """
    # we don't consider shrinking operation (rate > 1)
    assert 0 < min_rate < max_rate <= 1

    audio = data["audio"]  # shape: (timestep, freq)
    onset = data["onset"]  # shape: (timestep, 1)

    rate = np.random.uniform(min_rate, max_rate)
    audio_conv = _audio_stretch(audio, rate)
    onset_conv = _label_stretch(onset, rate)

    data["audio"] = audio_conv[: audio.shape[0]]
    data["onset"] = onset_conv[: onset.shape[0]]

    if "beats" in data:
        beats = data["beats"]  # shape: (timestep, 1)
        beats_conv = _label_stretch(beats, rate)
        data["beats"] = beats_conv[: beats.shape[0]]

    return data


def abs_mel(data: Sample):
    """ABS Mel Log Value"""
    data["audio"] = torch.abs(data["audio"])
    return data


def time_stretch_abs(data: Sample, min_rate: float, max_rate: float):
    data = time_stretch(data, min_rate, max_rate)
    data["audio"] = torch.abs(data["audio"])
    return data
