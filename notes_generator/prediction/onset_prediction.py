from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from notes_generator.constants import ConvStackType
from notes_generator.models.beats import gen_beats_array


def predict(
    model,
    mel,
    condition: int,
    threshold: float = 0.5,
    bpm_info: Optional[List[Tuple[Union[float, int], int, int]]] = None,
    device: str = "cpu",
    use_rand: bool = False,
):
    probs = predict_proba(model, mel, condition, bpm_info, device)
    if use_rand:
        rn = np.random.rand(*probs.shape)
        prediction_ = (probs >= rn).astype(np.int)
    else:
        prediction_ = (probs >= threshold).astype(np.int)
    return prediction_, probs


def predict_proba(
    model,
    mel,
    condition: int,
    bpm_info: Optional[List[Tuple[Union[float, int], int, int]]] = None,
    device: str = "cpu",
):
    model.eval()
    with torch.no_grad():
        mel = torch.tensor(mel)
        mel = mel.reshape(1, mel.shape[0], mel.shape[1]).float().to(device)
        # shape: (BATCH, TIME, N_MEL)
        condition = torch.tensor([condition]).expand(1, mel.shape[1], 1).float().to(device)
        beats_arr = None
        if bpm_info is not None:
            beats_arr = (
                torch.from_numpy(gen_beats_array(mel.shape[1], bpm_info, mel.shape[1]))
                .reshape(1, -1, 1)
                .float()
                .to(device)
            )
        prediction = model(mel, condition, beats_arr)
        probs = (prediction[0]).cpu().numpy()
        return probs
