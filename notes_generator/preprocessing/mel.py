"""Create mel-spectrogram from audio
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from notes_generator.constants import *


def _format(d):
    if "." in d:
        return float(d)
    else:
        return int(d)


def convert_all(
    m_live_data: pd.DataFrame,
    wav_base_path: Path,
    save_base_path: Path,
    aug_count: int = 0,
    log_enable: bool = False,
    noise_rate: float = 0.005,
):
    df = m_live_data.groupby("live_id", as_index=False).first()
    for i, row in df.iterrows():
        bgm_path = row["bgm_path"]
        live_id = row["live_id"]
        bpm_info = [[_format(d) for d in line.split(",")] for line in row["bpm_info"].split("\n")]
        bpm = [_format(bpm) for bpm in row["bpm"].split(",")]
        print(bpm)
        print(bpm_info)
        save_path = save_base_path / str(live_id)
        convert(
            bgm_path,
            live_id,
            wav_base_path,
            save_path,
            aug_count=aug_count,
            bpm_info=bpm_info,
            bpm=bpm,
            log_enable=log_enable,
            noise_rate=noise_rate,
        )


def convert_all_parallel(
    m_live_data: pd.DataFrame,
    wav_base_path: Path,
    save_base_path: Path,
    aug_count: int = 0,
    log_enable: bool = False,
    noise_rate: float = 0.005,
):
    future_list = []
    with ProcessPoolExecutor() as executor:
        df = m_live_data.groupby("live_id", as_index=False).first()
        for i, row in df.iterrows():
            # bpm_path is the stem of audio file name
            bgm_path = row["bgm_path"]
            live_id = row["live_id"]
            bpm_info = [
                [_format(d) for d in line.split(",")] for line in row["bpm_info"].split("\n")
            ]
            bpm = [_format(bpm) for bpm in row["bpm"].split(",")]
            save_path = save_base_path / str(live_id)
            future = executor.submit(
                convert,
                bgm_path,
                live_id,
                wav_base_path,
                save_path,
                aug_count=aug_count,
                bpm_info=bpm_info,
                bpm=bpm,
                log_enable=log_enable,
                noise_rate=noise_rate,
            )
            future_list.append(future)
        for future in future_list:
            print(future.result())


def add_white_noise(x, rate=0.005):
    return x + rate * np.random.randn(len(x))


def convert(
    bgm_path: str,
    live_id: int,
    wav_base_path: Path,
    save_base_path: Path,
    aug_count: int = 0,
    log_enable: bool = True,
    bpm_info: Optional[List] = None,
    bpm: Optional[List[int]] = None,
    noise_rate: float = 0.005,
):
    print(f"bgm_path: {bgm_path} live_id: {live_id}, log_enable: {log_enable}")
    full_path = wav_base_path / f"{bgm_path}"
    print(f"full path: {full_path}")
    data, sr = librosa.load(str(full_path), sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    # Song length [second]
    wav_length = data.shape[0] / sr
    # Tempo
    tempo = list(librosa.beat.tempo(data, sr))

    data_dic = dict()
    save_path = save_base_path / "mel.npz"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    mel = librosa.feature.melspectrogram(
        data, sr, hop_length=HOP_LENGTH, fmin=30.0, n_mels=NMELS, htk=True
    )
    if log_enable:
        mel = np.log(np.clip(mel, 1e-5, None))
    mel = mel.T
    data_dic["mel"] = mel
    idx = 0
    while aug_count > 0:
        # Add white noise for an augmentation
        data2 = add_white_noise(data, rate=noise_rate)
        mel2 = librosa.feature.melspectrogram(
            data2, sr, hop_length=HOP_LENGTH, fmin=30.0, n_mels=NMELS, htk=True
        )
        if log_enable:
            mel2 = np.log(np.clip(mel2, 1e-5, None))
        mel2 = mel2.T
        data_dic[f"mel_noise_{idx}"] = mel2
        idx += 1
        aug_count -= 1
    np.savez(str(save_path), **data_dic)
    mel_length = mel.shape[0] * HOP_LENGTH / SAMPLE_RATE
    metadata = dict(
        live_id=live_id,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        wav_length=wav_length,
        mel_length=mel_length,
        tempo=tempo,
    )
    if bpm_info:
        # assert shape
        print(bpm_info)
        assert (
            np.array(bpm_info).shape[-1] == 3
        ), f"The shape of bpm_info is invalid. live_id={live_id}"
        assert (
            np.array(bpm_info).ndim == 2
        ), f"The dimension of bpm_info is invalid. live_id={live_id}"
        metadata["bpm_info"] = bpm_info
    if bpm:
        metadata["tempo"] = bpm
    print(f"mel length: {mel_length} wav length: {wav_length} tempo: {tempo}")
    metapath = save_path.parent / "meta.json"
    with metapath.open("w") as fp:
        json.dump(metadata, fp)


def get_song_length(audio_path: Path) -> int:
    """Get song length in milliseconds"""
    data, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    return data.shape[0] * 1000 // sr
