"""メルスペクトラムデータ作成
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd

from constants import *
from dataset.smdataset.parse import extract_time_signature


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
    path2row = {d["bgm_path"]: d for idx, d in m_live_data.iterrows()}
    for bgm_path, row in path2row.items():
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
    path2row = {d["bgm_path"]: d for idx, d in m_live_data.iterrows()}
    future_list = []
    with ProcessPoolExecutor() as executor:
        for bgm_path, row in path2row.items():
            live_id = row["live_id"]
            bpm_info = [[int(d) for d in line.split(",")] for line in row["bpm_info"].split("\n")]
            bpm = [int(bpm) for bpm in row["bpm"].split(",")]
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
        music_file_path: str,
        live_id: str,
        save_base_path: Path,
        aug_count: int = 0,
        log_enable: bool = True,
        bpm_info: Optional[List] = None,
        bpm: Optional[List[int]] = None,
        noise_rate: float = 0.005,
):
    print(f"bgm_path: {music_file_path} live_id: {live_id}, log_enable: {log_enable}")
    full_path = Path(music_file_path)
    print(f"full path: {full_path}")
    data, sr = librosa.load(str(full_path), sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    # 曲の長さ(秒)
    wav_length = data.shape[0] / sr
    # テンポ
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
        # ホワイトノイズでデータ水増し
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
        print(bpm_info)
        print(np.array(bpm_info).shape)
        if np.array(bpm_info).shape[-1] == 2:
            for index, bpm_list in enumerate(bpm_info):
                bpm_info[index].append(4.0)
        print(bpm_info)
        # assert shape
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
    """曲の長さをミリ秒で取得
    """
    data, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    return data.shape[0] * 1000 // sr


if __name__ == "__main__":
    save_dir = "../data/mel_log"
    future_list = []
    with ProcessPoolExecutor() as executor:
        for root, subdir, files in os.walk("../data/json_raw"):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file)) as json_file:
                        json_info = json.load(json_file)
                    music_file_path = json_info['music_fp']
                    bpms = json_info['bpms']
                    sm_file_path = Path(json_info["sm_fp"])
                    ssc_file_path = os.path.join(sm_file_path.parent, Path(sm_file_path).stem + ".ssc")
                    print("BPM: ", end='')
                    print(bpms)
                    bpm_df = pd.DataFrame(bpms, columns=['time', 'bpm'])
                    if Path(ssc_file_path).exists():
                        with open(ssc_file_path, 'r') as ssc_file:
                            time_signatures = extract_time_signature(ssc_file.read())['timesignatures']
                            time_signature_df = pd.DataFrame(time_signatures, columns=['time', 'time_signature'])
                            time_signature_df[["numerator", "denominator"]] = pd.DataFrame(
                                time_signature_df['time_signature'].tolist())
                            time_signature_df["regularized_numerator"] = 4 // time_signature_df["denominator"] * \
                                                                         time_signature_df["numerator"]
                            print("Time Signature: ", end='')
                            print(time_signatures)
                        bpm_df = pd.merge(bpm_df, time_signature_df, on="time", how="outer").sort_values(
                            by=['time']).fillna(method='ffill')
                        bpm_df = bpm_df[["time", "bpm", "regularized_numerator"]].copy()
                    file_path = Path(os.path.join(save_dir, root.split('/')[-2], root.split('/')[-1], Path(file).stem))
                    convert(music_file_path=music_file_path, live_id=json_info['pack'] + '_' + json_info['title'],
                            save_base_path=file_path, bpm_info=bpm_df.values.tolist())
                    # future = executor.submit(
                    #     convert,
                    #     music_file_path=music_file_path,
                    #     live_id=json_info['pack'] + '_' + json_info['title'],
                    #     save_base_path=file_path,
                    #     bpm_info=bpm_df.values.tolist()
                    # )
                    # future_list.append(future)
