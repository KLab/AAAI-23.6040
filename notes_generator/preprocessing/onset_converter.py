"""Create train label for onset model
"""
import json
import math
from enum import Enum
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd

from notes_generator.constants import AppName, FRAME, get_difficulty_type_enum

allowed_appname = [AppName.STEPMANIA, AppName.STEPMANIA_I, AppName.STEPMANIA_F]


def validate_data(df):
    """Raise an error if source data is invalid.

    Expected state
    ------------
    For a tuple (live_id, difficulty), there exists only single corresponding
    `live_difficulty_id`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing master data of notes.

    """
    # Count live_difficulty_id for each (live_id, live_difficulty_type) pair.
    df_agg = df.groupby(["live_id", "live_difficulty_type"]).agg({"live_difficulty_id": "nunique"})

    # Filter live_difficulty_id whose count is more than one
    invalid_live_ids = df_agg[df_agg["live_difficulty_id"] != 1].reset_index()["live_id"].unique()

    assert len(invalid_live_ids) == 0, (
        "JSON file contains unnecessary data. "
        "Please check the data corresponding to live_ids below:\n"
        f"{invalid_live_ids}"
    )
    return


def convert(values, length: int, unit: int):
    """
    Convert notes data into train label

    Parameters
    ----------
    values: Notes data
    length: Song length
    unit: The length of single frame

    Returns
    -------

    """
    # frame count
    hop_length = int(math.ceil(length / unit)) + 2
    # return: #frame×1
    ret = np.zeros((hop_length, 1))
    for row in values:
        msec = row[6]
        time = int(round(msec / unit))
        ret[time, 0] = 1
    return ret


def handle_df(df, save_path, difftype_cls: Type[Enum]):
    live_ids = list(df.groupby("live_id").groups.keys())
    live_ids.sort()
    mt = df[["live_id", "timing_msec"]].groupby("live_id", as_index=False).max()
    song_length_dict = {k: v + 1 for k, v in mt.values}
    for live_id in live_ids:
        dic = dict()
        for diff_type in [t.value for t in difftype_cls]:
            score = df[(df["live_id"] == live_id) & (df["live_difficulty_type"] == diff_type)]
            # skip non-existing difficulty
            if not len(score):
                continue
            converted = convert(score.values, song_length_dict[live_id], FRAME)
            print(f"live_id: {live_id} diff_type: {diff_type} length: {len(converted)}")
            dic[difftype_cls(diff_type).name] = converted
        base_path = save_path / str(live_id)
        if not base_path.exists():
            base_path.mkdir(parents=True)
        file_path = base_path / "dump"
        np.savez(str(file_path), **dic)
        metadata = dict(live_id=int(live_id))
        metadata_path = base_path / "meta.json"
        with metadata_path.open("w") as fp:
            json.dump(metadata, fp)


def main(data_path, save_path, app_name: AppName):
    assert app_name in allowed_appname
    difftype_cls = get_difficulty_type_enum(app_name)
    df = pd.read_json(data_path, lines=True)
    validate_data(df)
    save_path = Path(save_path)
    handle_df(df, save_path, difftype_cls)
