import argparse
import json
import os
import shutil
from collections import OrderedDict
from operator import itemgetter

import pandas as pd

difficulty_id_map = {
    "Beginner": 10,
    "Easy": 20,
    "Medium": 30,
    "Hard": 40,
    "Challenge": 50,
}


package_ids = {
    "fraxtil/Fraxtil_sArrowArrangements": 1,
    "fraxtil/Fraxtil_sBeastBeats": 2,
    "fraxtil/TsunamixIII": 3,
    "itg/InTheGroove": 4,
    "itg/InTheGroove2": 5,
}


def parse_code_string(code: str):
    # code is 4 charactered string
    rails, note_types = [], []
    for i, char in enumerate(code):
        if char != "0":
            rails.append(i + 1)  # integer within [1, 4]
            note_types.append(int(char))  # integer within [1, 3]
    return rails, note_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", help="The directory containing filtered JSON files")
    parser.add_argument("save_dir", help="Output directory")
    args = parser.parse_args()

    live_data = []
    notes_data = []
    song_paths = []
    save_dir = args.save_dir
    json_dir = args.json_dir
    print(f"save_dir: {str(save_dir)}")
    print(f"json_dir: {str(json_dir)}")

    for pack_dir, pack_id in package_ids.items():
        search_dir = os.path.join(json_dir, pack_dir)
        song_count = 1
        for root, subdir, files in os.walk(search_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(root, file)) as json_file:
                    print(file)
                    json_info = json.load(json_file)

                music_file_path = json_info["music_fp"]
                title = json_info["title"]
                bpms = json_info["bpms"]
                bpm_info = json_info["bpm_info"]
                bpms = sorted(bpms, key=itemgetter(0))
                live_id_str = str(pack_id * 10000 + song_count)
                charts = json_info["charts"]

                song_paths.append((live_id_str, music_file_path))

                for chart in charts:
                    difficulty_coarse = chart["difficulty_coarse"]
                    notes = chart["notes"]
                    difficulty_type = difficulty_id_map[difficulty_coarse]
                    live_diff_id = live_id_str + str(difficulty_type)
                    note_id = 0
                    for note in notes:
                        timing_ms = note[2] * 1000
                        code = note[3]  # 4 charactered string indicating notes type
                        rails, note_types = parse_code_string(code)
                        for rail, note_type in zip(rails, note_types):
                            note_id += 1
                            notes_data.append(
                                OrderedDict(
                                    [
                                        ("live_id", live_id_str),
                                        ("live_difficulty_id", live_diff_id),
                                        ("live_difficulty_type", difficulty_type),
                                        ("title", title),
                                        ("note_id", note_id),
                                        ("timing_msec", timing_ms),
                                        ("rail", rail),
                                        ("note_type", note_type),
                                        ("diff", ""),
                                        ("bgm_path", music_file_path),
                                    ]
                                )
                            )

                    bpm_info_str = "\n".join([",".join([str(e) for e in x]) for x in bpm_info])
                    live_data.append(
                        OrderedDict(
                            [
                                ("live_id", live_id_str),
                                ("live_difficulty_id", live_diff_id),
                                ("live_difficulty_type", difficulty_type),
                                ("title", title),
                                ("bgm_path", music_file_path),
                                ("notes_count", note_id),
                                ("bpm_info", bpm_info_str),
                                ("bpm", bpm_info[0][0]),
                            ]
                        )
                    )

                song_count += 1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    notes_data_df = pd.DataFrame(notes_data)
    notes_data_df["diff"] = notes_data_df["timing_msec"].diff(periods=1)
    notes_data_df["diff"].fillna(notes_data_df["timing_msec"], inplace=True)
    notes_data_df = notes_data_df.sort_values(["live_difficulty_id", "note_id"])
    notes_data_df.to_json(
        os.path.join(save_dir, "notes_data.json"), orient="records", lines=True
    )

    live_data_df = pd.DataFrame(live_data)
    live_data_df = live_data_df.sort_values(["live_difficulty_id"])
    live_data_df.to_csv(
        os.path.join(save_dir, "m_live_data.csv"), index=False, header=True
    )

    # copy and rename music files
    audio_save_dir = os.path.join(save_dir, "audio")
    os.makedirs(audio_save_dir, exist_ok=True)
    for live_id, song_path in song_paths:
        extension = song_path.split(".")[-1]
        shutil.copy(song_path, os.path.join(audio_save_dir, live_id + "." + extension))
