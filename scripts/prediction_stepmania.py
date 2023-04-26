import argparse
import json
import logging
import tempfile
from ast import literal_eval
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from notes_generator.constants import ConvStackType, NMELS
from notes_generator.models.onsets import SimpleOnsets
from notes_generator.prediction.predictor import Predictor, SMPredictorDDC
from notes_generator.prediction.step_mania.midi import (
    create_from_dataframe as create_midi_stepmania,
)
from notes_generator.preprocessing import mel

logger = getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def preprocess_audio(
    audio_path,
    live_id,
    bpm_info,
):
    with tempfile.TemporaryDirectory() as tempdir:
        mel.convert(
            audio_path.name,
            live_id,
            audio_path.parent,
            Path(tempdir),
            aug_count=0,
            bpm_info=bpm_info,
        )
        mel_data, mel_meta = _load_mel(Path(tempdir) / "mel.npz")
    return mel_data, mel_meta


def prediction_main(
    onset_model_path: Path,
    audio_path: Path,
    live_id: int,
    midi_save_path: Path,
    bpm_info: List[Tuple],
    device: str = "cpu",
    inference_chunk_length: int = 640,
):
    mel_data, mel_meta = preprocess_audio(audio_path, live_id, bpm_info)

    onset_model = SimpleOnsets(
        NMELS,
        1,
        enable_condition=True,
        enable_beats=True,
        conv_stack_type=ConvStackType.v7,
        num_layers=2,
        onset_weight=64,
        dropout=0.5,
        inference_chunk_length=inference_chunk_length,
    )
    onset_model.load_state_dict(
        torch.load(str(onset_model_path), map_location=torch.device(device))
    )
    sym_model = None
    ddc_predictor = SMPredictorDDC(onset_model, sym_model, device)
    midi_path = midi_save_path / f"{live_id}.mid"
    logger.info("prediction start")
    predict_and_save(ddc_predictor, mel_data, mel_meta, live_id, midi_path)
    return


def predict_and_save(
    predictor: Predictor,
    mel: np.ndarray,
    meta_data: Dict,
    live_id: int,
    midi_save_path: Path,
):
    bpm_info = meta_data["bpm_info"]
    scores_dict, probs = predictor.predict_all(mel, bpm_info)
    logger.info("prediction complete")

    # add live_notes_id to all songs (an unique ID for live_id and difficulty pair)
    scores = []
    for difficulty in sorted(predictor.difficulties):
        # ex: live_id=1001, difficulty=20 -> live_diff_id = 10012
        live_diff_id = str(live_id) + str(difficulty)[0]
        for note in scores_dict[difficulty]:
            note["live_notes_id"] = live_diff_id
        scores += scores_dict[difficulty]

    df_notes = pd.DataFrame.from_records(scores)
    df_notes = df_notes[
        ["live_notes_id", "tap_time", "track_index", "is_long_head", "is_long_tail"]
    ]
    midi = create_midi_stepmania(df_notes, bpm_info, live_id)

    logger.info(f"midi save to {str(midi_save_path)}")
    # create midi
    midi.save(str(midi_save_path))


def _load_mel(mel_path: Path):
    with mel_path.open("rb") as fp:
        data = np.load(fp)
        mel = data["mel"]
    meta_path = mel_path.parent / "meta.json"
    with meta_path.open() as fp:
        meta_data = json.load(fp)
    return mel, meta_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset_model_path", type=Path, required=True)
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--midi_save_path", type=Path)
    parser.add_argument("--bpm_info", type=str)
    parser.add_argument("--inference_chunk_length", type=int, default=640)
    args = parser.parse_args()
    onset_model_path = Path(args.onset_model_path)
    assert onset_model_path.exists()
    audio_path = Path(args.audio_path)
    assert audio_path.exists()
    midi_save_path = Path(args.midi_save_path)
    if not midi_save_path.exists():
        midi_save_path.mkdir()
    bpm_info = literal_eval(args.bpm_info)
    prediction_main(
        onset_model_path,
        audio_path,
        live_id=0,
        midi_save_path=midi_save_path,
        bpm_info=bpm_info,
        inference_chunk_length=args.inference_chunk_length,
    )
