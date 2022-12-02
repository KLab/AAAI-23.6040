import logging
from logging import getLogger
from typing import Callable, Dict, List, Optional
from typing import Tuple, Union

import numpy as np
import torch

from notes_generator.constants import (
    FRAME,
    MAX_THRESHOLD,
    SMDifficultyType,
    SMNotesType,
    sm_init_threshold,
    sm_max_notes,
    sm_min_note_distance,
)
from notes_generator.prediction.onset_prediction import predict as onset_predict, predict_proba

logger = getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Predictor:
    def predict_all(self, mel: np.array, bpm_info: Optional[List] = None):
        raise NotImplementedError

    def predict(self, mel: np.array, condition: int):
        raise NotImplementedError

    @property
    def difficulties(self):
        raise NotImplementedError


class PredictorDDC(Predictor):
    def __init__(self, onset_model: torch.nn.Module, sym_model: torch.nn.Module, device):
        self.onset_model = onset_model
        self.sym_model = sym_model
        self.device = device

    @property
    def min_note_distance(self):
        raise NotImplementedError

    @property
    def max_threshold(self):
        raise NotImplementedError

    @property
    def init_threshold(self):
        raise NotImplementedError

    @property
    def sym_predict(self) -> Callable:
        raise NotImplementedError

    def predict_onsets_proba(
        self,
        mel: np.array,
        condition: int,
        bpm_info: Optional[List[Tuple[Union[float, int], int, int]]] = None,
    ):
        return predict_proba(self.onset_model, mel, condition, bpm_info, self.device)

    def predict_onsets(
        self,
        mel: np.array,
        condition: int,
        threshold: Optional[float] = None,
        bpm_info: Optional[List[Tuple[Union[float, int], int, int]]] = None,
    ):
        # predict notes timings
        if threshold is None:
            sequence, probs = onset_predict(
                self.onset_model,
                mel,
                condition,
                threshold=self.init_threshold[condition],
                bpm_info=bpm_info,
                device=self.device,
            )

        else:
            sequence, probs = onset_predict(
                self.onset_model,
                mel,
                condition,
                device=self.device,
                threshold=threshold,
                bpm_info=bpm_info,
            )
        logger.info("predict onset complete")
        sequence = sequence.nonzero()[0]
        sequence = filter_overwrap(sequence, probs, self.min_note_distance[condition])
        if threshold is None:
            songlen = mel.shape[0] * FRAME  # ms
            sequence = self._adjust_threshold(sequence, probs, condition, songlen, bpm_info)
        return sequence, probs

    def _adjust_threshold(
        self,
        sequence: np.ndarray,
        probs: np.ndarray,
        condition: int,
        songlen: int,
        bpm_info: List,
    ) -> np.ndarray:
        raise NotImplementedError

    def predict_sym(self, sequence, condition, bpm_info=None):
        notes_sequence, sym_probs = self.sym_predict(
            self.sym_model, sequence, condition, self.device
        )
        logger.info("predict symbol complete")
        # timing(ms)、note type(left and right lane)
        # [[ 2144     6     4]
        #  [ 4896     0     1]
        # ...
        notes_sequence = filter_notes(notes_sequence, sym_probs)
        return notes_sequence

    def predict(
        self,
        mel: np.array,
        condition: int,
        threshold: Optional[float] = None,
        bpm_info: Optional[List[Tuple[Union[float, int], int, int]]] = None,
    ):
        """Predict from the beginning"""
        # predict notes timing
        sequence, probs = self.predict_onsets(mel, condition, threshold, bpm_info)
        # predict notes type
        notes_sequence = self.predict_sym(sequence, condition, bpm_info)
        return notes_sequence, probs


class SMPredictorDDC(PredictorDDC):
    """Combine onset and sym model"""

    @property
    def difficulties(self):
        return (d.value for d in SMDifficultyType)

    @property
    def min_note_distance(self):
        return sm_min_note_distance

    @property
    def max_threshold(self):
        return MAX_THRESHOLD

    @property
    def init_threshold(self) -> Dict:
        return sm_init_threshold

    def _adjust_threshold(
        self,
        sequence: np.ndarray,
        probs: np.ndarray,
        condition: int,
        songlen: int,
        bpm_info: List,
    ) -> np.ndarray:
        assert bpm_info is not None, "bpm_info must be provided"
        max_notes = sm_max_notes[condition]
        sequence, _ = adjust_threshold(
            sequence,
            probs,
            max_notes,
            self.init_threshold[condition],
            self.max_threshold,
        )
        return sequence

    def predict_sym(self, sequence, condition, bpm_info=None):
        # notes_sequence:
        #   timing(ms), note type for each lane
        #   [[ 2144     0     1     0     0]
        #    [ 4896     0     1     1     0]
        #   ...
        # notes_sequence, sym_probs = self.sym_predict(
        #     self.sym_model, sequence, condition, self.device
        # )
        notes_sequence = np.zeros((sequence.shape[0], 5), dtype=np.int)
        for i, x in enumerate(sequence[:-1]):
            # In this demo, we focus on notes timing
            # so we place all notes in the first lane
            notes_sequence[i] = np.array([int(x * FRAME), 1, 0, 0, 0])
        logger.info("predict symbol complete")
        return notes_sequence

    def predict_all(self, mel: np.array, bpm_info: Optional[List] = None):
        scores = dict()
        probs = dict()
        for difficulty in self.difficulties:
            score_, prob_ = self.predict(mel, difficulty, bpm_info=bpm_info)
            scores[difficulty] = _convert_to_csv_type_stepmania(score_)
            probs[difficulty] = prob_
            logger.info(f"predicted notes: {len(scores[difficulty])} difficulty: {difficulty}")
        return scores, probs


def _convert_to_csv_type_stepmania(score: np.ndarray) -> List[Dict]:
    times, rails_array = score[:, 0], score[:, 1:]
    score_converted = []

    def convert(note, time) -> Dict:
        rail, note_type = note
        track_index = rail * 4
        is_long_head, is_long_tail = False, False
        if note_type == SMNotesType.hold_roll_head.value:
            is_long_head = True
        elif note_type == SMNotesType.hold_roll_tail.value:
            is_long_tail = True
        return {
            "tap_time": time,
            "track_index": track_index,
            "is_long_head": is_long_head,
            "is_long_tail": is_long_tail,
        }

    for time, rails in zip(times, rails_array):
        notes = [(rail, notetype) for rail, notetype in enumerate(rails) if notetype != 0]
        conv_notes = [convert(note, time) for note in notes]
        score_converted += conv_notes

    return score_converted


def filter_overwrap(sequence, probs, min_distance):
    """Filter nearby notes
    Parameters
    ----------
    sequence
    probs
    min_distance

    Returns
    -------

    """
    prev_frame = 0
    prev_prob = None
    excludes = set()
    for frame in sequence:
        prob = probs[frame]
        if prev_prob and (frame - prev_frame) <= min_distance:
            if prob > prev_prob:
                # logger.info(f'exclude overwrap {prev_frame} {frame} {float(prev_prob):.3f} < {float(prob):.3f}')
                excludes.add(prev_frame)
            else:
                # logger.info(f'exclude overwrap {prev_frame} {frame} {float(prev_prob):.3f} > {float(prob):.3f}')
                excludes.add(frame)
        prev_frame = frame
        prev_prob = prob
    logger.info(f"exclude overwrap {len(excludes)}")
    return np.array([frame for frame in sequence if frame not in excludes])


def filter_by_threshold(sequence, probs, threshold):
    """Filter notes whose probabilities are below threshold"""
    probs_ = probs[sequence].flatten()
    filtered_seq = sequence[np.argwhere(probs_ >= threshold).flatten()]
    return np.sort(filtered_seq)


def adjust_threshold(onset_seq, probs, max_notes, init_threshold, max_threshold):
    """Adjust a threshold so that the notes number is within acceptable limits."""
    # Start from small value, then gradually raise the threshold
    # until the number of notes falls below the limit
    threshold_ = init_threshold
    while len(onset_seq) > max_notes and threshold_ <= max_threshold:
        threshold_ += 0.05
        onset_seq = filter_by_threshold(onset_seq, probs, threshold_)
    logger.info(f"threshold was chosen to be {threshold_}")
    return onset_seq, threshold_


def filter_notes(notes_sequence, probs):
    """Filter irregular notes
    Parameters
    ----------
    notes_sequence

    Returns
    -------

    """
    flags = [None, False, False]
    return_values = []
    for idx in range(len(notes_sequence)):
        # Delete inconsistent long notes
        val = notes_sequence[idx]
        msec, lnotes, rnotes = val
        if msec == 0:
            continue
        lnotes = rewrite_long_notes(1, idx, notes_sequence, probs, flags)
        rnotes = rewrite_long_notes(2, idx, notes_sequence, probs, flags)
        # lnotes = handle_long_notes(1, idx, notes_sequence, flags)
        # rnotes = handle_long_notes(2, idx, notes_sequence, flags)
        return_values.append((msec, lnotes, rnotes))
    return return_values


def rewrite_long_notes(rail, cur_idx, values, probs, flags):
    """Rewrite long notes to be consistent
    Parameters
    ----------
    rail
    cur_idx
    values
    probs
    flags

    Returns
    -------

    """
    notes = values[cur_idx][rail]
    if notes not in (6, 7):
        return notes
    if notes == 7:
        # In the case of "long notes end":
        # check "long notes start" exists before the note
        if flags[rail]:
            flags[rail] = False
            notes = 7
        else:
            logger.info(f"[{cur_idx}]: invalid long notes 7 {probs[cur_idx]:.2f}")
            notes = 1
        return notes

    # long notes start
    start_idx = cur_idx + 1
    end_idx = cur_idx + 7
    next_values = [
        (i, row[rail]) for i, row in zip(range(start_idx, end_idx), values[start_idx:end_idx])
    ]
    close_idx = None
    excludes = set()
    exclude_p = 0.0
    for i, val in next_values:
        if val == 7:
            close_idx = i
            break
        elif val == 0:
            continue
        else:
            # exclude inconsistent note
            excludes.add(i)
            exclude_p *= probs[i]
    if close_idx:
        long_notes_p = probs[cur_idx] * probs[close_idx]
    else:
        long_notes_p = probs[cur_idx]
    if not close_idx:
        # When "long note end" does not exist
        logger.info(f"[{cur_idx}]: invalid long notes 6 {long_notes_p:.2f}")
        values[cur_idx][rail] = 1
        notes = 1
    elif not excludes or long_notes_p > exclude_p:
        if excludes:
            # exclude inconsistent note
            logger.info(
                f"[{cur_idx}]: invalid long notes exclude middle {long_notes_p:.2f} {exclude_p:.2f}"
            )
            for idx in excludes:
                values[idx][rail] = 0
        notes = 6
        assert flags[rail] is False
        flags[rail] = True
    else:
        logger.info(
            f"[{cur_idx}]: invalid long notes prefer middle sequence {long_notes_p:.4f} {exclude_p:.4f}"
        )
        values[cur_idx][rail] = 1
        values[close_idx][rail] = 1
        notes = 1
    return notes


def current_beat_interval(msec, bpm_info):
    assert (
        msec >= bpm_info[0][1]
    ), f"A note exists before the start of bpm info (Note position: {msec}, start of bpm_info: {bpm_info[0][1]})"

    current_bpm = None
    for (bpm, msec_, _) in bpm_info:
        if msec >= msec_:
            current_bpm = bpm
        else:
            break
    return 60000 / current_bpm


def count_measures(songlen, bpm_info):
    """Count measures of the song

    Parameters
    ----------
    songlen: int
        Song length [ms]
    bpm_info: List[Tuple[Union[int, float], int, int]]

    Returns
    -------
    measure_count: int
        Count of measures

    """
    measure_count = 0
    for i, (bpm, time, beat) in enumerate(bpm_info):
        if time > songlen:
            break
        if i == len(bpm_info) - 1:
            current_bpm_length = songlen - time
        else:
            next_time = bpm_info[i + 1][1]
            current_bpm_length = min(next_time, songlen) - time
        beat_intv = current_beat_interval(time, bpm_info)
        measure_intv = beat_intv * beat
        measure_count += current_bpm_length / measure_intv

    return measure_count
