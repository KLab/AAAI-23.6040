"""The beat guide proposed in our paper
"""
import bisect
import enum
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np

from notes_generator.constants import FRAME


class TimeUnit(enum.Enum):
    milliseconds = "milliseconds"
    frames = "frames"
    seconds = "seconds"


def convert_units(ms: int, units: TimeUnit) -> float:
    if units == TimeUnit.milliseconds:
        return float(ms)
    elif units == TimeUnit.seconds:
        return float(ms) / 1000.0
    elif units == TimeUnit.frames:
        return float(ms) / 32.0


def validate(bpm_info: List[Tuple[float, float, int]], mel_length: int):
    for x, (bpm, start, beat) in enumerate(bpm_info):
        assert bpm > 0
        assert start >= 0
        assert beat > 0
        assert (
            np.round(start / FRAME) <= mel_length
        ), f"The start position of bpm_info ({start}) is outside the song length ({mel_length})."


def process_bpminfo(
    bpm_info: List, max_length_ms: int, units: TimeUnit = TimeUnit.milliseconds
) -> Tuple[List, List, List, List]:
    bpms, starts, ends, beats = [], [], [], []
    max_length = convert_units(max_length_ms, units)
    for i, (bpm, start, beat) in enumerate(bpm_info):
        start = convert_units(start, units)
        if start > max_length:
            continue
        if i < len(bpm_info) - 1:
            bpms.append(bpm)
            starts.append(start)
            next_start = convert_units(bpm_info[i + 1][1], units)
            ends.append(min(next_start, max_length))
            beats.append(beat)
        else:
            bpms.append(bpm)
            starts.append(start)
            ends.append(max_length)
            beats.append(beat)
    return bpms, starts, ends, beats


def gen_beats_array(
    length: int,
    bpm_info: List[Tuple[float, float, int]],
    mel_length: int,
    distinguish_downbeat: bool = True,
):
    """

    Parameters
    ----------
    length : int
        length of onset label sequence
    bpm_info : List[Tuple[float, float, int]]
        list of tuple (bpm, start(ms), beats)
        If there are tempo changes during the song, bpm_info contains
        more than one tuples.
    distinguish_downbeat : bool
        If `True`, distinguish downbeat of a measure from other beats.
    mel_length : int
        length of mel spectrogram sequence

    Returns
    -------

    """
    validate(bpm_info, mel_length)

    # The time range that nth(>= 0) frame represents is
    #   (n - 0.5) * FRAME < time <= (n + 0.5) * FRAME [ms]
    # , where `FRAME` is a constant for frame length.
    # Set an upper bound of the time for last frame (n = length - 1)
    max_length_ms = max(0, (length - 0.5) * FRAME)
    arr = np.zeros(length)
    bpms, starts, ends, beats = process_bpminfo(bpm_info, max_length_ms)
    for i, (bpm, start, end, beat) in enumerate(zip(bpms, starts, ends, beats)):
        # calc beat timing in milliseconds
        interval = 60 * 1000 / bpm  # ms
        beats_timing = np.arange(start, end, interval)

        # convert to array index
        # The conversion method (float -> int) is taking round, which is
        # the same as the preprocessing method for onset.
        # (in preprocessing/onset_converter.py)
        arg_beats = np.round(beats_timing / FRAME).astype("int64")

        # assign flag to beats array
        for b in range(beat):
            if b == 0:  # downbeat (beginning of a measure)
                if distinguish_downbeat:
                    arr[arg_beats[b::beat]] = 2
                else:
                    arr[arg_beats[b::beat]] = 1
            else:  # other beats
                arr[arg_beats[b::beat]] = 1

    return arr.reshape([-1, 1])


def get_beat_list(bpm_info: List, max_length_ms: int, units: TimeUnit = TimeUnit.milliseconds):
    """
    Return the lists containing beats.

    All lists are same lengths. The ith element of a list holds ith beat information.

    usage:

    >>> get_beat_list([[120, 0, 4]], 4000)   # BPM120。500ms
    ([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0],
     [1, 2, 3, 4, 1, 2, 3, 4],
     [120, 120, 120, 120, 120, 120, 120, 120],
     [1, 1, 1, 1, 2, 2, 2, 2])

    Parameters
    ----------
    bpm_info
    max_length_ms
    units

    Returns
    -------
        time_list: The list containing the start position of the beat.
            (The unit is one specified at 'units' parameter.)
        beat_list: The list containing beat number
        bpm_list: BPM
        bar_list: The list containing bar number.
    """
    bpms, starts, ends, beats = process_bpminfo(bpm_info, max_length_ms, units)
    beat_list = []
    time_list = []
    bpm_list = []
    bar_list = []
    bar_nth = 0
    for (bpm, start, end, beat) in zip(bpms, starts, ends, beats):
        beat_duration = convert_units((60.0 / bpm) * 1000.0, units)
        beat_num = 0
        for b in np.arange(start, end, beat_duration):  # iterate over each beat
            bpm_list.append(bpm)
            time_list.append(b)
            beat_list.append(beat_num + 1)
            bar_list.append(bar_nth + 1)
            beat_num += 1
            beat_num = beat_num % beat
            if beat_num == 0:
                bar_nth += 1
        if beat_num != 0:
            bar_nth += 1
    return time_list, beat_list, bpm_list, bar_list


def get_score_beats(
    score_positions,
    bpm_info,
    max_length_ms,
    units: TimeUnit = TimeUnit.frames,
    max_distance: int = 2,
) -> List[Tuple[int, Tuple]]:
    """
    Get the beat numbers where notes exist in each bars.

    Usage::

    >>> import numpy as np
    >>> dummy_score = np.zeros((30, 1))
    >>> dummy_score[0, 0] = 1
    >>> dummy_score[20, 0] = 1
    >>> score_pos = np.nonzero(dummy_score[:, 0])[0]
    >>> score_pos
    array([0, 20])
    >>> get_score_beats(score_pos, [[187, 0, 4]], 30 * 32)  # notes exist on 1st and 3rd beats for 1st bar.
    [(1, [1, 3])]

    Parameters
    ----------
    score_positions: np.ndarray: The positions of notes(unit is e.g. frame)。
        The values are not raw score but returned value of np.nonzero().
    bpm_info
    max_length_ms
    units: The unit of values in the list `score_positions`. Defaults to `TimeUnit.frames`.
    max_distance: The tolerance of a distance from notes to beats.

    Returns
    -------
        bar_beats_list: List[Tuple[int, List]] Bar number and list of beats where notes exist.
    """
    bpm_pos_list, beat_list, bpm_list, bar_list = get_beat_list(
        bpm_info, max_length_ms, units=units
    )
    # Store notes positions in each bars
    bar_beats_dict = defaultdict(list)

    # Compare positions of each notes with beat positions
    for idx, notes_pos in enumerate(score_positions):
        # Get the closest beat which is larger than notes_pos
        pos = bisect.bisect_left(bpm_pos_list, notes_pos)
        if pos < len(bpm_pos_list) and abs(bpm_pos_list[pos] - notes_pos) <= max_distance:
            # When a note is on pos
            fr_pos = pos
        elif pos > 0 and abs(bpm_pos_list[pos - 1] - notes_pos) <= max_distance:
            # When a note is on one beat before pos
            fr_pos = pos - 1
        else:
            # When a note is not on beat.
            fr_pos = -1

        if fr_pos >= 0:
            beat = beat_list[fr_pos]
            bar = bar_list[fr_pos]
        else:
            beat = beat_list[pos]
            bar = bar_list[pos]
            bpm = bpm_list[pos]
            # If a note is between beats, store float.
            beat_duration = convert_units((60 / bpm) * 1000.0, units)
            beat = beat - (abs(bpm_pos_list[pos] - notes_pos) / beat_duration)
        bar_beats_dict[bar].append(beat)
    bar_beats_list: List[Tuple[int, Tuple]] = []
    for bar, beats in bar_beats_dict.items():
        bar_beats_list.append((bar, tuple(beats)))
    bar_beats_list.sort(key=lambda x: x[0])
    return bar_beats_list


def fill_beat_list(bar_beat_list):
    """Pad the bars where no notes exist.
    Parameters
    ----------
    bar_beat_list

    Returns
    -------

    """
    max_bar = bar_beat_list[-1][0]
    bar_dic = {bar: in_bar for bar, in_bar in bar_beat_list}
    return_values = []
    for bar in range(1, max_bar + 1):
        if bar in bar_dic:
            in_bar = bar_dic[bar]
            return_values.append(tuple(in_bar))
        else:
            return_values.append(())
    return return_values


def format_beats(beats) -> str:
    """Format beats list for display.
    Parameters
    ----------
    beats

    Returns
    -------

    """
    if not beats:
        return ""
    displays = []
    for b in beats:
        if isinstance(b, float):
            displays.append(f"{b:.1f}")
        else:
            displays.append(str(b))
    return " ".join(displays)


def count_bar_beats(bar_bpm_list):
    """Aggregate bar_beat_list。

    Parameters
    ----------
    bar_bpm_list

    Returns
    -------

    """
    stat = Counter()
    logs = []
    for bar, beats in enumerate(bar_bpm_list):
        f_beats = format_beats(beats)
        logs.append((bar, f_beats))
        stat[f_beats] += 1
    return logs, stat
