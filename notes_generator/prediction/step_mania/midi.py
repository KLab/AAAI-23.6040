"""Convert generated score of stepmania to midi format and write down.
"""

from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import mido
import pandas as pd
from mido import Message, MetaMessage, MidiFile, MidiTrack

from notes_generator.constants import SMNotesType, STEPMANIA_MIDI_RESOLUTION


class Note(NamedTuple):
    tap_time: int
    track_index: int
    is_long_head: bool
    is_long_tail: bool


class BpmInfo(NamedTuple):
    bpm: Union[int, float]
    msec: int
    beat: int

    @staticmethod
    def from_tuple(bpm_infos: List[Tuple[Union[int, float], float, int]]):
        return [BpmInfo(*e) for e in bpm_infos]


class BpmChange(NamedTuple):
    msec: int
    bpm: float


class BeatChange(NamedTuple):
    msec: int
    beat: int


class EventType(Enum):
    on = 1
    off = 0


class NoteEvent(NamedTuple):
    absolute_tick: int
    note_number: int  # midi note number
    event_type: EventType
    is_long: bool
    notes_id: int


def decompose_bpminfo(bpm_infos: List[BpmInfo]) -> Tuple[List[BpmChange], List[BeatChange]]:
    bpm_changes, beat_changes = [], []
    bpm_before, beat_before = None, None
    bpm_infos = sorted(bpm_infos, key=lambda x: x.msec)
    # The BPM for the first bar is adjusted so that the second bar
    # is start with right place
    if bpm_infos[0].msec > 0:
        bpm_changes.append(BpmChange(0, 60 * 1000 / bpm_infos[0].msec * bpm_infos[0].beat))
    for bpm_info in bpm_infos:
        msec, bpm, beat = bpm_info.msec, bpm_info.bpm, bpm_info.beat
        if bpm_before is None or bpm != bpm_before:
            bpm_changes.append(BpmChange(msec, bpm))
        if beat_before is None or beat != beat_before:
            beat_changes.append(BeatChange(msec, beat))
        bpm_before, beat_before = bpm, beat

    return bpm_changes, beat_changes


def msecs2ticks(msecs: List[int], bpm_changes: List[BpmChange], resolution: int) -> List[int]:
    assert msecs == sorted(msecs), "msecs is not sorted in ascending order."

    # Convert given msec to the tick from head of the song
    abs_ticks = []
    msecs_index = 0
    msec_before = 0
    tick_before = 0
    for i, bpm_change in enumerate(bpm_changes):
        if i == len(bpm_changes) - 1:
            while msecs_index < len(msecs):
                delta_msec = msecs[msecs_index] - msec_before
                delta_tick = int(
                    round(
                        mido.second2tick(
                            delta_msec / 1000,
                            resolution,
                            mido.bpm2tempo(bpm_change.bpm),
                        )
                    )
                )
                abs_ticks.append(tick_before + delta_tick)
                msecs_index += 1
                msec_before += delta_msec
                tick_before += delta_tick
        else:
            next_bpm_change = bpm_changes[i + 1]
            while msecs[msecs_index] < next_bpm_change.msec and msecs_index < len(msecs):
                delta_msec = msecs[msecs_index] - msec_before
                delta_tick = int(
                    round(
                        mido.second2tick(
                            delta_msec / 1000,
                            resolution,
                            mido.bpm2tempo(bpm_change.bpm),
                        )
                    )
                )
                abs_ticks.append(tick_before + delta_tick)
                msecs_index += 1
                msec_before += delta_msec
                tick_before += delta_tick

            delta_msec = next_bpm_change.msec - msec_before
            delta_tick = int(
                round(
                    mido.second2tick(delta_msec / 1000, resolution, mido.bpm2tempo(bpm_change.bpm))
                )
            )
            msec_before += delta_msec
            tick_before += delta_tick

    return abs_ticks


def create_metatrack(
    live_id: int, bpm_changes: List[BpmChange], beat_changes: List[BeatChange], resolution: int
) -> MidiTrack:
    metatrack = MidiTrack()
    metatrack.name = str(live_id)

    meta_messages = sorted(bpm_changes + beat_changes, key=lambda x: x.msec)
    msec_before = meta_messages[0].msec
    current_tempo = mido.bpm2tempo(bpm_changes[0].bpm)
    for meta_m in meta_messages:
        time_diff = (meta_m.msec - msec_before) / 1000
        if isinstance(meta_m, BpmChange):
            metatrack.append(
                MetaMessage(
                    "set_tempo",
                    tempo=mido.bpm2tempo(meta_m.bpm),
                    time=round(mido.second2tick(time_diff, resolution, current_tempo)),
                )
            )
            current_tempo = mido.bpm2tempo(meta_m.bpm)
        if isinstance(meta_m, BeatChange):
            metatrack.append(
                MetaMessage(
                    "time_signature",
                    numerator=meta_m.beat,
                    denominator=4,
                    time=round(mido.second2tick(time_diff, resolution, current_tempo)),
                )
            )
        msec_before = meta_m.msec

    return metatrack


def write_track(note_events: List[NoteEvent], track: MidiTrack) -> MidiTrack:
    current_tick = 0
    lane_stack = defaultdict(list)
    for i, note_event in enumerate(note_events):
        delta_tick = note_event.absolute_tick - current_tick
        if note_event.event_type == EventType.on:
            # Assign notes number for a purpose of examination
            # notes_id: nth on master data、midi_id: nth on MIDI file
            # The index begins from 1
            assert not lane_stack[note_event.note_number], (
                f"[notes_id={note_event.notes_id}][midi_id=({i + 1})]"
                f"[note_number={note_event.note_number}] New note has begun before the previous note ends"
            )
            track.append(
                Message("note_on", note=note_event.note_number, velocity=100, time=delta_tick)
            )
            lane_stack[note_event.note_number].append(note_event)
        else:
            assert len(lane_stack[note_event.note_number]) == 1, (
                f"[notes_id={note_event.notes_id}][midi_id=({i + 1})]"
                f"[note_number={note_event.note_number}] The note ended before the note begun."
            )
            track.append(Message("note_off", note=note_event.note_number, time=delta_tick))
            lane_stack[note_event.note_number].pop()

        current_tick += delta_tick

    return track


def _create_notes_data_for_track(notes, note_ticks, fn_calc_note_number):
    lane_stack = defaultdict(list)
    notes_data_list = []
    next_notes: Dict[int, int] = dict()
    for i, (note, abs_tick) in enumerate(zip(notes, note_ticks)):
        note_number = fn_calc_note_number(note)
        if lane_stack[note_number]:
            prev_notes_data = lane_stack[note_number].pop(-1)
            prev_id = prev_notes_data[0]
            # Next note on the same lane
            next_notes[prev_id] = i
        notes_data = (i, note, note_number, abs_tick)
        lane_stack[note_number].append(notes_data)
        notes_data_list.append(notes_data)
    return notes_data_list, next_notes


def create_track(
    notes: List[Note],
    track_name: str,
    bpm_changes: List[BpmChange],
    resolution: int,
    fn_calc_note_number: Callable,
) -> MidiTrack:
    default_tap_duration = 120  # ticks
    track = mido.MidiTrack()
    track.name = track_name
    # Get the notes' timing in tick
    notes = sorted(notes, key=lambda x: x.tap_time)
    note_msecs = [n.tap_time for n in notes]
    note_ticks = msecs2ticks(note_msecs, bpm_changes, resolution)
    notes_data_list, next_notes = _create_notes_data_for_track(
        notes, note_ticks, fn_calc_note_number
    )
    # Convert all notes to NoteEvent instance and sort by timing.
    # Handle long notes and normal notes in an integrated format.
    note_events = []  # type: List[NoteEvent]
    for i, note, note_number, abs_tick in notes_data_list:
        next_note_id = next_notes.get(i)
        if not any([note.is_long_head, note.is_long_tail]):
            tap_duration = default_tap_duration
            if next_note_id:
                next_diff = notes_data_list[next_note_id][-1] - abs_tick
                # When the distance of two notes on the same lane is short,
                # shorten the tap_duration
                while tap_duration >= next_diff:
                    tap_duration = tap_duration // 2
            note_events.append(NoteEvent(abs_tick, note_number, EventType.on, False, i))
            note_events.append(
                NoteEvent(abs_tick + tap_duration, note_number, EventType.off, False, i)
            )
        elif note.is_long_head:
            note_events.append(NoteEvent(abs_tick, note_number, EventType.on, True, i))
        elif note.is_long_tail:
            note_events.append(NoteEvent(abs_tick, note_number, EventType.off, True, i))
    note_events = sorted(note_events, key=lambda x: x.absolute_tick)

    # write down to midi track
    track = write_track(note_events, track)

    return track


def create_midi(
    live_id: int,
    scores: Dict[int, List[Note]],
    bpm_infos: List[BpmInfo],
    resolution: int,
    fn_calc_note_number: Callable,
) -> MidiFile:
    mid = MidiFile(ticks_per_beat=resolution)
    bpm_changes, beat_changes = decompose_bpminfo(bpm_infos)
    metatrack = create_metatrack(live_id, bpm_changes, beat_changes, resolution)
    mid.tracks.append(metatrack)

    for live_diff_id in sorted(scores.keys()):
        print(live_diff_id)
        score = scores[live_diff_id]
        track = create_track(
            score,
            track_name=str(live_diff_id),
            bpm_changes=bpm_changes,
            resolution=resolution,
            fn_calc_note_number=fn_calc_note_number,
        )
        mid.tracks.append(track)

    return mid


def calc_note_number(note: Note) -> int:
    """Decide note number

    track_index are 0, 4, 8, 12
    corresponding musical scales are C1, C2, C3, C4
    corresponding note numbers are 36, 48, 60, 72

    Parameters
    ----------
    note

    Returns
    -------

    """
    track_index = note.track_index
    base_note_number = 36
    return int(base_note_number + 12 * track_index)


def note_type_flags(note_type):
    if note_type == SMNotesType.normal.value:
        is_long_head, is_long_tail = False, False
    elif note_type == SMNotesType.hold_roll_head.value:
        is_long_head, is_long_tail = True, False
    elif note_type == SMNotesType.hold_roll_tail.value:
        is_long_head, is_long_tail = False, True
    else:
        raise RuntimeError(f"Invalid note_type: {note_type}")
    return is_long_head, is_long_tail


def create_from_dataframe(
    df: pd.DataFrame,
    bpm_info: List[Tuple[Union[int, float], float, int]],
    live_id: int,
) -> MidiFile:
    live_difficulty_ids = sorted(df["live_notes_id"].unique())
    scores = dict()
    for live_diff_id in live_difficulty_ids:
        df_notes_subset = df[df["live_notes_id"] == live_diff_id]
        notes = [
            Note(r["tap_time"], r["track_index"], r["is_long_head"], r["is_long_tail"])
            for i, r in df_notes_subset.iterrows()
        ]
        scores[live_diff_id] = notes
    bpm_infos = BpmInfo.from_tuple(bpm_info)
    return create_midi(live_id, scores, bpm_infos, STEPMANIA_MIDI_RESOLUTION, calc_note_number)
