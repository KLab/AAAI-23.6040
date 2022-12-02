import pandas as pd

_EPSILON = 1e-6


def bpm_to_spb(bpm):
    return 60.0 / bpm


def calc_segment_lengths(bpms):
    assert len(bpms) > 0
    segment_lengths = []
    for i in range(len(bpms) - 1):
        spb = bpm_to_spb(bpms[i][1])
        segment_lengths.append(spb * (bpms[i + 1][0] - bpms[i][0]))
    return segment_lengths


def calc_abs_for_beat(offset, bpms, stops, segment_lengths, beat):
    bpm_idx = 0
    while bpm_idx < len(bpms) and beat + _EPSILON > bpms[bpm_idx][0]:
        bpm_idx += 1
    bpm_idx -= 1

    stop_len_cumulative = 0.0
    for stop_beat, stop_len in stops:
        diff = beat - stop_beat
        # We are at this stop which should not count to its timing
        if abs(diff) < _EPSILON:
            break
        # We are before this stop
        elif diff < 0:
            break
        # We are above this stop
        else:
            stop_len_cumulative += stop_len

    full_segment_total = sum(segment_lengths[:bpm_idx])
    partial_segment_spb = bpm_to_spb(bpms[bpm_idx][1])
    partial_segment = partial_segment_spb * (beat - bpms[bpm_idx][0])

    return full_segment_total + partial_segment - offset + stop_len_cumulative


def calc_note_beats_and_abs_times(offset, bpms, stops, note_data):
    segment_lengths = calc_segment_lengths(bpms)

    # copy bpms
    bpms = bpms[:]
    inc = None
    inc_prev = None
    time = offset

    # beat loop
    note_beats_abs_times = []
    beat_times = []
    for measure_num, measure in enumerate(note_data):
        ppm = len(measure)
        for i, code in enumerate(measure):
            beat = measure_num * 4.0 + 4.0 * (float(i) / ppm)
            # TODO: This could be much more efficient but is not the bottleneck for the moment.
            beat_abs = calc_abs_for_beat(offset, bpms, stops, segment_lengths, beat)
            note_beats_abs_times.append(((measure_num, ppm, i), beat, beat_abs, code))
            beat_times.append(beat_abs)

    # handle negative stops
    beat_time_prev = float('-inf')
    del_idxs = []
    for i, beat_time in enumerate(beat_times):
        if beat_time_prev > beat_time:
            del_idxs.append(i)
        else:
            beat_time_prev = beat_time
    for del_idx in sorted(del_idxs, reverse=True):
        del note_beats_abs_times[del_idx]
        del beat_times[del_idx]

    # TODO: remove when stable
    assert sorted(beat_times) == beat_times

    return note_beats_abs_times


def calc_bpm_info(offset, bpms, stops, time_sigs):
    segment_lengths = calc_segment_lengths(bpms)

    # copy bpms
    bpms = bpms[:]

    if time_sigs is not None:
        bpm_df = pd.DataFrame(bpms, columns=['beat', 'bpm'])
        time_sig_df = pd.DataFrame(time_sigs, columns=['beat', 'time_signature'])
        time_sig_df[["numerator", "denominator"]] = pd.DataFrame(
            time_sig_df['time_signature'].tolist())
        time_sig_df["regularized_numerator"] = 4 // time_sig_df[
            "denominator"] * time_sig_df["numerator"]
        bpm_df = pd.merge(bpm_df, time_sig_df, on="beat", how="outer").sort_values(
            by=['beat']).fillna(method='ffill')
        bpm_df = bpm_df[["beat", "bpm", "regularized_numerator"]].copy()
        bpms, time_sigs = [], []
        for i, (beat, bpm, sig) in bpm_df.iterrows():
            bpms.append((beat, bpm))
            time_sigs.append(sig)
    else:
        time_sigs = [4] * len(bpms)

    # beat loop
    bpm_infos = []
    beat_times = []
    for (beat, bpm), sig in zip(bpms, time_sigs):
        beat_abs = calc_abs_for_beat(offset, bpms, stops, segment_lengths,
                                     beat)
        # ensure beat_abs is positive
        while beat_abs < 0:
            beat_abs += sig * bpm_to_spb(bpm)
        beat_abs_ms = round(beat_abs * 1000)
        bpm_infos.append((bpm, beat_abs_ms, int(sig)))
        beat_times.append(beat_abs)

    # handle negative stops
    beat_time_prev = float('-inf')
    del_idxs = []
    for i, beat_time in enumerate(beat_times):
        if beat_time_prev > beat_time:
            del_idxs.append(i)
        else:
            beat_time_prev = beat_time
    for del_idx in sorted(del_idxs, reverse=True):
        del bpm_infos[del_idx]
        del beat_times[del_idx]

    # TODO: remove when stable
    assert sorted(beat_times) == beat_times

    return bpm_infos
