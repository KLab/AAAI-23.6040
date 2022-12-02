import sys
from collections import defaultdict
from typing import List, Type

import numpy as np
import torch
from mir_eval.onset import f_measure as evaluate_onset
from mir_eval.transcription import match_notes, precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz

from notes_generator.constants import *
from notes_generator.models.onsets import OnsetsBase

eps = sys.float_info.epsilon
MIN_MIDI = 21


def extract_notes(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return
    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]
    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def to_onehot_score(score_l, score_r, device):
    onehot_l = torch.zeros((score_l.shape[0], score_l.shape[1], 8)).long().to(device)
    onehot_l = onehot_l.scatter(2, score_l.unsqueeze(-1), 1)[:, :, 1:]
    onehot_r = torch.zeros((score_r.shape[0], score_r.shape[1], 8)).long().to(device)
    onehot_r = onehot_r.scatter(2, score_r.unsqueeze(-1), 1)[:, :, 1:]
    return torch.cat([onehot_l, onehot_r], dim=-1)


def prepare_mireval(target, pred):
    """Prepare features in a form that can be used in mir_eval.

    Parameters
    ----------
    target : torch.Tensor
    pred : torch.Tensor

    Returns
    -------
    i_ref : np.ndarray
        The array containing (onset_index, offset_index) of the target.
    p_ref : np.ndarray
        The array containing bin_indices of the target.
    i_est : np.ndarray
        The array of (onset_index, offset_index) of the prediction.
    p_est : np.ndarray
        The array containing bin_indices of the prediction.

    """
    p_ref, i_ref = extract_notes(target, target)
    p_est, i_est = extract_notes(pred, pred)
    scaling = HOP_LENGTH / SAMPLE_RATE
    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
    return i_ref, p_ref, i_est, p_est


def evaluate(model, loader, device, tolerance=0.05):
    metrics = defaultdict(list)
    for batch in loader:
        pred = model.predict(batch).long()
        target = batch["onset"].long().to(device)
        if pred.shape[-1] > 1:
            target = to_onehot_score(target[:, :, 0], target[:, :, 1], device)
            pred = to_onehot_score(pred[:, :, 0], pred[:, :, 1], device)
        batch_size = batch["onset"].size(0)
        for b in range(batch_size):
            i_ref, p_ref, i_est, p_est = prepare_mireval(target[b], pred[b])
            p, r, f, o = evaluate_notes(
                i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance
            )
            metrics["metric/note/precision"].append(p)
            metrics["metric/note/recall"].append(r)
            metrics["metric/note/f1"].append(f)
            metrics["metric/note/overlap"].append(o)
            onset_ref = np.array([o[0] for o in i_ref])
            onset_est = np.array([o[0] for o in i_est])
            f_o, p_o, r_o = evaluate_onset(onset_ref, onset_est, tolerance)
            metrics["metric/onset/precision"].append(p_o)
            metrics["metric/onset/recall"].append(r_o)
            metrics["metric/onset/f1"].append(f_o)
    return metrics


def prec_rec_f1(tp, fp, fn):
    prec = 0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return prec, rec, f1


def predict_test(model: Type[OnsetsBase], loader, result_dict, device):
    for batch in loader:
        _, proba = model.predict_with_probs(batch)
        target = batch["onset"].long().to(device)
        live_ids = batch["live_id"]
        difficulties = batch["condition"]
        batch_size = batch["onset"].size(0)
        # Assume that `b` contains data of whole song.
        for b in range(batch_size):
            live_id = int(live_ids[b])
            diff = int(difficulties[b, 0])
            result_dict[live_id][diff] = {
                "proba": proba[b],
                "target": target[b],
            }
    return result_dict


def micro_metrics(
    probs: List[torch.Tensor], targets: List[torch.Tensor], threshold: float, tolerance: float
):
    """Calculate micro averaged metrics.

    Parameters
    ----------
    probs : List[torch.Tensor]
        The list containing model predictions for each charts.
    targets : List[torch.Tensor]
        The list containing ground truth labels for each charts.
    threshold : float
        The frame having probability above this threshold is considered
        that the note exists on the frame.
    tolerance : float
        The minimum tolerance for onset matching.

    Returns
    -------
        eval_results : Dict[str, float]
            The dict containing evaluated metrics.
            {metric_key: metric_value}

    Notes
    -----
    Explanation about micro average:
        Sum up TP, FP, FN for all charts,
        then calculate precision, recall, F1 at the end.

    """
    metrics = defaultdict(list)
    tp_micro, fp_micro, fn_micro = 0, 0, 0
    for prob, tgt in zip(probs, targets):
        pred = prob >= threshold
        i_ref, p_ref, i_est, p_est = prepare_mireval(tgt, pred)
        matching = match_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance
        )
        tp = len(matching)
        fp = len(p_est) - len(matching)
        fn = len(p_ref) - len(matching)
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn

    p_micro, r_micro, f_micro = prec_rec_f1(tp_micro, fp_micro, fn_micro)
    metrics["metric/note/precision.micro"].append(p_micro)
    metrics["metric/note/recall.micro"].append(r_micro)
    metrics["metric/note/f1.micro"].append(f_micro)

    # take the average
    eval_results = dict()
    for key, val in metrics.items():
        eval_results[key] = np.array(val).mean()
    return eval_results


def chart_metrics(
    probs: List[torch.Tensor], targets: List[torch.Tensor], threshold: float, tolerance: float
):
    """Calculate chart averaged metrics.

    Parameters
    ----------
    probs : List[torch.Tensor]
        The list containing model predictions for each charts.
    targets : List[torch.Tensor]
        The list containing ground truth labels for each charts.
    threshold : float
        The frame having probability above this threshold is considered
        that the note exists on the frame.
    tolerance : float
        The minimum tolerance for onset matching.

    Returns
    -------
        eval_results : Dict[str, float]
            The dict containing evaluated metrics.
            {metric_key: metric_value}

    Notes
    -----
    Explanation aboun chart average:
        Calculate precision, recall, F1 for each chart,
        then take an average of these metrics as final result.

    """
    metrics = defaultdict(list)
    for prob, tgt in zip(probs, targets):
        pred = prob > threshold
        i_ref, p_ref, i_est, p_est = prepare_mireval(tgt, pred)
        matching = match_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance
        )
        tp = len(matching)
        fp = len(p_est) - len(matching)
        fn = len(p_ref) - len(matching)

        p, r, f = prec_rec_f1(tp, fp, fn)
        metrics["metric/note/precision.chart"].append(p)
        metrics["metric/note/recall.chart"].append(r)
        metrics["metric/note/f1.chart"].append(f)

    # take the average
    eval_results = dict()
    for key, val in metrics.items():
        eval_results[key] = np.array(val).mean()
    return eval_results


def evaluate_test(model, loaders, difficulties, device, tolerance=0.05):
    # the dict containing prediction results:  {live_id: {diff: {res_key: res_value}}}
    pred_results = defaultdict(lambda: defaultdict(dict))
    for diff in difficulties:
        test_loader = loaders[diff.value]
        pred_results = predict_test(model, test_loader, pred_results, device)

    # calculate metrics for each difficulties
    eval_dict = dict()
    threshold = 0.5
    for diff in difficulties:
        probs = [
            pred_results[l_id][diff_]["proba"]
            for l_id, diffs in pred_results.items()
            for diff_ in diffs
            if diff_ == diff.value
        ]
        targets = [
            pred_results[l_id][diff_]["target"]
            for l_id, diffs in pred_results.items()
            for diff_ in diffs
            if diff_ == diff.value
        ]
        eval_results = chart_metrics(probs, targets, threshold, tolerance)
        eval_results.update(micro_metrics(probs, targets, threshold, tolerance))
        eval_dict[diff.name] = eval_results

    # calculate metrics for all difficulties
    probs = [
        pred_results[l_id][diff_]["proba"]
        for l_id, diffs in pred_results.items()
        for diff_ in diffs
    ]
    targets = [
        pred_results[l_id][diff_]["target"]
        for l_id, diffs in pred_results.items()
        for diff_ in diffs
    ]
    eval_results = chart_metrics(probs, targets, threshold, tolerance)
    eval_results.update(micro_metrics(probs, targets, threshold, tolerance))
    eval_dict["all_diff"] = eval_results

    return eval_dict
