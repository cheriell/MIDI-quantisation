import madmom
import numpy as np
from enum import IntEnum, auto
from collections import defaultdict

from quantmidi.data.constants import resolution, tolerance

min_bpm = 40
max_bpm = 220

# ==================== DBN beat tracker ====================

transition_lambda = 100.0
beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
    min_bpm=min_bpm,
    max_bpm=max_bpm,
    fps=int(1 / resolution),
    transition_lambda=transition_lambda,
)
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4],
    min_bpm=min_bpm,
    max_bpm=max_bpm,
    fps=int(1 / resolution),
    transition_lambda=transition_lambda,
)

def DBN_beat_track(beat_act, downbeat_act):
    """
    Beat tracking using the DBN algorithm.

    Args:
        beat_act: beat activation tensor
        downbeat_act: downbeat activation tensor
    Returns:
        beats: beat times
        downbeats: downbeat times
    """
    beats = beat_tracker(beat_act)
    combined_act = np.vstack((np.maximum(beat_act - downbeat_act, 0), downbeat_act)).T
    downbeats = downbeat_tracker(combined_act)
    downbeats = downbeats[:, 0][downbeats[:, 1] == 1]
    return beats, downbeats
    
# ==================== Proposed beat tracker ====================


def post_process(
    onsets, 
    beat_probs, 
    downbeat_probs, 
    dynamic_thresholding=True, 
    merge_downbeats=True,
    dynamic_programming=True,
):
    """
    Post-processing of beat and downbeat tracking results for proposed model.

    Args:
        onsets: onsets for each note
        beat_probs: beat probabilities for each note
        downbeat_probs: downbeat probabilities for each note
        dynamic_thresholding: whether to use dynamic thresholding or not
    Returns:
        beats: beat times
        downbeats: downbeat times
    """
    N_notes = len(onsets)

    # ========= Dynamic thresholding =========
    if dynamic_thresholding:
        # window length in seconds
        wlen_beats = (60. / min_bpm) * 4
        wlen_downbeats = (60. / min_bpm) * 8

        # initialize beat and downbeat thresholds
        thresh_beats = np.ones(N_notes) * 0.5
        thresh_downbeats = np.ones(N_notes) * 0.5
        
        l_b, r_b, l_db, r_db = 0, 0, 0, 0  # sliding window indices
        
        for i, onset in enumerate(onsets):
            # udpate pointers
            while onsets[l_b] < onset - wlen_beats / 2:
                l_b += 1
            while r_b < N_notes and onsets[r_b] < onset + wlen_beats / 2:
                r_b += 1
            while onsets[l_db] < onset - wlen_downbeats / 2:
                l_db += 1
            while r_db < N_notes and onsets[r_db] < onset + wlen_downbeats / 2:
                r_db += 1
            # update beat and downbeat thresholds
            thresh_beats[i] = np.max(beat_probs[l_b:r_b]) * 0.5
            thresh_downbeats[i] = np.max(downbeat_probs[l_db:r_db]) * 0.5

        # threshold beat and downbeat probabilities
        beats = onsets[beat_probs > thresh_beats]
        downbeats = onsets[downbeat_probs > thresh_downbeats]

    else:
        beats = onsets[beat_probs > 0.5]
        downbeats = onsets[downbeat_probs > 0.5]

    # ========= remove beats that are too close to each other =========
    beats_min = beats[np.concatenate([[True], np.abs(np.diff(beats)) > tolerance * 2])]
    beats_max = beats[::-1][np.concatenate([[True], np.abs(np.diff(beats[::-1])) > tolerance * 2])][::-1]
    beats = np.mean([beats_min, beats_max], axis=0)
    downbeats_min = downbeats[np.concatenate([[True], np.abs(np.diff(downbeats)) > tolerance * 2])]
    downbeats_max = downbeats[::-1][np.concatenate([[True], np.abs(np.diff(downbeats[::-1])) > tolerance * 2])][::-1]
    downbeats = np.mean([downbeats_min, downbeats_max], axis=0)

    # ========= merge downbeats to beats if they are not in beat prediction =========
    if merge_downbeats:
        beats_to_merge = []
        for downbeat in downbeats:
            if np.min(np.abs(beats - downbeat)) > tolerance * 2:
                beats_to_merge.append(downbeat)
        beats = np.concatenate([beats, beats_to_merge])
        beats = np.sort(beats)

    # ========= fill up out-of-note beats by inter-beat intervals =========
    # fill up by neighboring beats
    wlen = 5  # window length for getting neighboring inter-beat intervals (+- wlen)
    IBIs = np.diff(beats)
    beats_filled = []

    for i in range(len(beats) - 1):
        beats_filled.append(beats[i])

        # current and neighboring inter-beat intervals
        ibi = IBIs[i]
        ibis_near = IBIs[max(0, i-wlen):min(len(IBIs), i+wlen+1)]
        ibis_near_median = np.median(ibis_near)

        for ratio in [2, 3, 4]:
            if abs(ibi / ibis_near_median - ratio) / ratio < 0.15:
                for x in range(1, ratio):
                    beats_filled.append(beats[i] + x * ibi / ratio)
    beats = np.sort(np.array(beats_filled))

    if dynamic_programming:
        beats = run_dynamic_programming(beats)

    return beats, downbeats


# ========= dynamic programming ========================================
# minimize objective function:
#   O = sum(abs(log((t[k] - t[k-1]) / (t[k-1] - t[k-2]))))      (O1)
#       + lam1 * insertions                                     (O2)
#       + lam2 * deletions                                      (O3)
#   t[k] is the kth beat after dynamic programming.
# ======================================================================

def run_dynamic_programming(beats, penalty=1.0):
    beats_dp = [
        [beats[0], beats[1]],     # no insertion
        [beats[0], beats[1]],     # insert one beat
        [beats[0], beats[1]],     # insert two beats
        [beats[0], beats[1]],     # insert three beats
    ]
    obj_dp = [0, 0, 0, 0]

    for i in range(2, len(beats)):
        beats_dp_new = [0] * len(beats_dp)
        obj_dp_new = [0, 0, 0, 0]

        # insert x beats
        for x in range(4):
            ibi = (beats[i] - beats[i-1]) / (x + 1)
            objs = []
            for x_prev in range(4):
                o1 = np.abs(np.log(ibi / (beats_dp[x_prev][-1] - beats_dp[x_prev][-2])))
                o = obj_dp[x_prev] + o1 + penalty * x
                objs.append(o)

            x_prev_best = np.argmin(objs)
            beats_dp_new[x] = beats_dp[x_prev_best] + [beats[i-1] + ibi * k for k in range(1, x+1)] + [beats[i]]
            obj_dp_new[x] = objs[x_prev_best]

        beats_dp = beats_dp_new
        obj_dp = obj_dp_new

    x_best = np.argmin(obj_dp)
    beats = beats_dp[x_best]
    return np.array(beats)

