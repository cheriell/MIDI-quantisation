import madmom
import numpy as np

from quantmidi.data.constants import resolution, tolerance

min_bpm = 50
max_bpm = 220
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
    
def post_process(onsets, beat_probs, downbeat_probs, dynamic_thresholding=True):
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

    # remove beats that are too close to each other
    beats_min = beats[np.concatenate([[True], np.abs(np.diff(beats)) > tolerance * 2])]
    beats_max = beats[::-1][np.concatenate([[True], np.abs(np.diff(beats[::-1])) > tolerance * 2])][::-1]
    beats = np.mean([beats_min, beats_max], axis=0)
    downbeats_min = downbeats[np.concatenate([[True], np.abs(np.diff(downbeats)) > tolerance * 2])]
    downbeats_max = downbeats[::-1][np.concatenate([[True], np.abs(np.diff(downbeats[::-1])) > tolerance * 2])][::-1]
    downbeats = np.mean([downbeats_min, downbeats_max], axis=0)

    # fill up out-of-note beats by inter-beat intervals
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
    beats = np.array(beats_filled)

    return beats, downbeats


