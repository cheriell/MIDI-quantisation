import madmom
import numpy as np

from quantmidi.data.constants import resolution

min_bpm = 55.0
max_bpm = 215.0
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
    
def post_process(onsets, beat_probs, downbeat_probs):
    """
    Post-processing of beat tracking results for proposed model.

    Args:
        onsets: onsets for each note
        beat_probs: beat probabilities for each note
        downbeat_probs: downbeat probabilities for each note
    Returns:
        beats: beat times
        downbeats: downbeat times
    """
    for i in range(10):
        print(onsets[i], beat_probs[i], downbeat_probs[i])
    input()
    return


