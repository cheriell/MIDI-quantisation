import torch
import random

from quantmidi.data.constants import tolerance, keySharps2Number, keyNumber2Sharps

class DataAugmentation():
    def __init__(self, 
        tempo_change_prob=1.0,
        tempo_change_range=(0.8, 1.2),
        pitch_shift_prob=1.0,
        pitch_shift_range=(-12, 12),
        extra_note_prob=0.5,
        missing_note_prob=0.5):

        if extra_note_prob + missing_note_prob > 1.:
            extra_note_prob, missing_note_prob = extra_note_prob / (extra_note_prob + missing_note_prob), \
                                                missing_note_prob / (extra_note_prob + missing_note_prob)
            print('INFO: Reset extra_note_prob and missing_note_prob to', extra_note_prob, missing_note_prob)
        
        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.extra_note_prob = extra_note_prob
        self.missing_note_prob = missing_note_prob

    def __call__(self, note_sequence, annotations):
        # tempo change
        if random.random() < self.tempo_change_prob:
            note_sequence, annotations = self.tempo_change(note_sequence, annotations)

        # pitch shift
        if random.random() < self.pitch_shift_prob:
            note_sequence, annotations = self.pitch_shift(note_sequence, annotations)

        # extra note or missing note
        extra_or_missing = random.random()
        if extra_or_missing < self.extra_note_prob:
            note_sequence = self.extra_note(note_sequence)
        elif extra_or_missing > 1. - self.missing_note_prob:
            note_sequence = self.missing_note(note_sequence)

        return note_sequence, annotations

    def tempo_change(self, note_sequence, annotations):
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        note_sequence[:,1:3] *= 1 / tempo_change_ratio
        annotations['beats'] *= 1 / tempo_change_ratio
        annotations['downbeats'] *= 1 / tempo_change_ratio
        annotations['time_signatures'][:,0] *= 1 / tempo_change_ratio
        annotations['key_signatures'][:,0] *= 1 / tempo_change_ratio
        return note_sequence, annotations

    def pitch_shift(self, note_sequence, annotations):
        shift = round(random.uniform(*self.pitch_shift_range))
        note_sequence[:,0] += shift

        for i in range(len(annotations['key_signatures'])):
            annotations['key_signatures'][i,1] += shift
            annotations['key_signatures'][i,1] %= 24
            
        return note_sequence, annotations

    def extra_note(self, note_sequence):
        # duplicate
        note_sequence_new = torch.zeros(len(note_sequence) * 2, 4)
        note_sequence_new[::2,:] = note_sequence.clone()
        note_sequence_new[1::2,:] = note_sequence.clone()

        # keep a random ratio of extra notes
        ratio = random.random() * 0.3
        probs = torch.rand(len(note_sequence_new))
        probs[::2] = 0.
        remaining = probs < ratio
        note_sequence_new = note_sequence_new[remaining]

        # pitch shift for extra notes (+-12)
        shift = ((torch.round(torch.rand(len(note_sequence_new))) - 0.5) * 24).int()
        shift[::2] = 0
        note_sequence_new[:,0] += shift
        note_sequence_new[:,0][note_sequence_new[:,0] < 0] += 12
        note_sequence_new[:,0][note_sequence_new[:,0] > 127] -= 12

        return note_sequence_new

    def missing_note(self, note_sequence):
        # find successing concurrent notes
        candidates = torch.diff(note_sequence[:,1]) < tolerance

        # randomly select a ratio of candidates to be removed
        ratio = random.random()
        candidates_probs = candidates * torch.rand(len(candidates))
        remaining = torch.cat([torch.tensor([True]), candidates_probs < (1 - ratio)])

        # remove selected candidates
        note_sequence = note_sequence[remaining]

        return note_sequence
