import pretty_midi as pm
from functools import reduce, cmp_to_key
from pathlib import Path
import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from collections import defaultdict
import random
import pickle
import numpy as np


batch_size = 32
max_length = 500  # maximum note sequence length during training
resolution = 0.01  # quantization resolution: 0.01s = 10ms
tolerance = 0.05  # tolerance for beat alignment: 0.05s = 50ms

# key in sharps in mido
keySharps2Name = [
    'C', 'G', 'D', 'A', 'E', 'B', 'F#', 
    'C#m', 'G#m', 'D#m', 'Bbm', 'Fm',
    'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 
    'Db', 'Ab', 'Eb', 'Bb', 'F',
]
keyName2Sharps = dict([(name, sharp if sharp <= 11 else sharp - 23) for sharp, name in enumerate(keySharps2Name)])
# key in numbers in pretty_midi
keyNumber2Name = [
    'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm',
]
keyName2Number = dict([(name, number) for number, name in enumerate(keyNumber2Name)])

class QuantMIDIDataModule(LightningDataModule):
    def __init__(self, feature_folder, model_type, data_aug_args, workers):
        super().__init__()
        self.feature_folder = feature_folder
        self.model_type = model_type
        self.data_aug_args = data_aug_args

        self.workers = workers if model_type == 'note_sequence' else 0
        self.bs = batch_size if model_type == 'note_sequence' else 1

    def train_dataloader(self):
        dataset = QuantMIDIDataset(
            self.feature_folder, 
            'train', 
            self.model_type, 
            data_aug_args=self.data_aug_args
        )
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        train_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=self.bs,
            sampler=sampler,
            num_workers=self.workers,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        dataset = QuantMIDIDataset(self.feature_folder, 'valid', self.model_type)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        val_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=self.bs,
            sampler=sampler,
            num_workers=self.workers,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self):
        dataset = QuantMIDIDataset(self.feature_folder, 'test', self.model_type)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        test_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            drop_last=False
        )
        return test_loader

class QuantMIDIDataset(torch.utils.data.Dataset):

    def __init__(self, 
        feature_folder, 
        split, 
        model_type,
        **kwargs
    ):
        super().__init__()
        self.feature_folder = feature_folder
        self.split = split
        self.model_type = model_type

        # get metadata
        self.metadata = pd.read_csv(str(Path(self.feature_folder, 'metadata.csv')))
        # get split
        self.metadata = self.metadata[self.metadata['split'] == self.split]
        self.metadata.reset_index(inplace=True)

        # get piece2row dictionary
        self.piece2row = defaultdict(list)
        for i, row in self.metadata.iterrows():
            self.piece2row[row['piece_id']].append(i)

        # initialize data augmentation
        if split == 'train':
            data_aug_args = kwargs['data_aug_args']

            self.dataaug = DataAugmentation(
                tempo_change_prob=data_aug_args['tempo_change_prob'],
                tempo_change_range=data_aug_args['tempo_change_range'],
                pitch_shift_prob=data_aug_args['pitch_shift_prob'],
                pitch_shift_range=data_aug_args['pitch_shift_range'],
                extra_note_prob=data_aug_args['extra_note_prob'],
                missing_note_prob=data_aug_args['missing_note_prob'],
            )

    def __len__(self):
        if self.split == 'train':
            if self.model_type == 'note_sequence':
                return batch_size * 4 * 200     # constantly update 200 steps per epoch
                                                # not related to training dataset size
            elif self.model_type == 'baseline':
                return 4 * 200

        elif self.split == 'valid':
            # by istinct pieces in validation set
            self.pieces = list(self.piece2row.keys())
            if self.model_type == 'note_sequence':
                return batch_size * len(self.piece2row)  # valid dataset size
            elif self.model_type == 'baseline':
                return 4 * len(self.piece2row)

        elif self.split == 'test':
            return len(self.metadata)

    def __getitem__(self, idx):
        # get row
        if self.split == 'train':
            piece_id = random.choice(list(self.piece2row.keys()))   # random sampling by piece
            row_id = random.choice(self.piece2row[piece_id])
        elif self.split == 'valid':
            piece_id = self.pieces[idx // batch_size]    # by istinct pieces in validation set
            row_id = self.piece2row[piece_id][idx % batch_size % len(self.piece2row[piece_id])]
        elif self.split == 'test':
            row_id = idx
        row = self.metadata.iloc[row_id]

        # get feature
        note_sequence, annotations = pickle.load(open(str(Path(self.feature_folder, row['feature_file'])), 'rb'))
        beats = annotations['beats']

        # data augmentation
        if self.split == 'train':
            note_sequence, beats = self.dataaug(note_sequence, beats)

        # sample segment and get model input & output

        def sample_segment_note_sequence(note_sequence, beats):
            # ========== get model input ==========
            # list of tuples (pitch, onset, duration, velocity) in torch tensor
            # randomly select a segment by max_length
            if len(note_sequence) > max_length:
                if self.split == 'train':
                    start_idx = random.randint(0, len(note_sequence) - max_length)
                    end_idx = start_idx + max_length
                elif self.split == 'valid':
                    start_idx, end_idx = 0, max_length  # validate on the segment starting with the first note
                elif self.split == 'test':
                    start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence
            else:
                start_idx, end_idx = 0, len(note_sequence)
            note_sequence = note_sequence[start_idx:end_idx]

            # =========== get model output ===========
            # list of beat probs in torch tensor
            # onset to beat dict
            end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
            onset2beat = torch.zeros(int(torch.ceil(end_time / resolution)))
            for beat in beats:
                l = torch.round((beat - tolerance) / resolution).to(int)
                r = torch.round((beat + tolerance) / resolution).to(int)
                onset2beat[l:r+1] = 1.0
            
            # get beat probabilities at note onsets
            beat_probs = torch.zeros(len(note_sequence), dtype=torch.float32)
            for i in range(len(note_sequence)):
                onset = note_sequence[i][1]
                beat_probs[i] = onset2beat[torch.round(onset / resolution).to(int)]

            # ============ pad if length is shorter than max_length ============
            length = len(note_sequence)
            if len(note_sequence) < max_length:
                note_sequence = torch.cat([note_sequence, torch.zeros((max_length - len(note_sequence), 4))])
                beat_probs = torch.cat([beat_probs, torch.zeros(max_length - len(beat_probs))])

            return note_sequence, beat_probs, length

        def sample_segment_baseline(note_sequence, beats):
            # ========== get model input ==========
            # still note sequence, but do not segment by max_length this time.
            # convert to pianoroll in model forward - it's faster to use the GPU.
            # forward note_sequence directly to model input.

            # =========== get model output ===========
            # list of beat probs in torch tensor.
            beat_probs, length = DataUtils.get_beat_activation(note_sequence, beats)

            return note_sequence, beat_probs, length

        if self.model_type == 'note_sequence':
            return sample_segment_note_sequence(note_sequence, beats)
        elif self.model_type == 'baseline':
            return sample_segment_baseline(note_sequence, beats)

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

    def __call__(self, note_sequence, beats):
        # tempo change
        if random.random() < self.tempo_change_prob:
            note_sequence, beats = self.tempo_change(note_sequence, beats)

        # pitch shift
        if random.random() < self.pitch_shift_prob:
            note_sequence = self.pitch_shift(note_sequence)

        # extra note or missing note
        extra_or_missing = random.random()
        if extra_or_missing < self.extra_note_prob:
            note_sequence = self.extra_note(note_sequence)
        elif extra_or_missing > 1. - self.missing_note_prob:
            note_sequence = self.missing_note(note_sequence)

        return note_sequence, beats

    def tempo_change(self, note_sequence, beats):
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        note_sequence[:,1:3] *= 1 / tempo_change_ratio
        beats *= 1 / tempo_change_ratio
        return note_sequence, beats

    def pitch_shift(self, note_sequence):
        shift = round(random.uniform(*self.pitch_shift_range))
        note_sequence[:,0] += shift
        return note_sequence

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

class DataUtils():
    
    @staticmethod
    def get_note_sequence_from_midi(midi_file):
        """
        Get note sequence from midi file.
        Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in torch.Tensor.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        # conver to Tensor
        note_sequence = torch.Tensor([(note.pitch, note.start, note.end-note.start, note.velocity) \
                                        for note in note_sequence])
        return note_sequence

    @staticmethod
    def get_annotations_from_annot_file(annot_file):
        """
        Get annotations from annotation file in ASAP dataset.
        annotatioins in a dict of {
            beats: list of beat times,
            downbeats: list of downbeat times,
            time_signatures: list of (time, numerator, denominator) tuples,
            key_signatures: list of (time, sharps) tuples
        }, all in torch.Tensor.
        """
        annot_data = pd.read_csv(str(Path(annot_file)), header=None, sep='\t')

        beats, downbeats, key_signatures, time_signatures = [], [], [], []
        for i, row in annot_data.iterrows():
            a = row[2].split(',')
            # beats
            beats.append(row[0])
            # downbeats
            if a[0] == 'db':
                downbeats.append(row[0])
            # time_signatures
            if len(a) >= 2 and a[1] != '':
                numerator, denominator = a[1].split('/')
                time_signatures.append((row[0], int(numerator), int(denominator)))
            # key_signatures
            if len(a) == 3 and a[2] != '':
                key_signatures.append((row[0], int(a[2])))

        # save as annotation dict
        annotations = {
            'beats': torch.Tensor(beats),
            'downbeats': torch.Tensor(downbeats),
            'time_signatures': torch.Tensor(time_signatures),
            'key_signatures': torch.Tensor(key_signatures),
        }
        return annotations

    @staticmethod
    def get_note_sequence_and_annotations_from_midi(midi_file):
        """
        Get beat sequence and annotations from midi file.
        Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in torch.Tensor.
        annotations in a dict of {
            beats: list of beat times,
            downbeats: list of downbeat times,
            time_signatures: list of (time, numerator, denominator) tuples,
            key_signatures: list of (time, sharps) tuples,
            onsets_musical: list of onsets in musical time (within a beat),
            note_value: list of note values (in beats),
            hand: list of hand (0: left, 1: right)
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))

        # note sequence
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        # conver to Tensor
        note_sequence = torch.Tensor([[note.pitch, note.start, note.end-note.start, note.velocity] \
                                        for note in note_sequence])

        # beats
        beats = midi_data.get_beats()
        # downbeats
        downbeats = midi_data.get_downbeats()
        # time_signatures
        time_signatures = [(t.time, t.numerator, t.denominator) for t in midi_data.time_signature_changes]
        # key_signatures
        key_signatures = [(k.time, keyName2Sharps[keyNumber2Name[k.key_number]]) for k in \
                            midi_data.key_signature_changes]
        # onsets_musical
        # note_value
        
        # hand

        # save as annotation dict
        annotations = {
            'beats': torch.Tensor(beats),
            'downbeats': torch.Tensor(downbeats),
            'time_signatures': torch.Tensor(time_signatures),
            'key_signatures': torch.Tensor(key_signatures),
            'onsets_musical': None,
            'note_value': None,
            'hand': None,
        }
        return note_sequence, annotations

    @staticmethod
    def compare_note_order(note1, note2):
        """
        Compare two notes by firstly onset and then pitch.
        """
        if note1.start < note2.start:
            return -1
        elif note1.start == note2.start:
            if note1.pitch < note2.pitch:
                return -1
            elif note1.pitch == note2.pitch:
                return 0
            else:
                return 1
        else:
            return 1

    @staticmethod
    def get_beat_activation(note_sequence, beats):
        """
        Get beat activation from beat sequence.
        """
        beat_activation_length = (torch.max(note_sequence[:,1] + note_sequence[:,2]) * (1 / resolution) + 1).long()
        beat_activation = torch.zeros(beat_activation_length).float()
        for beat in beats:
            left = int(max(0, torch.round(beat - tolerance)))
            right = int(min(beat_activation_length, torch.round(beat + tolerance)))
            beat_activation[left:right] = 1.0
        return beat_activation, beat_activation_length