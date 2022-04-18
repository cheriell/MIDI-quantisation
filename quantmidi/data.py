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


class QuantMIDIDataModule(LightningDataModule):
    def __init__(self, feature_folder, model_type, workers):
        super().__init__()
        self.feature_folder = feature_folder
        self.model_type = model_type
        self.workers = workers

    def train_dataloader(self):
        dataset = QuantMIDIDataset(self.feature_folder, 'train', self.model_type)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        train_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
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
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.workers,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self):
        dataset = QuantMIDIDataModule(self.feature_folder, 'test', self.model_type)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        test_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=1,
            drop_last=False
        )

class QuantMIDIDataset(torch.utils.data.Dataset):

    def __init__(self, feature_folder, split, model_type):
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
        self.dataaug = DataAugmentation()

    def __len__(self):
        if self.split == 'train':
            return batch_size * 4 * 200     # constantly update 200 steps per epoch
                                            # not related to training dataset size
        elif self.split == 'valid':
            # by istinct pieces in validation set
            self.pieces = list(self.piece2row.keys())
            return batch_size * len(self.piece2row)  # valid dataset size
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
        note_sequence, beats = pickle.load(open(str(Path(self.feature_folder, row['feature_file'])), 'rb'))

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

        def sample_segment_pianoroll(note_sequence, beats):
            return

        if self.model_type == 'note_sequence':
            return sample_segment_note_sequence(note_sequence, beats)
        elif self.model_type == 'pianoroll':
            return sample_segment_pianoroll(note_sequence, beats)

class DataAugmentation():
    def __init__(self, 
        tempo_change_prob=1.0,
        tempo_change_range=(0.8, 1.2),
        pitch_shift_prob=1.0,
        pitch_shift_range=(-12, 12),
        extra_note_prob=0.0,
        missing_note_prob=1.0):

        if extra_note_prob + missing_note_prob > 1.:
            raise ValueError('extra_note_prob + missing_note_prob must be less than 1.')
        
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
        shift = random.uniform(*self.pitch_shift_range)
        note_sequence[:,0] += shift
        return note_sequence

    def extra_note(self, note_sequence):
        return note_sequence

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
        Note sequence is a list of (pitch, onset, duration, velocity) tuples.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        # conver to Tensor
        note_sequence = torch.Tensor([[note.pitch, note.start, note.end-note.start, note.velocity] \
                                        for note in note_sequence])
        return note_sequence

    @staticmethod
    def get_note_sequence_and_beats_from_midi(midi_file):
        """
        Get beat sequence from midi file.
        Note sequence is a list of (pitch, onset, duration, velocity) tuples.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        beats = midi_data.get_beats()
        # conver to Tensor
        note_sequence = torch.Tensor([[note.pitch, note.start, note.end-note.start, note.velocity] \
                                        for note in note_sequence])
        beats = torch.Tensor(beats)
        return note_sequence, beats

    @staticmethod
    def get_beats_from_annot_file(annot_file):
        """
        Get beat sequence from annotation file.
        """
        annot_data = pd.read_csv(str(Path(annot_file)), header=None, sep='\t')
        beats = annot_data[0].tolist()
        # conver to Tensor
        beats = torch.Tensor(beats)
        return beats

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
