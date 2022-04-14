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
        return self.get_dataloader(split='train')

    def val_dataloader(self):
        return self.get_dataloader(split='valid')

    def test_dataloader(self):
        return self.get_dataloader(split='test')

    def get_dataloader(self, split):

        dataset = QuantMIDIDataset(self.feature_folder, split, self.model_type)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.workers,
            drop_last=True
        )

        return data_loader


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
            return batch_size * 4 * 500     # constantly update 500 steps per epoch
                                            # not related to training dataset size
        else:
            return batch_size * len(self.metadata)  # valid dataset size

    def __getitem__(self, idx):
        # random sampling by piece
        piece_id = random.choice(list(self.piece2row.keys()))
        row_id = random.choice(self.piece2row[piece_id])
        row = self.metadata.iloc[row_id]

        # get feature
        note_sequence, beats = pickle.load(open(str(Path(self.feature_folder, row['feature_file'])), 'rb'))
        note_sequence = note_sequence
        beats = beats

        # data augmentation
        note_sequence, beats = self.dataaug(note_sequence, beats)

        # sample segment and get model input & output

        def sample_segment_note_sequence(note_sequence, beats):
            # ========== get model input ==========
            # randomly select a segment of max_length
            if len(note_sequence) > max_length:
                start_idx = random.randint(0, len(note_sequence) - max_length)
                end_idx = start_idx + max_length
            else:
                start_idx, end_idx = 0, len(note_sequence)
            note_sequence = note_sequence[start_idx:end_idx]

            # =========== get model output ===========
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
        pitch_shift_prob=0.8,
        pitch_shift_range=(-12, 12),
        extra_note_prob=0.2,
        missing_note_prob=0.8,):
        
        pass

    def __call__(self, note_sequence, beats):
        return note_sequence, beats


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
