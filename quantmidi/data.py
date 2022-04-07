import pretty_midi as pm
from functools import reduce, cmp_to_key
from pathlib import Path
import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from collections import defaultdict
import random
import pickle


batch_size = 32


class QuantMIDIDataModule(LightningDataModule):
    def __init__(self, feature_folder, workers):
        super().__init__()
        self.feature_folder = feature_folder
        self.workers = workers

    def train_dataloader(self):
        return self.get_dataloader(feature_folder=self.feature_folder, split='train')

    def val_dataloader(self):
        return self.get_dataloader(feature_folder=self.feature_folder, split='valid')

    def test_dataloader(self):
        return self.get_dataloader(feature_folder=self.feature_folder, split='test')

    def get_dataloader(self, split):

        dataset = QuantMIDIDataset(split=split)
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

    def __init__(self, feature_folder, split):
        super().__init__()
        self.feature_folder = feature_folder
        self.split = split

        # get metadata
        self.metadata = pd.read_csv(str(Path(self.feature_folder), 'metadata.csv'))
        # get split
        self.metadata = self.metadata[self.metadata['split'] == self.split]
        self.metadata.reset_index(inplace=True)

        # get piece2row dictionary
        self.piece2row = defaultdict(list)
        for i, row in self.metadata.iterrows():
            self.piece2row[row['piece_id']].append(i)

    def __len__(self):
        return len(self.metadata) * 20

    def __getitem__(self, idx):
        # random sampling by piece
        piece_id = random.choice(list(self.piece2row.keys()))
        row_id = random.choice(self.piece2row[piece_id])
        row = self.metadata.iloc[row_id]

        # get feature
        feature = pickle.load(str(Path(self.feature_folder, row['feature_file'])), 'rb')
        note_sequence, beats = feature

        # get model input & output
        input_data = DataUtils.get_model_input(note_sequence)
        output_data = DataUtils.get_model_output(note_sequence, beats)

        return input_data, output_data



class DataUtils():
    
    @staticmethod
    def get_note_sequence_from_midi(midi_file):
        """
        Get note sequence from midi file.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        return note_sequence

    @staticmethod
    def get_note_sequence_and_beats_from_midi(midi_file):
        """
        Get beat sequence from midi file.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        beats = midi_data.get_beats()
        return note_sequence, beats

    @staticmethod
    def get_beats_from_annot_file(annot_file):
        """
        Get beat sequence from annotation file.
        """
        annot_data = pd.read_csv(str(Path(annot_file)), header=None, sep='\t')
        beats = annot_data[0].tolist()
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

    @staticmethod
    def get_model_input(note_sequence):
        """
        Get model input from note sequence.
        """
        # ========== get note sequence ==========
        note_sequence = [note.pitch for note in note_sequence]
        # ========== get beat sequence ==========
        beats = [int(beat) for beat in beats]
        # ========== get beat resolution ==========
        beat_resolution = int(beat_resolution)
        # ========== get model input ==========
        model_input = {
            'note_sequence': note_sequence,
            'beats': beats,
            'beat_resolution': beat_resolution
        }
        return model_input

    @staticmethod
    def get_model_output(note_sequence, beats):
        """
        Get model output from note sequence and beats.
        """
        # ========== get note sequence ==========
        note_sequence = [note.pitch for note in note_sequence]
        # ========== get beat sequence ==========
        beats = [int(beat) for beat in beats]
        # ========== get beat resolution ==========
        beat_resolution = int(beat_resolution)
        # ========== get model output ==========
        model_output = {
            'note_sequence': note_sequence,
            'beats': beats,
            'beat_resolution': beat_resolution
        }
        return model_output