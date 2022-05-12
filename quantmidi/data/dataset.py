import torch
import random
import pickle
import pandas as pd

from pathlib import Path
from collections import defaultdict

from quantmidi.data.data_aug import DataAugmentation
from quantmidi.data.data_utils import DataUtils
from quantmidi.data.constants import (
    tolerance, 
    resolution, 
    max_length_note_sequence, 
    batch_size_note_sequence, 
    batch_size_baseline,
    batch_size_proposed,
    tsDeno2Index,
    keyVocabSize,
)


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

        if self.model_type == 'note_sequence':
            self.batch_size = batch_size_note_sequence
        elif self.model_type == 'baseline':
            self.batch_size = batch_size_baseline
        elif self.model_type == 'proposed':
            self.batch_size = batch_size_proposed

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
            # constantly update 200 steps per epoch, not related to training dataset size
            return self.batch_size * 4 * 200

        elif self.split == 'valid':
            # by istinct pieces in validation set
            self.pieces = list(self.piece2row.keys())
            return self.batch_size * len(self.piece2row)  # valid dataset size

        elif self.split == 'test':
            return len(self.metadata)

    def __getitem__(self, idx):
        # get row
        if self.split == 'train':
            piece_id = random.choice(list(self.piece2row.keys()))   # random sampling by piece
            row_id = random.choice(self.piece2row[piece_id])
        elif self.split == 'valid':
            piece_id = self.pieces[idx // self.batch_size]    # by istinct pieces in validation set
            row_id = self.piece2row[piece_id][idx % self.batch_size % len(self.piece2row[piece_id])]
        elif self.split == 'test':
            row_id = idx
        row = self.metadata.iloc[row_id]

        # get feature
        note_sequence, annotations = pickle.load(open(str(Path(self.feature_folder, row['feature_file'])), 'rb'))

        # data augmentation
        if self.split == 'train':
            note_sequence, annotations = self.dataaug(note_sequence, annotations)

        # sample segment and get model input & output

        def get_data_note_sequence(note_sequence, annotations):

            # ========== get model input ==========
            # list of tuples (pitch, onset, duration, velocity) in torch tensor
            # randomly select a segment by max_length_note_sequence
            if len(note_sequence) > max_length_note_sequence:
                if self.split == 'train':
                    start_idx = random.randint(0, len(note_sequence) - max_length_note_sequence)
                    end_idx = start_idx + max_length_note_sequence
                elif self.split == 'valid':
                    start_idx, end_idx = 0, max_length_note_sequence  # validate on the segment starting with the first note
                elif self.split == 'test':
                    start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence
            else:
                start_idx, end_idx = 0, len(note_sequence)
            note_sequence = note_sequence[start_idx:end_idx]

            # =========== get model output ===========
            beats = annotations['beats']
            downbeats = annotations['downbeats']

            # onset to beat/downbeat dict
            end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
            onset2beat = torch.zeros(int(torch.ceil(end_time / resolution)))
            onset2downbeat = torch.zeros(int(torch.ceil(end_time / resolution)))
            for beat in beats:
                l = torch.round((beat - tolerance) / resolution).to(int)
                r = torch.round((beat + tolerance) / resolution).to(int)
                onset2beat[l:r+1] = 1.0
            for downbeat in downbeats:
                l = torch.round((downbeat - tolerance) / resolution).to(int)
                r = torch.round((downbeat + tolerance) / resolution).to(int)
                onset2downbeat[l:r+1] = 1.0
            
            # get beat/downbeat probabilities at note onsets
            beat_probs = torch.zeros(len(note_sequence), dtype=torch.float32)
            downbeat_probs = torch.zeros(len(note_sequence), dtype=torch.float32)
            for i in range(len(note_sequence)):
                onset = note_sequence[i][1]
                beat_probs[i] = onset2beat[torch.round(onset / resolution).to(int)]
                downbeat_probs[i] = onset2downbeat[torch.round(onset / resolution).to(int)]

            # ============ pad if length is shorter than max_length ============
            length = len(note_sequence)
            if len(note_sequence) < max_length_note_sequence:
                note_sequence = torch.cat([note_sequence, torch.zeros((max_length_note_sequence - len(note_sequence), 4))])
                beat_probs = torch.cat([beat_probs, torch.zeros(max_length_note_sequence - len(beat_probs))])
                downbeat_probs = torch.cat([downbeat_probs, torch.zeros(max_length_note_sequence - len(downbeat_probs))])

            return note_sequence, beat_probs, downbeat_probs, length

        def get_data_baseline(note_sequence, annotations):
            # ========== get model input ==========
            # Convert to pianoroll in model forward - it's faster to use the GPU.
            # forward note_sequence directly to model input.
            # List of tuples (pitch, onset, duration, velocity) in torch tensor
            # randomly select a segment by max_length_note_sequence
            if len(note_sequence) > max_length_note_sequence:
                if self.split == 'train':
                    start_idx = random.randint(0, len(note_sequence) - max_length_note_sequence)
                    end_idx = start_idx + max_length_note_sequence
                elif self.split == 'valid':
                    start_idx, end_idx = 0, max_length_note_sequence  # validate on the segment starting with the first note
                elif self.split == 'test':
                    start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence
            else:
                start_idx, end_idx = 0, len(note_sequence)
            note_sequence = note_sequence[start_idx:end_idx]
            # pad if length is shorter than max_length_note_sequence
            if len(note_sequence) < max_length_note_sequence:
                note_sequence = torch.cat([note_sequence, torch.zeros((max_length_note_sequence - len(note_sequence), 4))])

            # =========== get model output ===========
            # list of beat probs in torch tensor.
            if self.split == 'train':
                beat_act, downbeat_act, ts_denos, key_numbers, length = DataUtils.get_baseline_model_output_data(
                    note_sequence, 
                    annotations,
                    sample_segment=True,
                )
            elif self.split == 'valid':
                beat_act, downbeat_act, ts_denos, key_numbers, length = DataUtils.get_baseline_model_output_data(
                    note_sequence,
                    annotations,
                    sample_segment=True,
                )
            
            return note_sequence, beat_act, downbeat_act, ts_denos, key_numbers, length

        def get_data_proposed(note_sequence, annotations):
            # ========== get model input ==========
            # list of tuples (pitch, onset, duration, velocity) in torch tensor
            # randomly select a segment by max_length_note_sequence
            if len(note_sequence) > max_length_note_sequence:
                if self.split == 'train':
                    start_idx = random.randint(0, len(note_sequence) - max_length_note_sequence)
                    end_idx = start_idx + max_length_note_sequence
                elif self.split == 'valid':
                    start_idx, end_idx = 0, max_length_note_sequence  # validate on the segment starting with the first note
                elif self.split == 'test':
                    start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence
            else:
                start_idx, end_idx = 0, len(note_sequence)
            note_sequence = note_sequence[start_idx:end_idx]


            # =========== get model output ===========
            # list of beat probs in torch tensor
            beats = annotations['beats']
            downbeats = annotations['downbeats']

            # onset to beat/downbeat dict
            end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
            onset2beat = torch.zeros(int(torch.ceil(end_time / resolution)))
            onset2downbeat = torch.zeros(int(torch.ceil(end_time / resolution)))
            for beat in beats:
                l = torch.round((beat - tolerance) / resolution).to(int)
                r = torch.round((beat + tolerance) / resolution).to(int)
                onset2beat[l:r+1] = 1.0
            for downbeat in downbeats:
                l = torch.round((downbeat - tolerance) / resolution).to(int)
                r = torch.round((downbeat + tolerance) / resolution).to(int)
                onset2downbeat[l:r+1] = 1.0
            
            # get beat probabilities at note onsets
            beat_probs = torch.zeros(len(note_sequence), dtype=torch.float32)
            downbeat_probs = torch.zeros(len(note_sequence), dtype=torch.float32)
            for i in range(len(note_sequence)):
                onset = note_sequence[i][1]
                beat_probs[i] = onset2beat[torch.round(onset / resolution).to(int)]
                downbeat_probs[i] = onset2downbeat[torch.round(onset / resolution).to(int)]

            # time signature denomitators
            time_signatures = annotations['time_signatures']
            ts_denos = torch.zeros(len(note_sequence))

            for i in range(len(note_sequence)):
                onset = note_sequence[i][1]
                for ts in time_signatures:
                    if ts[0] >= onset - tolerance:
                        ts_denos[i] = tsDeno2Index[int(ts[2])]
                        break

            # key signatures
            key_signatures = annotations['key_signatures']
            key_numbers = torch.zeros(len(note_sequence))

            for i in range(len(note_sequence)):
                onset = note_sequence[i][1]
                for ks in key_signatures:
                    if ks[0] >= onset - tolerance:
                        key_numbers[i] = ks[1] % keyVocabSize
                        break

            # ============ pad if length is shorter than max_length ============
            length = len(note_sequence)
            if len(note_sequence) < max_length_note_sequence:
                note_sequence = torch.cat([note_sequence, torch.zeros((max_length_note_sequence - len(note_sequence), 4))])
                beat_probs = torch.cat([beat_probs, torch.zeros(max_length_note_sequence - len(beat_probs))])
                downbeat_probs = torch.cat([downbeat_probs, torch.zeros(max_length_note_sequence - len(downbeat_probs))])
                ts_denos = torch.cat([ts_denos, torch.zeros(max_length_note_sequence - len(ts_denos))])
                key_numbers = torch.cat([key_numbers, torch.zeros(max_length_note_sequence - len(key_numbers))])

            return note_sequence, beat_probs, downbeat_probs, ts_denos, key_numbers, length

        if self.model_type == 'note_sequence':
            return get_data_note_sequence(note_sequence, annotations)
        elif self.model_type == 'baseline':
            return get_data_baseline(note_sequence, annotations)
        elif self.model_type == 'proposed':
            return get_data_proposed(note_sequence, annotations)
