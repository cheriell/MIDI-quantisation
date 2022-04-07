import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.abspath('./'))
import argparse
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import pretty_midi as pm
import pickle

from quantmidi.data import DataUtils


def main():

    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Path to the dataset folders in the order \
                        of ASAP, A_MAPS, CPM, ACPAS')
    parser.add_argument('--feature_folder', type=str, help='Path to the feature folder')
    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    featprep = FeaturePreparation(args.dataset_folder, args.feature_folder, args.workers, args.verbose)
    featprep.prepare_metadata()
    featprep.prepare_features()


class FeaturePreparation():

    def __init__(self, dataset_folder, feature_folder, workers, verbose):
        self.dataset_folder = dataset_folder
        self.feature_folder = feature_folder
        self.workers = workers
        self.verbose = verbose

        # dataset folders
        [self.ASAP, self.A_MAPS, self.CPM, self.ACPAS] = self.dataset_folder
        # feature folder
        Path(self.feature_folder).mkdir(parents=True, exist_ok=True)
        # workers
        if self.workers > mp.cpu_count():
            self.workers = mp.cpu_count()
            self.verboseprint('WARNING: Number of workers is greater than the number of available cores. \
                                Setting number of workers to {}'.format(self.workers))
        # verbose
        if self.verbose:
            def verboseprint(*args):
                for arg in args:
                    print(arg, end=' ')
                print()
        else:
            verboseprint = lambda *a: None      # do-nothing function
        self.verboseprint = verboseprint

    def prepare_metadata(self):
        self.verboseprint('INFO: Preparing metadata...')

        # get ACPAS metadata
        self.verboseprint('INFO: Getting ACPAS metadata')
        ACPAS_metadata_S = pd.read_csv(str(Path(self.ACPAS, 'metadata_S.csv')))
        ACPAS_metadata_R = pd.read_csv(str(Path(self.ACPAS, 'metadata_R.csv')))
        ACPAS_metadata = pd.concat([
            ACPAS_metadata_R[ACPAS_metadata_R['source'] == 'MAPS'], 
            ACPAS_metadata_S
        ], ignore_index=True)

        # create metadata
        self.verboseprint('INFO: Creating metadata')
        metadata = pd.DataFrame(columns=[
            'performance_id',
            'piece_id',
            'source',
            'split',
            'midi_perfm',
            'annot_file',
            'feature_file',
        ])

        format_path = lambda x: x.format(ASAP=self.ASAP, A_MAPS=self.A_MAPS, CPM=self.CPM)
        test_ids = set()

        for _, row in ACPAS_metadata.iterrows():
            # performance_id
            performance_id = row['performance_id']
            # piece_id
            piece_id = row['piece_id']
            # source
            source = row['source']
            # split
            split = '--'
            if performance_id[0] == 'R' and source == 'MAPS':
                subset = row['performance_MIDI_external'][:-4].split('_')[-1]
                if subset == 'ENSTDkCl':
                    split = 'test'
                    test_ids.add(piece_id)
            if split == '--':
                if piece_id in test_ids:
                    pass
                elif piece_id % 10 != 0:
                    split = 'train'
                else:
                    split = 'valid'
            # midi_perfm
            if str(row['performance_MIDI_external']) == 'nan':
                midi_perfm = '--'
                self.verboseprint('WARNING: Performance {} has no MIDI file'.format(row['performance_id']))
            else:
                midi_perfm = format_path(row['performance_MIDI_external'])
            # annot_file
            if row['source'] == 'ASAP':
                if str(row['performance_annotation_external']) == 'nan':
                    annot_file = '--'
                    self.verboseprint('WARNING: Performance {} has no annotation file'.format(row['performance_id']))
                else:
                    annot_file = format_path(row['performance_annotation_external'])
            else:
                annot_file = '--'
            # feature_file
            feature_file = Path(self.feature_folder, '{}.pkl'.format(performance_id))

            metadata = metadata.append({
                'performance_id': performance_id,
                'piece_id': piece_id,
                'source': source,
                'split': split,
                'midi_perfm': midi_perfm,
                'annot_file': annot_file,
                'feature_file': str(feature_file),
            }, ignore_index=True)

        # save metadata
        metadata.to_csv(str(Path(self.feature_folder, 'metadata.csv')), index=False)
        self.verboseprint('INFO: Metadata saved to {}'.format(Path(self.feature_folder, 'metadata.csv')))
        
        self.metadata = metadata

    def load_metadata(self):
        self.metadata = pd.read_csv(str(Path(self.feature_folder, 'metadata.csv')))

    def prepare_features(self):
        self.verboseprint('INFO: Preparing features...')

        def prepare_one_feature(row):
            self.verboseprint('INFO: Preparing feature {}'.format(row['performance_id']))

            if row['source'] == 'ASAP':
                # get note sequence
                note_sequence = DataUtils.get_note_sequence_from_midi(row['midi_perfm'])
                # get beat sequence
                beats = DataUtils.get_beats_from_annot_file(row['annot_file'])
            else:
                note_sequence, beats = DataUtils.get_note_sequence_and_beats_from_midi(row['midi_perfm'])

            feature = (note_sequence, beats)
            pickle.dump(feature, open(row['feature_file'], 'wb'))

        # prepare features
        rows = [row for _, row in self.metadata.iterrows()]
        pool = Pool(self.workers)
        pool.map(prepare_one_feature, rows)

        self.verboseprint('INFO: Features prepared')
        

if __name__ == '__main__':
    main()