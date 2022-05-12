import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.abspath('./'))
import argparse
import pytorch_lightning as pl
pl.seed_everything(42)
import pandas as pd
import pickle
from pathlib import Path

from quantmidi.data.data_module import QuantMIDIDataModule
from quantmidi.models.note_sequence import NoteSequenceModel
from quantmidi.models.baseline import BaselineModel
from quantmidi.models.proposed import ProposedModel
from quantmidi.post_processing import DBN_beat_track, post_process

## -------------------------
## DEBUGGING BLOCK
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.autograd.set_detect_anomaly(True)
## END DEBUGGING BLOCK
## -------------------------


def train_or_test(args):
    
    # ========= get workspace =========
    feature_folder = str(Path(args.workspace, 'features'))
    tracking_uri = str(Path(args.workspace, 'mlruns'))

    # ========= create dataset, model, logger =========
    data_aug_args = {
        'tempo_change_prob': args.tempo_change_prob,
        'tempo_change_range': args.tempo_change_range,
        'pitch_shift_prob': args.pitch_shift_prob,
        'pitch_shift_range': args.pitch_shift_range,
        'extra_note_prob': args.extra_note_prob,
        'missing_note_prob': args.missing_note_prob,
    }
    datamodule = QuantMIDIDataModule(feature_folder=feature_folder, model_type=args.model_type, 
                                    data_aug_args=data_aug_args, workers=args.workers)

    if args.model_type == 'note_sequence':
        model = NoteSequenceModel(
            features=args.features,
            pitch_encoding=args.pitch_encoding,
            onset_encoding=args.onset_encoding,
            duration_encoding=args.duration_encoding,
            downbeats=args.downbeats,
        )
    elif args.model_type == 'baseline':
        model = BaselineModel()
    elif args.model_type == 'proposed':
        model = ProposedModel()

    logger = pl.loggers.MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=tracking_uri,
        run_name=args.run_name,
        tags={
            'option': args.option,
            'model_type': args.model_type,
            'features': ','.join(args.features),
            'pitch_encoding': args.pitch_encoding,
            'onset_encoding': args.onset_encoding,
            'duration_encoding': args.duration_encoding,
            'tempo_change_prob': args.tempo_change_prob,
            'tempo_change_range': ','.join(map(str, args.tempo_change_range)),
            'pitch_shift_prob': args.pitch_shift_prob,
            'pitch_shift_range': ','.join(map(str, args.pitch_shift_range)),
            'extra_note_prob': args.extra_note_prob,
            'missing_note_prob': args.missing_note_prob,
            'output_type': args.output_type,
            'workers': args.workers,
            'gpus': args.gpus,
        },
    )

    # ========= create trainer =========
    trainer = pl.Trainer(
        default_root_dir=tracking_uri,
        logger=logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        gpus=args.gpus,
        # resume_from_checkpoint=args.model_checkpoint if args.resume_training else None,
    )

    # ========= train/test =========
    if args.option == 'train':
        trainer.fit(model, datamodule=datamodule)
    elif args.option == 'test':
        trainer.test(model, ckpt_path=args.model_checkpoint, datamodule=datamodule)

def evaluate(args):

    from madmom.evaluation.beats import BeatEvaluation, BeatMeanEvaluation

    feature_folder = str(Path(args.workspace, 'features'))
    device = torch.device('cuda') if args.gpus else torch.device('cpu')

    # ========= load pre-trained model from checkpoint =========
    if args.model_type == 'note_sequence':
        model = NoteSequenceModel.load_from_checkpoint(
            args.model_checkpoint,
            features=args.features,
            pitch_encoding=args.pitch_encoding,
            onset_encoding=args.onset_encoding,
            duration_encoding=args.duration_encoding,
            downbeats=args.downbeats,
        ).to(device)
    elif args.model_type == 'baseline':
        model = BaselineModel.load_from_checkpoint(args.model_checkpoint).to(device)
    elif args.model_type == 'proposed':
        model = ProposedModel.load_from_checkpoint(args.model_checkpoint).to(device)
    model.eval()

    # ========= get test set metadata =========
    metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))
    metadata = metadata[metadata['split'] == 'test']
    metadata.reset_index(inplace=True)

    # ========= iterate over the test set and evaluate =========
    evals_beats, evals_downbeats = [], []  # beat-level F-measure

    for i, row in metadata.iterrows():
        print('Evaluating {}/{}'.format(i+1, len(metadata)), end='\r')
        # get model input and ground truth annotations
        note_sequence, annotations = pickle.load(open(str(Path(feature_folder, row['feature_file'])), 'rb'))
        note_sequence = note_sequence.unsqueeze(0).to(device)
        beats_targ = annotations['beats']
        downbeats_targ = annotations['downbeats']

        # get model predictions
        if args.model_type == 'baseline':
            length = torch.Tensor([len(note_sequence)]).unsqueeze(0).to(device)
            y_b, y_db = model(note_sequence, length)
            y_b = y_b.squeeze(0).cpu().detach().numpy()
            y_db = y_db.squeeze(0).cpu().detach().numpy()
            
            beats_pred, downbeats_pred = DBN_beat_track(y_b, y_db)

        elif args.model_type == 'proposed':
            continue

        elif args.model_type == 'note_sequence':
            y_b, y_db = model(note_sequence)
            y_b = y_b.squeeze(0).cpu().detach().numpy()
            y_db = y_db.squeeze(0).cpu().detach().numpy()

            onsets = note_sequence[0,:,1].cpu().detach().numpy()
            beats_pred, downbeats_pred = post_process(onsets, y_b, y_db)

            # # debug
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(20,10))
            # plt.vlines(beats_pred[beats_pred < 30], 0, 1, color='r', linestyle='--')
            # plt.vlines(beats_targ[beats_targ < 30], 1, 2, color='b', linestyle='--')
            # plt.savefig('debug.png'); input()

        # evaluate using beat-level F-measure
        evals_beats.append(BeatEvaluation(beats_pred, beats_targ))
        evals_downbeats.append(BeatEvaluation(downbeats_pred, downbeats_targ, downbeats=True))
    
    print('\n ======== Beat-level F-measure =========')
    print('Beat tracking:')
    print(BeatMeanEvaluation(evals_beats))
    print('Downbeat tracking:')
    print(BeatMeanEvaluation(evals_downbeats))

if __name__ == '__main__':
    
    # ========= parse arguments =========
    parser = argparse.ArgumentParser(description='Main programme for model training.')

    # dataset path and workspace
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Path to the dataset folders \
                        in the order of ASAP, A_MAPS, CPM, ACPAS')
    parser.add_argument('--workspace', type=str, help='Path to the workspace')

    # experiment settings
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment', default='full-training')
    parser.add_argument('--run_name', type=str, help='Name of the run', default='run-0')
    
    parser.add_argument('--option', type=str, help='Options for the experiment, select from [train, test, \
                        evaluate]', default='train')
    parser.add_argument('--model_type', type=str, help='Type of the model, select one from [note_sequence, \
                        baseline, proposed]', default='proposed')
    parser.add_argument('--resume_training', type=bool, help='Whether to resume training from the last \
                        checkpoint', default=False)

    # input data comparison (features and encoding)
    parser.add_argument('--features', type=str, nargs='+', help='List of features to be used, select one or \
                        more from [pitch, onset, duration, velocity]', default=['pitch', 'onset', 'duration', \
                        'velocity'])
    parser.add_argument('--pitch_encoding', type=str, help='Pitch encoding, select from [midi, chroma]', \
                        default='midi')
    parser.add_argument('--onset_encoding', type=str, help='Encoding of onset features. Select one from \
                        [absolute-raw, shift-raw, absolute-onehot, shift-onehot].', default='shift-raw')
    parser.add_argument('--duration_encoding', type=str, help='Encoding of duration features. Select one \
                        from [raw, onehot].', default='raw')

    # data augmentation parameters
    parser.add_argument('--tempo_change_prob', type=float, help='Probability of tempo change', default=1.0)
    parser.add_argument('--tempo_change_range', type=float, nargs='+', help='Range of tempo change', \
                        default=[0.8, 1.2])
    parser.add_argument('--pitch_shift_prob', type=float, help='Probability of pitch shift', default=1.0)
    parser.add_argument('--pitch_shift_range', type=float, nargs='+', help='Range of pitch shift', \
                        default=[-12, 12])
    parser.add_argument('--extra_note_prob', type=float, help='Probability of extra note', default=0.5)
    parser.add_argument('--missing_note_prob', type=float, help='Probability of missing note', default=0.5)

    # downbeats or not in note sequence model
    parser.add_argument('--downbeats', type=bool, help='Whether to output downbeats or not', default=False)

    # output data comparison
    parser.add_argument('--output_type', type=str, help='Type of output for musical onsets and note values, \
                        select from [regression, classification]', default='regression')

    # parallelization
    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing', default=8)
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use', default=4)

    parser.add_argument('--model_checkpoint', type=str, help='Path to the checkpoint to resume training \
                        from or for testing/evaluation', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    
    args = parser.parse_args()

    # ========= input check =========
    # dataset_folder
    # workspace
    # experiment_name
    # option
    # model_type
    if args.model_type not in ['note_sequence', 'baseline', 'proposed']:
        raise ValueError('Invalid model type: {}'.format(args.model_type))
    # features
    if args.model_type in ['baseline', 'proposed']:
        if args.features != ['pitch', 'onset', 'duration', 'velocity']:
            print("INFO: Invalid features for {} model. Using default features instead.".format(args.model_type))
            args.features = ['pitch', 'onset', 'duration', 'velocity']
    # workers
    if args.option == 'test' or args.option == 'evaluate':
        if args.workers >= 1:
            print('INFO: Reset number of workers to 0 for testing/evaluation.')
            args.workers = 0
    # gpus
    if args.option == 'test' or args.option == 'evaluate':
        if args.gpus > 1:
            print('INFO: Reset number of GPUs to 1 for testing/evaluation.')
            args.gpus = 1
    # verbose


    # ========= run train/test or evaluate =========
    if args.option in ['train', 'test']:
        train_or_test(args)
    elif args.option == 'evaluate':
        evaluate(args)
