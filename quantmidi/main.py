import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.abspath('./'))
import argparse
import pytorch_lightning as pl
pl.seed_everything(42)
from pathlib import Path

from quantmidi.data import QuantMIDIDataModule
from quantmidi.model import QuantMIDIModel

## -------------------------
## DEBUGGING BLOCK
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.autograd.set_detect_anomaly(True)
## END DEBUGGING BLOCK
## -------------------------

def main():
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
    parser.add_argument('--model_type', type=str, help='Type of the model, select from [note_sequence], \
                        [pianoroll]', default='note_sequence')

    # input data comparison (features and encoding)
    parser.add_argument('--features', type=str, nargs='+', help='List of features to be used, select one or \
                        more from [pitch, onset, duration, velocity]', default=['onset', 'duration'])
    parser.add_argument('--pitch_encoding', type=str, help='Pitch encoding, select from [midi, chroma]', \
                        default='midi')
    parser.add_argument('--onset_encoding', type=str, help='Encoding of onset features. Select one from \
                        [absolute-raw, shift-raw, absolute-onehot, shift-onehot].', default='shift-raw')
    parser.add_argument('--duration_encoding', type=str, help='Encoding of duration features. Select one \
                        from [raw, onehot].', default='raw')

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
    # model_type
    if args.model_type not in ['note_sequence', 'pianoroll']:
        raise ValueError('Invalid model type: {}'.format(args.model_type))
    # workers
    # verbose
    
    # ========= create workspace =========
    feature_folder = str(Path(args.workspace, 'features'))
    tracking_uri = str(Path(args.workspace, 'mlruns'))

    # ========= create dataset, model, logger =========
    datamodule = QuantMIDIDataModule(feature_folder=feature_folder, model_type=args.model_type, 
                                    workers=args.workers)
    model = QuantMIDIModel(
        model_type=args.model_type,
        features=args.features,
        pitch_encoding=args.pitch_encoding,
        onset_encoding=args.onset_encoding,
        duration_encoding=args.duration_encoding,
    )
    logger = pl.loggers.MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=tracking_uri,
        run_name=args.run_name,
        tags={
            'model_type': args.model_type,
            'features': ','.join(args.features),
            'pitch_encoding': args.pitch_encoding,
            'onset_encoding': args.onset_encoding,
            'duration_encoding': args.duration_encoding,
        },
    )

    # ========= create trainer =========
    trainer = pl.Trainer(
        default_root_dir=tracking_uri,
        logger=logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        gpus=args.gpus,
        # resume_from_checkpoint=args.model_checkpoint,
    )

    # ========= train/test =========
    if args.option == 'train':
        trainer.fit(model, datamodule=datamodule)
    elif args.option == 'test':
        trainer.test(model, ckpt_path=args.model_checkpoint, datamodule=datamodule)

if __name__ == '__main__':
    main()