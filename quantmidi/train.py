import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.abspath('./'))
import argparse
import pytorch_lightning as pl
from pathlib import Path

from quantmidi.data import QuantMIDIDataModule


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Main programme for model training.')
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Path to the dataset folders \
                        in the order of ASAP, A_MAPS, CPM, ACPAS')
    parser.add_argument('--workspace', type=str, help='Path to the workspace')

    # experiment settings
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment', default='full-training')
    parser.add_argument('--model_type', type=str, help='Type of the model, select from [note_sequence], \
                        [pianoroll]', default='note_sequence')

    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing')
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

    train(args)

def train(args):

    feature_folder = args.workspace + '/features'

    datamodule = QuantMIDIDataModule(feature_folder=feature_folder, model_type=args.model_type, 
                                    workers=args.workers)
    model = QuantMIDIModel(args.model_type)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(Path(args.workspace, 'tflogs', args.experiment_name)),
        name=args.experiment_name,
    )

    trainer = pl.Trainer(
        default_save_path=str(Path(args.workspace, 'models', args.experiment_name)),
        logger=logger,
        log_every_n_steps=100,
        reload_dataloaders_every_epoch=True,
        auto_select_gpus=True,
        gpus=[0,1,2,3],
        # resume_from_checkpoint='19/0f4d93088716431fb52854d9162e9582/checkpoints/last.ckpt',
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()