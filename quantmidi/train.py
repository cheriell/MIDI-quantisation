import argparse
from pytorch_lightning import Trainer
from pathlib import Path




def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Main programme for model training.')
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Path to the dataset folders in the order \
                        of ASAP, A_MAPS, CPM, ACPAS')
    parser.add_argument('--workspace', type=str, help='Path to the workspace')

    parser.add_argument('--experiment_name', type=str, help='Name of the experiment', default='full-training')
    parser.add_argument('--model_type', type=str, help='Type of the model', default='best')

    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    # ========= input check =========
    # dataset_folder
    # workspace
    # experiment_name
    # model_type
    # workers
    # verbose

def train(args):

    datamodule = QuantMIDIDataModule()
    model = QuantMIDIModel(model_type)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(Path(args.workspace, 'tflogs', args.experiment_name)),
        name=args.experiment_name,
    )

    trainer = Trainer(
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