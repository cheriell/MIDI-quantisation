import torch
import pytorch_lightning as pl

from quantmidi.data.dataset import batch_size, QuantMIDIDataset

class QuantMIDIDataModule(pl.LightningDataModule):
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
