import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from quantmidi.models.model_utils import ModelUtils

learning_rate = 1e-3
dropout = 0.15
hidden_size=512
kernel_size=9
gru_layers=2

class NoteSequenceModel(pl.LightningModule):
    def __init__(self,
        features=['pitch', 'onset', 'duration', 'velocity'],
        pitch_encoding='midi',
        onset_encoding='shift-onehot',
        duration_encoding='raw',
    ):
        super().__init__()

        self.features = features


        """
        Model for quantization of MIDI note sequences.

        Args:
            features (list): List of features to use as input. Select one or more from [pitch, onset,
                            duration, velocity].
            pitch_encoding (str): Encoding of the pitch. Select one from ['midi', 'chroma'].
            onset_encoding (str): Encoding of onset features. Select one from ['absolute-raw', 'shift-raw',
                            'absolute-onehot', 'shift-onehot'].
            duration_encoding (str): Encoding of duration features. Select one from ['raw', 'onehot'].
            hidden_size (int): Size of the hidden state of the GRU.
            kernel_size (int): Size of the convolutional kernel.
            gru_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.features = features
        self.pitch_encoding = pitch_encoding
        self.onset_encoding = onset_encoding
        self.duration_encoding = duration_encoding
        
        in_features = ModelUtils.get_encoding_in_features(features, pitch_encoding, onset_encoding, duration_encoding)

        # ======== CNNBlock ========
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=hidden_size // 4, 
                kernel_size=(kernel_size, in_features),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 4,
                out_channels=hidden_size // 2,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 2,
                out_channels=hidden_size,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ELU(),
        )

        # ========= GRUBlock ========
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

        # ======== OutputBlock ========
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x.shape = (batch_size, max_length, len(features))

        # ======== Encoding ========
        x = ModelUtils.encode_input_feature(x, self.features, self.pitch_encoding, self.onset_encoding, self.duration_encoding)
        # x.shape = (batch_size, max_length, in_features)

        # ======== CNNBlock ========
        x = x.unsqueeze(1)  # (batch_size, 1, max_length, in_features)
        x = self.conv_layers(x)  # (batch_size, hidden_size, max_length, 1)
        x = x.squeeze(3).transpose(1, 2)  # (batch_size, max_length, hidden_size)

        # ========= GRUBlock ========
        x, _ = self.gru(x)  # (batch_size, max_length, hidden_size * 2)
        x = self.linear(x)  # (batch_size, max_length, hidden_size)

        # ======== OutputBlock ========
        x = self.output_layer(x)  # (batch_size, max_length, 1)

        return x.squeeze(2)  # (batch_size, max_length)
    
    def training_step(self, batch, batch_idx):
        # data
        x, y, length = batch
        x = x.float()
        y = y.float()

        # predict
        x = ModelUtils.input_feature_ablation(x, self.features)
        y_hat = self(x)

        # mask out the padded part (avoid inplace operation)
        mask = torch.ones(y_hat.shape).float().to(y_hat.device)
        for i in range(y_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_hat = y_hat * mask

        # compute loss
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss, 'logs': {'train_loss': loss}}
    
    def validation_step(self, batch, batch_idx):
        # data
        x, y, length = batch
        x = x.float()
        y = y.float()

        # predict
        x = ModelUtils.input_feature_ablation(x, self.features)
        y_hat = self.forward(x)

        # mask out the padded part
        for i in range(y_hat.shape[0]):
            y_hat[i, length[i]:] = 0

        # compute loss
        loss = F.binary_cross_entropy(y_hat, y)

        # metrics
        accs, precs, recs, f1s = 0, 0, 0, 0
        for i in range(x.shape[0]):
            y_hat_i = torch.round(y_hat[i, :length[i]])
            y_i = y[i, :length[i]]

            accs += (y_hat_i == y_i).float().mean()
            TP = torch.logical_and(y_hat_i==1, y_i==1).float().sum()
            FP = torch.logical_and(y_hat_i==1, y_i==0).float().sum()
            FN = torch.logical_and(y_hat_i==0, y_i==1).float().sum()

            p = TP / (TP + FP + np.finfo(float).eps)
            r = TP / (TP + FN + np.finfo(float).eps)
            f1 = 2 * p * r / (p + r + np.finfo(float).eps)
            precs += p
            recs += r
            f1s += f1

        # log
        logs = {
            'val_loss': loss,
            'val_acc': accs / x.shape[0],
            'val_p': precs / x.shape[0],
            'val_r': recs / x.shape[0],
            'val_f1': f1s / x.shape[0],
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}

    def test_step(self, batch, batch_idx):
        # data
        x, y, length = batch
        x = x.float()
        y = y.float()

        # predict
        x = ModelUtils.input_feature_ablation(x, self.features)
        y_hat = self.forward(x)

        # mask out the padded part
        for i in range(y_hat.shape[0]):
            y_hat[i, length[i]:] = 0

        # compute loss
        loss = F.binary_cross_entropy(y_hat, y)

        # metrics
        accs, precs, recs, f1s = 0, 0, 0, 0
        for i in range(x.shape[0]):
            y_hat_i = torch.round(y_hat[i, :length[i]])
            y_i = y[i, :length[i]]

            accs += (y_hat_i == y_i).float().mean()
            TP = torch.logical_and(y_hat_i==1, y_i==1).float().sum()
            FP = torch.logical_and(y_hat_i==1, y_i==0).float().sum()
            FN = torch.logical_and(y_hat_i==0, y_i==1).float().sum()

            p = TP / (TP + FP + np.finfo(float).eps)
            r = TP / (TP + FN + np.finfo(float).eps)
            f1 = 2 * p * r / (p + r + np.finfo(float).eps)
            precs += p
            recs += r
            f1s += f1

        # log
        logs = {
            'test_loss': loss,
            'test_acc': accs / x.shape[0],
            'test_p': precs / x.shape[0],
            'test_r': recs / x.shape[0],
            'test_f1': f1s / x.shape[0],
        }
        self.log_dict(logs, prog_bar=True)

        return {'test_loss': loss, 'logs': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=(0.8, 0.8),
            eps=1e-4,
            weight_decay=0.01,
        )
        scheduler_lrdecay = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10000,
            gamma=0.1,
        )
        return [optimizer], [scheduler_lrdecay]
    
    def configure_callbacks(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=3,
            filename='{epoch}-{val_f1:.4f}',
            save_last=True,
        )
        earlystop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=200,
            mode='max',
        )
        return [checkpoint_callback, earlystop_callback]
