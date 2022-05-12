import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from quantmidi.models.model_utils import ModelUtils
from quantmidi.models.proposed import ConvBlock, GRUBlock, LinearOutput

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
        downbeats=False,
        tempos=False,
        reverse_link=False,
    ):
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
        self.downbeats = downbeats
        self.reverse_link = reverse_link
        
        in_features = ModelUtils.get_encoding_in_features(features, pitch_encoding, onset_encoding, duration_encoding)

        self.convs_beat = ConvBlock(in_features=in_features)
        self.grus_beat = GRUBlock(in_features=hidden_size)
        self.out_beat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')

        if self.downbeats:
            self.convs_downbeat = ConvBlock(in_features=in_features)
            self.grus_downbeat = GRUBlock(in_features=hidden_size+1)  # +1 for beat
            self.out_downbeat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')

    def forward(self, x):
        # x.shape = (batch_size, max_length, len(features))

        # ======== Encoding ========
        x_encoded = ModelUtils.encode_input_feature(x, self.features, self.pitch_encoding, self.onset_encoding, self.duration_encoding)
        # (batch_size, max_length, in_features)

        x = self.convs_beat(x_encoded)  # (batch_size, sequence_length, hidden_size)
        x = self.grus_beat(x)  # (batch_size, sequence_length, hidden_size)
        y_b = self.out_beat(x)  # (batch_size, sequence_length, 1)

        if self.downbeats:
            x = self.convs_downbeat(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x = torch.cat((x, y_b), dim=-1)  # (batch_size, sequence_length, hidden_size+1)
            x = self.grus_downbeat(x)  # (batch_size, sequence_length, hidden_size)
            y_db = self.out_downbeat(x)  # (batch_size, sequence_length, 1)

            # squeeze and transpose
            y_b = y_b.squeeze(dim=-1)  # (batch_size, sequence_length)
            y_db = y_db.squeeze(dim=-1)  # (batch_size, sequence_length)
            
            if self.reverse_link:
                return y_db, y_b
            else:
                return y_b, y_db

        else:
            return y_b.squeeze(dim=-1)  # (batch_size, sequence_length)

    def training_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()

        # predict
        x = ModelUtils.input_feature_ablation(x, self.features)
        if self.downbeats:
            y_b_hat, y_db_hat = self(x)
        else:
            y_b_hat = self(x)

        # mask out the padding part (avoid inplace operation)
        mask = torch.ones(y_b_hat.shape).to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        if self.downbeats:
            y_db_hat = y_db_hat * mask

        # loss and logs
        if self.downbeats:
            loss_b = F.binary_cross_entropy(y_b_hat, y_b)
            loss_db = F.binary_cross_entropy(y_db_hat, y_db)
            loss = loss_b + loss_db
            logs = {
                'train_loss': loss,
                'train_loss_b': loss_b,
                'train_loss_db': loss_db,
            }
        else:
            loss = F.binary_cross_entropy(y_b_hat, y_b)
            logs = {'train_loss': loss}

        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()

        # predict
        x = ModelUtils.input_feature_ablation(x, self.features)
        if self.downbeats:
            y_b_hat, y_db_hat = self(x)
        else:
            y_b_hat = self(x)

        # mask out the padded part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            if self.downbeats:
                y_db_hat[i, length[i]:] = 0

        # loss
        if self.downbeats:
            loss_b = F.binary_cross_entropy(y_b_hat, y_b)
            loss_db = F.binary_cross_entropy(y_db_hat, y_db)
            loss = loss_b + loss_db
        else:
            loss = F.binary_cross_entropy(y_b_hat, y_b)

        # metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        if self.downbeats:
            accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        for i in range(x.shape[0]):
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_b_i = y_b[i, :length[i]]
            acc_b, prec_b, rec_b, f_b = ModelUtils.f_measure_framewise(y_b_i, y_b_hat_i)

            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            if self.downbeats:
                y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
                y_db_i = y_db[i, :length[i]]
                acc_db, prec_db, rec_db, f_db = ModelUtils.f_measure_framewise(y_db_i, y_db_hat_i)
                
                accs_db += acc_db
                precs_db += prec_db
                recs_db += rec_db
                fs_db += f_db

        # log
        logs = {
            'val_loss': loss,
            'val_loss_b': loss_b,
            'val_acc_beat': accs_b / x.shape[0],
            'val_p_beat': precs_b / x.shape[0],
            'val_r_beat': recs_b / x.shape[0],
            'val_f_beat': fs_b / x.shape[0],
        }
        if self.downbeats:
            logs.update({
                'val_loss_db': loss_db,
                'val_acc_db': accs_db / x.shape[0],
                'val_p_db': precs_db / x.shape[0],
                'val_r_db': recs_db / x.shape[0],
                'val_f_db': fs_db / x.shape[0],
            })
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}

    def test_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()

        # predict
        x = ModelUtils.input_feature_ablation(x, self.features)
        if self.downbeats:
            y_b_hat, y_db_hat = self(x)
        else:
            y_b_hat = self(x)

        # mask out the padded part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            if self.downbeats:
                y_db_hat[i, length[i]:] = 0

        # loss
        if self.downbeats:
            loss_b = F.binary_cross_entropy(y_b_hat, y_b)
            loss_db = F.binary_cross_entropy(y_db_hat, y_db)
            loss = loss_b + loss_db
        else:
            loss = F.binary_cross_entropy(y_b_hat, y_b)

        # metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        if self.downbeats:
            accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        for i in range(x.shape[0]):
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_b_i = y_b[i, :length[i]]
            acc_b, prec_b, rec_b, f_b = ModelUtils.f_measure_framewise(y_b_i, y_b_hat_i)

            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            if self.downbeats:
                y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
                y_db_i = y_db[i, :length[i]]
                acc_db, prec_db, rec_db, f_db = ModelUtils.f_measure_framewise(y_db_i, y_db_hat_i)
                
                accs_db += acc_db
                precs_db += prec_db
                recs_db += rec_db
                fs_db += f_db

        # log
        logs = {
            'test_loss': loss,
            'test_loss_b': loss_b,
            'test_acc_beat': accs_b / x.shape[0],
            'test_p_beat': precs_b / x.shape[0],
            'test_r_beat': recs_b / x.shape[0],
            'test_f_beat': fs_b / x.shape[0],
        }
        if self.downbeats:
            logs.update({
                'test_loss_db': loss_db,
                'test_acc_db': accs_db / x.shape[0],
                'test_p_db': precs_db / x.shape[0],
                'test_r_db': recs_db / x.shape[0],
                'test_f_db': fs_db / x.shape[0],
            })
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
            monitor='val_f_beat',
            mode='max',
            save_top_k=3,
            filename='{epoch}-{val_f_beat:.4f}',
            save_last=True,
        )
        earlystop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f_beat',
            patience=200,
            mode='max',
        )
        return [checkpoint_callback]#, earlystop_callback]
