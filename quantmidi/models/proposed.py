import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from quantmidi.models.model_utils import ModelUtils
from quantmidi.data.constants import tsVocabSize, keyVocabSize

learning_rate = 1e-3
dropout = 0.15
hidden_size=512
kernel_size=9
gru_layers=2

class ProposedModel(pl.LightningModule):
    def __init__(self,
        features=['pitch', 'onset', 'duration', 'velocity'],
        pitch_encoding='midi',
        onset_encoding='shift-onehot',
        duration_encoding='raw',
    ):
        super().__init__()

        self.features = features
        self.pitch_encoding = pitch_encoding
        self.onset_encoding = onset_encoding
        self.duration_encoding = duration_encoding

        in_features = ModelUtils.get_encoding_in_features(features, pitch_encoding, onset_encoding, duration_encoding)

        # ======== Beats, downbeats ========
        self.convs_beat = ConvBlock(in_features=in_features)
        self.convs_downbeat = ConvBlock(in_features=in_features)
        self.grus_beat = GRUBlock(in_features=hidden_size)
        self.grus_downbeat = GRUBlock(in_features=hidden_size+1)  # +1 for beat
        self.out_beat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
        self.out_downbeat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')

        # ======== Time signature denominator classification ========
        self.conv_ts = ConvBlock(in_features=in_features)
        self.gru_ts = GRUBlock(in_features=hidden_size+2)  # +2 for beat and downbeat
        self.out_ts = LinearOutput(in_features=hidden_size, out_features=tsVocabSize, activation_type='softmax')

        # ======== Key signature classification ========
        self.conv_key = ConvBlock(in_features=in_features)
        self.gru_key = GRUBlock(in_features=hidden_size+2)  # +2 for beat and downbeat
        self.out_key = LinearOutput(in_features=hidden_size, out_features=keyVocabSize, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, sequence_length, len(features))

        # ======== Encoding ========
        x_encoded = ModelUtils.encode_input_feature(x, self.features, self.pitch_encoding, self.onset_encoding, self.duration_encoding)
        # (batch_size, sequence_length, in_features)

        # ======== Beats, downbeats ========
        x = self.convs_beat(x_encoded)  # (batch_size, sequence_length, hidden_size)
        x = self.grus_beat(x)  # (batch_size, sequence_length, hidden_size)
        y_b = self.out_beat(x)  # (batch_size, sequence_length, 1)

        x = self.convs_downbeat(x_encoded)  # (batch_size, sequence_length, hidden_size)
        x = torch.cat((x, y_b), dim=-1)  # (batch_size, sequence_length, hidden_size+1)
        x = self.grus_downbeat(x)  # (batch_size, sequence_length, hidden_size)
        y_db = self.out_downbeat(x)  # (batch_size, sequence_length, 1)

        # ======== Time signature classification ========
        x = self.conv_ts(x_encoded)  # (batch_size, sequence_length, hidden_size)
        x = torch.cat((x, y_b, y_db), dim=-1)  # (batch_size, sequence_length, hidden_size+2)
        x = self.gru_ts(x)  # (batch_size, sequence_length, hidden_size)
        y_ts = self.out_ts(x)  # (batch_size, sequence_length, tsVocabSize)

        # ======== Key signature classification ========
        x = self.conv_key(x_encoded)  # (batch_size, sequence_length, hidden_size)
        x = torch.cat((x, y_b, y_db), dim=-1)  # (batch_size, sequence_length, hidden_size+2)
        x = self.gru_key(x)  # (batch_size, sequence_length, hidden_size)
        y_key = self.out_key(x)  # (batch_size, sequence_length, keyVocabSize)

        # squeeze and transpose
        y_b = y_b.squeeze(dim=-1)  # (batch_size, sequence_length)
        y_db = y_db.squeeze(dim=-1)  # (batch_size, sequence_length)
        y_ts = y_ts.transpose(1, 2)  # (batch_size, tsVocabSize, sequence_length)
        y_key = y_key.transpose(1, 2)  # (batch_size, keyVocabSize, sequence_length)
        
        return y_b, y_db, y_ts, y_key

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
        return [checkpoint_callback, earlystop_callback]

    def training_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, y_ts, y_key, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ts = y_ts.long()
        y_key = y_key.long()
        length = length.long()

        # predict
        y_b_hat, y_db_hat, y_ts_hat, y_key_hat = self(x)
        
        # mask out the padding part (avoid inplace operation)
        mask = torch.ones(y_b_hat.shape).to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        y_db_hat = y_db_hat * mask
        y_ts_hat = y_ts_hat * mask.unsqueeze(1)
        y_key_hat = y_key_hat * mask.unsqueeze(1)
        
        # loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ts = nn.NLLLoss()(y_ts_hat, y_ts)
        loss_key = nn.NLLLoss()(y_key_hat, y_key)
        loss = loss_b + loss_db + loss_ts + loss_key
        
        # logs
        logs = {
            'train_loss': loss,
            'train_loss_b': loss_b,
            'train_loss_db': loss_db,
            'train_loss_ts': loss_ts,
            'train_loss_key': loss_key,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, y_ts, y_key, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ts = y_ts.long()
        y_key = y_key.long()
        length = length.long()

        # predict
        y_b_hat, y_db_hat, y_ts_hat, y_key_hat = self(x)
        
        # mask out the padding part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            y_ts_hat[i, :, length[i]:] = 0
            y_key_hat[i, :, length[i]:] = 0

        # loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ts = nn.NLLLoss()(y_ts_hat, y_ts)
        loss_key = nn.NLLLoss()(y_key_hat, y_key)
        loss = loss_b + loss_db + loss_ts + loss_key
        
        # metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        precs_macro_ts, recs_macro_ts, fs_macro_ts = 0, 0, 0
        precs_weighted_ts, recs_weighted_ts, fs_weighted_ts = 0, 0, 0
        precs_macro_key, recs_macro_key, fs_macro_key = 0, 0, 0
        precs_weighted_key, recs_weighted_key, fs_weighted_key = 0, 0, 0

        for i in range(x.shape[0]):
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
            y_ts_hat_i = y_ts_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_key_hat_i = y_key_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_b_i = y_b[i, :length[i]]
            y_db_i = y_db[i, :length[i]]
            y_ts_i = y_ts[i, :length[i]]
            y_key_i = y_key[i, :length[i]]

            acc_b, prec_b, rec_b, f_b = ModelUtils.f_measure_framewise(y_b_i, y_b_hat_i)
            acc_db, prec_db, rec_db, f_db = ModelUtils.f_measure_framewise(y_db_i, y_db_hat_i)
            (
                prec_macro_ts, 
                rec_macro_ts, 
                f1_macro_ts, 
                prec_weighted_ts, 
                rec_weighted_ts, 
                f1_weighted_ts
            ) = ModelUtils.classification_report_framewise(y_ts_i, y_ts_hat_i)
            (
                prec_macro_key, 
                rec_macro_key, 
                f1_macro_key, 
                prec_weighted_key, 
                rec_weighted_key, 
                f1_weighted_key
            ) = ModelUtils.classification_report_framewise(y_key_i, y_key_hat_i)
            
            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            accs_db += acc_db
            precs_db += prec_db
            recs_db += rec_db
            fs_db += f_db

            precs_macro_ts += prec_macro_ts
            recs_macro_ts += rec_macro_ts
            fs_macro_ts += f1_macro_ts
            precs_weighted_ts += prec_weighted_ts
            recs_weighted_ts += rec_weighted_ts
            fs_weighted_ts += f1_weighted_ts

            precs_macro_key += prec_macro_key
            recs_macro_key += rec_macro_key
            fs_macro_key += f1_macro_key
            precs_weighted_key += prec_weighted_key
            recs_weighted_key += rec_weighted_key
            fs_weighted_key += f1_weighted_key
            
        # log
        logs = {
            'val_loss': loss,
            'val_loss_b': loss_b,
            'val_loss_db': loss_db,
            'val_loss_ts': loss_ts,
            'val_loss_key': loss_key,
            'val_acc_beat': accs_b / x.shape[0],
            'val_p_beat': precs_b / x.shape[0],
            'val_r_beat': recs_b / x.shape[0],
            'val_f_beat': fs_b / x.shape[0],
            'val_acc_db': accs_db / x.shape[0],
            'val_p_db': precs_db / x.shape[0],
            'val_r_db': recs_db / x.shape[0],
            'val_f_db': fs_db / x.shape[0],
            'val_p_macro_ts': precs_macro_ts / x.shape[0],
            'val_r_macro_ts': recs_macro_ts / x.shape[0],
            'val_f_macro_ts': fs_macro_ts / x.shape[0],
            'val_p_weighted_ts': precs_weighted_ts / x.shape[0],
            'val_r_weighted_ts': recs_weighted_ts / x.shape[0],
            'val_f_weighted_ts': fs_weighted_ts / x.shape[0],
            'val_p_macro_key': precs_macro_key / x.shape[0],
            'val_r_macro_key': recs_macro_key / x.shape[0],
            'val_f_macro_key': fs_macro_key / x.shape[0],
            'val_p_weighted_key': precs_weighted_key / x.shape[0],
            'val_r_weighted_key': recs_weighted_key / x.shape[0],
            'val_f_weighted_key': fs_weighted_key / x.shape[0],
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}


class ConvBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.conv = nn.Sequential(
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
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)

        x = x.unsqueeze(1)   # (batch_size, 1, sequence_length, in_features)
        x = self.conv(x)   # (batch_size, hidden_size, sequence_length, 1)
        x = x.squeeze(3).transpose(1, 2)  # (batch_size, sequence_length, hidden_size)

        return x


class GRUBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.grus_beat = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        x, _ = self.grus_beat(x)  # (batch_size, sequence_length, hidden_size*2)
        x = self.linear(x)  # (batch_size, sequence_length, hidden_size)

        return x


class LinearOutput(nn.Module):
    def __init__(self, in_features, out_features, activation_type='sigmoid'):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)
        if activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'softmax':
            self.activation = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)

        x = self.dropout(x)  # (batch_size, sequence_length, in_features)
        x = self.linear(x)  # (batch_size, sequence_length, out_features)
        x = self.activation(x)  # (batch_size, sequence_length, out_features)

        return x