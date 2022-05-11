import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from quantmidi.models.model_utils import ModelUtils
from quantmidi.data.constants import keyVocabSize, tsVocabSize

learning_rate = 1e-3
dropout = 0.1

class BaselineModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # ======== ConvBlock frontend ==========
        self.conv_frontend = ConvFrontEnd()

        # ========== TCN block 11 alyers, 1*2 dilation, 20 channels ==========
        self.tcn_layers = 11
        self.tcns = nn.ModuleList()

        for i in range(self.tcn_layers):
            tcn = TCNLayer(
                in_channels=20,
                out_channels=20,
                kernel_size=5,
                dilation=2 ** i,
                p_dropout=dropout
            )
            self.tcns.append(tcn)

        self.relu = nn.ELU()

        # ========== Linear output layers =============
        # beat and downbeat activation functions
        self.out_act = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(20, 2),
            nn.Sigmoid()
        )
        # time signature denominator classification
        self.out_ts = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(20, tsVocabSize),
            nn.LogSoftmax(dim=2)
        )

        # =========== Key signature classification ==========

        self.conv_frontend_key = ConvFrontEnd(out_features=18)

        self.tcn_layers_key = 5
        self.tcns_key = nn.ModuleList()

        for i in range(self.tcn_layers_key):
            tcn = TCNLayer(
                in_channels=20,
                out_channels=20,
                kernel_size=5,
                dilation=2 ** i,
                p_dropout=dropout
            )
            self.tcns_key.append(tcn)

        self.relu_key = nn.ELU()

        self.out_key = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(20, keyVocabSize),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x, length):
        # x.shape = (batch_size, note_sequence_length, len(features)), batch_size = 1

        # ======== get piano roll ==========
        pr = ModelUtils.get_pianoroll_from_batch_data(x, length)  # (batch_size, 128, pr_length)
        pr = pr.unsqueeze(1)  # (batch_size, 1, 128, pr_length)
        
        # ======== Beats, downbeats ==========
        ## ConvBlock frontend
        x = self.conv_frontend(pr)  # (batch_size, 20, 1, pr_length)
        x = x.squeeze(2)  # (batch_size, 20, pr_length)
        
        ## TCN block 11 alyers, 1*2 dilation, 20 channels
        for i in range(self.tcn_layers):
            x, x_skip = self.tcns[i](x)  # (batch_size, 20, pr_length)
        x = self.relu(x)  # (batch_size, 20, pr_length)
        x = x.transpose(1, 2)  # (batch_size, pr_length, 20)

        ## Linear output layers
        # beats and downbeats activation functions
        y_act = self.out_act(x)  # (batch_size, pr_length, 2)
        y_b = y_act[:,:,0]  # (batch_size, pr_length)
        y_db = y_act[:,:,1]  # (batch_size, pr_length)
        # time signature denominator classification
        y_ts = self.out_ts(x)  # (batch_size, pr_length, tsVocabSize)
        y_ts = y_ts.transpose(1, 2)  # (batch_size, tsVocabSize, pr_length)

        # ======== Key signature classification ==========
        x = self.conv_frontend_key(pr)  # (batch_size, 18, 1, pr_length)
        x = x.squeeze(2)  # (batch_size, 18, pr_length)

        # concatenate beats and downbeat output
        x = torch.cat((x, y_act.transpose(1, 2)), dim=1)  # (batch_size, 20, pr_length)

        for i in range(self.tcn_layers_key):
            x, x_skip = self.tcns_key[i](x)  # (batch_size, 20, pr_length)
        x = self.relu_key(x)  # (batch_size, 20, pr_length)
        x = x.transpose(1, 2)  # (batch_size, pr_length, 20)

        # key signature classification
        y_key = self.out_key(x)  # (batch_size, pr_length, keyVocabSize)
        y_key = y_key.transpose(1, 2)  # (batch_size, keyVocabSize, pr_length)

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

        # predict
        y_b_hat, y_db_hat, y_ts_hat, y_key_hat = self(x, length)

        # mask out the padded part (avoid inplace operation)
        mask = torch.ones(y_b_hat.shape).float().to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        y_db_hat = y_db_hat * mask
        y_ts_hat = y_ts_hat * mask.unsqueeze(1)
        y_key_hat = y_key_hat * mask.unsqueeze(1)

        # compute loss
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

        return {'loss': loss, 'logs': logs}

    
    def validation_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, y_ts, y_key, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ts = y_ts.long()
        y_key = y_key.long()

        # predict
        y_b_hat, y_db_hat, y_ts_hat, y_key_hat = self.forward(x, length)

        # mask out the padded part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            y_ts_hat[i, :, length[i]:] = 0
            y_key_hat[i, :, length[i]:] = 0

        # compute loss
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

class ConvFrontEnd(nn.Module):
    def __init__(self, out_features=20):
        super(ConvFrontEnd, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(20),
            nn.MaxPool2d((3, 1)),
            nn.ELU(),
            nn.Dropout(p=dropout),

            nn.Conv2d(
                in_channels=20,
                out_channels=20,
                kernel_size=(20, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(20),
            nn.MaxPool2d((3, 1)),
            nn.ELU(),
            nn.Dropout(p=dropout),

            nn.Conv2d(
                in_channels=20,
                out_channels=out_features,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_features),
            nn.MaxPool2d((3, 1)),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.convs(x)

class TCNLayer(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        p_dropout
    ):
        super().__init__()

        self.conv1_dilated = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=in_channels
        )
        self.conv2_dilated = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * 2 * (kernel_size - 1) // 2,
            dilation=dilation * 2,
            groups=in_channels
        )
        self.relu = nn.ELU()
        self.bn = nn.BatchNorm1d(out_channels*2)
        self.spatial_dropout = nn.Dropout2d(p_dropout)
        
        self.conv_res = nn.Conv1d(in_channels, out_channels, 1)
        self.conv_reshape = nn.Conv1d(out_channels*2, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1_dilated(x)
        x2 = self.conv2_dilated(x)
        x_concat = torch.cat([x1, x2], dim=1)
        x_concat = self.spatial_dropout(self.bn(self.relu(x_concat)))
        x_skip = self.conv_reshape(x_concat)  # forward to skip connection

        x_res = self.conv_res(x)
        x_out = x_res + x_skip  # forward to next TCN layer

        return x_out, x_skip
