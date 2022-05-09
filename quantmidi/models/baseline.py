import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from quantmidi.models.model_utils import ModelUtils

learning_rate = 1e-3
dropout = 0.1

class BaselineModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # ========== ConvBlock frontend ==========
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
        )

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

        # ========== Linear output layer and sigmoid activation ==========
        self.out_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(20, 2),
            nn.Sigmoid()
        )

    def forward(self, x, length):
        # x.shape = (batch_size, note_sequence_length, len(features)), batch_size = 1

        # ======== get piano roll ==========
        x = ModelUtils.get_pianoroll_from_batch_data(x, length)  # (1, 128, pr_length)
        x = x.unsqueeze(1)  # (1, 1, 128, pr_length)

        # ======== ConvBlock frontend ==========
        x = self.convs(x)  # (1, 20, 1, pr_length)
        x = x.squeeze(2)  # (1, 20, pr_length)
        
        # ======== TCN block 11 alyers, 1*2 dilation, 20 channels ==========
        for i in range(self.tcn_layers):
            x, x_skip = self.tcns[i](x)  # (1, 20, pr_length)
        x = self.relu(x)  # (1, 20, pr_length)
        x = x.transpose(1, 2)  # (1, pr_length, 20)

        # ======== Linear output layer and sigmoid activation ==========
        x = self.out_layer(x)  # (1, pr_length, 2)
        x = x[:,:,0]  # (1, pr_length)

        return x

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

    def training_step(self, batch, batch_idx):
        # data
        x, y, length = batch
        x = x.float()
        y = y.float()

        # predict
        y_hat = self(x, length)

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
        y_hat = self.forward(x, length)

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
