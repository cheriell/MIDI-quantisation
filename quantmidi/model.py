import pytorch_lightning as pl
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from quantmidi.data import resolution

learning_rate = 1e-3
dropout = 0.15

class QuantMIDIModel(pl.LightningModule):
    def __init__(self, 
        model_type='note_sequence',
        features=['onset', 'duration'],
        pitch_encoding='midi',
        onset_encoding='shift-raw',
        duration_encoding='raw',
    ):
        super().__init__()

        self.features = features

        if model_type == 'note_sequence':
            self.model = QuantMIDISequenceModel(
                features=features,
                pitch_encoding=pitch_encoding,
                onset_encoding=onset_encoding,
                duration_encoding=duration_encoding,
            )
        elif model_type == 'pianoroll':
            self.model = QuantMIDIBaselineModel()
        
    def forward(self, x):
        return self.model(x)
    
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
            save_top_k=9,
            filename='{epoch}-{val_f1:.4f}',
            save_last=True,
        )
        earlystop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=500,
            mode='max',
        )
        return [checkpoint_callback]#, earlystop_callback]


class QuantMIDISequenceModel(nn.Module):
    def __init__(self, 
        features=['onset', 'duration'],
        pitch_encoding='midi',
        onset_encoding='shift-raw',
        duration_encoding='raw',
        hidden_size=512,
        kernel_size=9,
        gru_layers=2,
        dropout=dropout,
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

class QuantMIDIBaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x


class ModelUtils():

    @staticmethod
    def input_feature_ablation(x, features):
        """
        Remove features from input.

        Args:
            x (torch.Tensor): Input tensor.
            features (list): List of features to keep.

        Returns:
            torch.Tensor: Tensor with features removed.
        """
        feature2idx = {'pitch': 0, 'onset': 1, 'duration': 2, 'velocity': 3}
        x_list = []
        for feature, idx in feature2idx.items():
            if feature in features:
                x_list.append(x[:,:,idx:idx+1].clone())
        x = torch.cat(x_list, dim=2)
        return x
        
    @staticmethod
    def encode_input_feature(x, features, pitch_encoding, onset_encoding, duration_encoding):
        """
        Encode input features.

        Args:
            x (torch.Tensor): Input tensor.
            features (list): List of features.
            pitch_encoding (str): Encoding of the pitch. Select one from ['midi', 'chroma'].
            onset_encoding (str): Encoding of onset features. Select one from ['absolute-raw', 'shift-raw',
                            'absolute-onehot', 'shift-onehot'].
            duration_encoding (str): Encoding of duration features. Select one from ['raw', 'onehot'].

        Returns:
            torch.Tensor: Tensor with features encoded.
        """
        feature2idx = dict([(feature, idx) for idx, feature in enumerate(features)])
        x_list = []

        # ======== Encode ========
        for feature in features:
            x_feature = x[:,:,feature2idx[feature]].clone()

            # pitch
            if feature == 'pitch':
                if pitch_encoding == 'midi':
                    vocab_size = 128
                elif pitch_encoding == 'chroma':
                    vocab_size = 12
                    x_feature = x_feature % 12
                x_encoded = F.one_hot(x_feature.long(), vocab_size).float()

            # onset
            elif feature == 'onset':
                if onset_encoding == 'absolute-raw':
                    # restart from 0s
                    x_feature = x_feature - x_feature[:,0:1].clone()
                    x_encoded = x_feature.float().unsqueeze(2)

                elif onset_encoding == 'shift-raw':
                    onsets_shift = x_feature[:,1:] - x_feature[:,:-1]
                    # set first onset shift to 0
                    x_encoded = torch.zeros(x.shape[0], x.shape[1], 1).float().to(x.device)  
                    x_encoded[:,1:,0] = onsets_shift

                elif onset_encoding == 'absolute-onehot':
                    # restart from 0s
                    x_feature = x_feature - x_feature[:,0:1].clone()
                    # restart every 20s
                    x_feature = x_feature % 20.0
                    # to one hot index
                    x_feature = torch.round(x_feature / resolution).long()
                    # to one hot
                    x_encoded = F.one_hot(x_feature, int(20.0 / resolution) + 1).float()
                    
                elif onset_encoding == 'shift-onehot':
                    onsets_shift_raw = x_feature[:,1:] - x_feature[:,:-1]
                    # set first onset shift to 0
                    onsets_shift = torch.zeros(x.shape[0], x.shape[1]).float().to(x.device)
                    onsets_shift[:,1:] = onsets_shift_raw
                    # maximum filter - set maximum onset shift to 4s
                    onsets_shift_filted = onsets_shift.clone()
                    onsets_shift_filted[onsets_shift > 4.0] = 4.0
                    # remove negative values introduced by zero padding
                    onsets_shift_filted[onsets_shift < 0.0] = 0.0
                    # to one hot index
                    onsets_shift_idx = torch.round(onsets_shift_filted / resolution).long()
                    # to one hot
                    x_encoded = F.one_hot(onsets_shift_idx, int(4.0 / resolution) + 1).float()

            # duration
            elif feature == 'duration':
                if duration_encoding == 'raw':
                    x_encoded = x_feature.float().unsqueeze(2)
                elif duration_encoding == 'onehot':
                    # maximum filter - set maximum duration to 4s
                    durations_filted = x_feature.clone()
                    durations_filted[durations_filted > 4.0] = 4.0
                    # to one hot index
                    durations_idx = torch.round(durations_filted / resolution).long()
                    # to one hot
                    x_encoded = F.one_hot(durations_idx, int(4.0 / resolution) + 1).float()

            # velocity
            elif feature == 'velocity':
                x_encoded = x_feature.float().unsqueeze(2) / 127.0

            x_list.append(x_encoded)

        x = torch.cat(x_list, dim=2)
        return x

    @staticmethod
    def get_encoding_in_features(features, pitch_encoding, onset_encoding, duration_encoding):
        """
        Get encoding in features.

        Args:
            features (list): List of features.
            pitch_encoding (str): Encoding of pitch features. Select one from ['midi', 'chroma'].
            onset_encoding (str): Encoding of onset features. Select one from ['absolute-raw', 'shift-raw',
                            'absolute-onehot', 'shift-onehot'].
            duration_encoding (str): Encoding of duration features. Select one from ['raw', 'onehot'].

        Returns:
            int: Number of in_features after encoding.
        """
        feature2idx = dict([(feature, idx) for idx, feature in enumerate(features)])
        in_features = 0

        for feature in features:
            # pitch
            if feature == 'pitch':
                if pitch_encoding == 'midi':
                    in_features += 128
                elif pitch_encoding == 'chroma':
                    in_features += 12

            # onset
            elif feature == 'onset':
                if onset_encoding == 'absolute-raw':
                    in_features += 1
                elif onset_encoding == 'shift-raw':
                    in_features += 1
                elif onset_encoding == 'absolute-onehot':
                    in_features += int(20.0 / resolution) + 1  # maximum 20s
                elif onset_encoding == 'shift-onehot':
                    in_features += int(4.0 / resolution) + 1   # maximum 4s

            # duration
            elif feature == 'duration':
                if duration_encoding == 'raw':
                    in_features += 1
                elif duration_encoding == 'onehot':
                    in_features += int(4.0 / resolution) + 1   # maximum 4s

            # velocity
            elif feature == 'velocity':
                in_features += 1

        return in_features