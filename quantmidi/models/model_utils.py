import torch
import torch.nn.functional as F
import numpy as np

from quantmidi.data.constants import resolution

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

    @staticmethod
    def get_pianoroll_from_batch_data(x, length):
        # x.shape = (batch_size, length, len(features)), batch_size = 1
        assert x.shape[0] == 1, 'Batch size must be 1.'
        assert x.shape[2] == 4, 'Number of features must be 4.'

        pr_length = torch.max(length).long()
        pr = torch.zeros(1, 128, pr_length).float().to(x.device)

        for i in range(x.shape[1]):
            start = torch.round(x[0,i,1] * (1 / resolution)).long()
            if start < pr_length:
                end = torch.round((x[0,i,1] + x[0,i,2]) * (1 / resolution)).long()
                pr[0,x[0,i,0].long(),start:min(end, pr_length)] = x[0,i,3] / 127.0

        return pr

    @staticmethod
    def f_measure_framewise(y, y_hat):
        acc = (y_hat == y).float().mean()
        TP = torch.logical_and(y_hat==1, y==1).float().sum()
        FP = torch.logical_and(y_hat==1, y==0).float().sum()
        FN = torch.logical_and(y_hat==0, y==1).float().sum()

        p = TP / (TP + FP + np.finfo(float).eps)
        r = TP / (TP + FN + np.finfo(float).eps)
        f = 2 * p * r / (p + r + np.finfo(float).eps)
        return acc, p, r, f