import pandas as pd
import pickle
import torch
from pathlib import Path
import matplotlib.pyplot as plt

feature_folder = '/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/features'
metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))

ibis = torch.Tensor([])

for i, row in metadata.iterrows():
    note_sequence, annotations = pickle.load(open(str(Path(feature_folder, row['feature_file'])), 'rb'))
    beats = annotations['beats']
    ibis = torch.cat((ibis, beats[1:] - beats[:-1]))

ibis = ibis.numpy()

plt.figure()
plt.hist(ibis, bins=100)
plt.savefig('debug.png')

print(ibis.mean(), ibis.min(), ibis.max())

