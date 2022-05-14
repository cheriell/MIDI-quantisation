import pandas as pd
import pickle
from pathlib import Path

feature_folder = '/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/features'
metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))

tsNumerators = set()

for i, row in metadata.iterrows():
    note_sequence, annotations = pickle.load(open(str(Path(feature_folder, row['feature_file'])), 'rb'))
    tsNumerators.update(set(annotations['time_signatures'][:,1].tolist()))

print(tsNumerators)
