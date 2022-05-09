#!/bin/bash

WORKSPACE="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation"

# =============== Download datasets ===============
# path to the folder where the datasets will be stored
ASAP="/import/c4dm-datasets/A2S_transcription/working/datasets/asap-dataset-master"
A_MAPS="/import/c4dm-datasets/A2S_transcription/working/datasets/A-MAPS_1.1"
CPM="/import/c4dm-datasets/A2S_transcription/working/datasets/CPM"
ACPAS="/import/c4dm-datasets/A2S_transcription/working/datasets/ACPAS"

# TODO: script to download the dataset

# # =============== Feature preparation ================
# echo ">>> Preparing the features"

# python3 quantmidi/feature_preparation.py \
#     --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
#     --feature_folder $WORKSPACE/features \
#     --workers -1 \
#     --verbose

# =============== Training/Testing ===============

experiment_name="Default"
run_name='testing'

option="train"  # "train", "test" or "evaluate"
model_type="note_sequence"  # "note_sequence", "baseline" or "proposed"  | default: "proposed"

## input features
features="pitch onset duration velocity"  # default: "pitch onset duration velocity"
pitch_encoding="midi"  # "midi" or "chroma" | best: "midi"
onset_encoding="shift-onehot"  # "shift-onehot", "shift-raw", "absolute-onehot" or "absolute-raw" | best: "shift-onehot"
duration_encoding="raw"  # "raw" or "onehot" | best: "raw"

## data augmentation
tempo_change_prob=1.0  # 0.0 to 1.0 | default: 1.0
tempo_change_range="0.8 1.2"  # default: [0.8, 1.2]
pitch_shift_prob=1.0  # 0.0 to 1.0 | default: 1.0
pitch_shift_range="-12 12"  # default: [-12, 12]
extra_note_prob=0.5  # 0.0 to 1.0 | default: 0.5 (extra note and missing note not added at the same time for a single piece)
missing_note_prob=0.5  # 0.0 to 1.0 | default: 0.5

## output data
output_type="regression"  # "regression" or "classification" | default: "regression"

## multiprocessing and data-parallel
workers=8  # default: 8, debug: 0
gpus=4  # default: 4, debug: 1

model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/3/a12bec94c23943a38a7bec342071b4ba/checkpoints/epoch=498-val_f1=0.8943.ckpt"

echo ">>> "$option" the model"

python3 quantmidi/main.py \
    --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
    --workspace $WORKSPACE \
    --experiment_name $experiment_name \
    --run_name $run_name \
    --option $option \
    --model_type $model_type \
    --features $features \
    --pitch_encoding $pitch_encoding \
    --onset_encoding $onset_encoding \
    --duration_encoding $duration_encoding \
    --tempo_change_prob $tempo_change_prob \
    --tempo_change_range $tempo_change_range \
    --pitch_shift_prob $pitch_shift_prob \
    --pitch_shift_range $pitch_shift_range \
    --extra_note_prob $extra_note_prob \
    --missing_note_prob $missing_note_prob \
    --output_type $output_type \
    --workers $workers \
    --gpus $gpus \
    --model_checkpoint $model_checkpoint \
    --verbose

