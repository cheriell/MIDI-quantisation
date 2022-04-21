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
#     --workers 16 \
#     --verbose

# =============== Training/Testing ===============

experiment_name="input_encoding_experiments"

option="train"
model_type="note_sequence"

features="pitch onset duration velocity"
pitch_encoding="midi"  # "midi" or "chroma" | best: "midi"
onset_encoding="shift-raw"  # "shift-onehot", "shift-raw", "absolute-onehot" or "absolute-raw" | best: "shift-onehot"
duration_encoding="onehot"  # "raw" or "onehot" | best: "raw"
run_name=$pitch_encoding"."$onset_encoding"."$duration_encoding

workers="8"
gpus="4"

model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/1/bef12d1ffcac4e10a36bddfb945d3aaa/checkpoints/epoch=217-val_f1=0.8862.ckpt"

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
    --workers $workers \
    --gpus $gpus \
    --model_checkpoint $model_checkpoint \
    # --verbose

