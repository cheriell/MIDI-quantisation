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
echo ">>> Training/Testing the model"

experiment_name="Default"
run_name="pitch_midi"

option="test"
model_type="note_sequence"

features="pitch onset duration velocity"
pitch_encoding="midi"
onset_encoding="shift-raw"
duration_encoding="raw"

workers="1"
gpus="1"

model_checkpoint="../workspace/MIDI-quantisation/mlruns/1/f968ce6ae5dd494db3aa692a7cac669c/checkpoints/last.ckpt"

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

