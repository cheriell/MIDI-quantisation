#!/bin/bash

WORKSPACE="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation"

# =============== Download datasets ===============
# path to the folder where the datasets will be stored
ASAP="/import/c4dm-datasets/A2S_transcription/working/datasets/asap-dataset-master"
A_MAPS="/import/c4dm-datasets/A2S_transcription/working/datasets/A-MAPS_1.1"
CPM="/import/c4dm-datasets/A2S_transcription/working/datasets/CPM"
ACPAS="/import/c4dm-datasets/A2S_transcription/working/datasets/ACPAS"

# TODO: script to download the dataset

# =============== Feature preparation ================
echo ">>> Preparing the features"

python3 quantmidi/feature_preparation.py \
    --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
    --feature_folder $WORKSPACE/features \
    --workers 8 \
    --verbose

# # =============== Training ===============
# echo ">>> Training the model"

# experiment_name="input_comparison"
# model_type="note_sequence"

# features="onset duration"
# onset_encoding="shift-raw"
# duration_encoding="raw"

# workers="8"
# gpus="4"

# python3 quantmidi/train.py \
#     --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
#     --workspace $WORKSPACE \
#     --experiment_name $experiment_name \
#     --model_type $model_type \
#     --features $features \
#     --onset_encoding $onset_encoding \
#     --duration_encoding $duration_encoding \
#     --workers $workers \
#     --gpus $gpus \
#     --verbose

