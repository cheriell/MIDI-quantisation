#!/bin/bash


# =============== Download datasets ===============
# path to the folder where the datasets will be stored
ASAP="/import/c4dm-datasets/A2S_transcription/working/datasets/asap-dataset-master"
A_MAPS="/import/c4dm-datasets/A2S_transcription/working/datasets/A-MAPS_1.1"
CPM="/import/c4dm-datasets/A2S_transcription/working/datasets/CPM"
ACPAS="/import/c4dm-datasets/A2S_transcription/working/datasets/ACPAS"

# script to download the dataset

# =============== Feature preparation ================

WORKSPACE="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation"

python3 quantmidi/feature_preparation.py \
    --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
    --feature_folder $WORKSPACE/features \
    --workers 8 \
    --verbose

# =============== Training ===============

python3 quantmidi/train.py \
    --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
    --workspace $WORKSPACE \
    --experiment_name 'input_comparison' \
    --workers 8 \
    --verbose

