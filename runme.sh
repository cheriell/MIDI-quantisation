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

# experiment_name="Default"
experiment_name="data_augmentation_ablation_study"

option="train"
model_type="note_sequence"  # "note_sequence" or "baseline"  | default: "note_sequence"

# run_name='baseline_model'  # model_type=baseline run_name

## input features
features="pitch onset duration velocity"  # default: "pitch onset duration velocity"
pitch_encoding="midi"  # "midi" or "chroma" | best: "midi"
onset_encoding="shift-onehot"  # "shift-onehot", "shift-raw", "absolute-onehot" or "absolute-raw" | best: "shift-onehot"
duration_encoding="raw"  # "raw" or "onehot" | best: "raw"

# run_name=$pitch_encoding"."$onset_encoding"."$duration_encoding  # input_encoding_experiments run_name
# run_name="no_velocity"  # input_ablation_study run_name

## data augmentation
tempo_change_prob=1.0  # 0.0 to 1.0 | default: 1.0
tempo_change_range="0.8 1.2"  # default: [0.8, 1.2]
pitch_shift_prob=0.0  # 0.0 to 1.0 | default: 1.0
pitch_shift_range="-12 12"  # default: [-12, 12]
extra_note_prob=0.5  # 0.0 to 1.0 | default: 0.5
missing_note_prob=0.5  # 0.0 to 1.0 | default: 0.5

run_name="no_pitch_shift"  # data augmentation ablation run_name

workers=8  # default: 8, debug: 0
gpus=4  # default: 4, debug: 1

model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/2/427166dd051d40379613ef5c13036f11/checkpoints/epoch=120-val_f1=0.8734.ckpt"

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
    --workers $workers \
    --gpus $gpus \
    --model_checkpoint $model_checkpoint \
    # --verbose

