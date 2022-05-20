#!/bin/bash

WORKSPACE="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation"

# =============== Download datasets ===============
# path to the folder where the datasets will be stored
ASAP="/import/c4dm-datasets/A2S_transcription/working/datasets/asap-dataset-master"
A_MAPS="/import/c4dm-datasets/A2S_transcription/working/datasets/A-MAPS_1.1"
CPM="/import/c4dm-datasets/A2S_transcription/working/datasets/CPM"
ACPAS="/import/c4dm-datasets/A2S_transcription/working/datasets/ACPAS"

MV2H_path="/import/c4dm-datasets/A2S_transcription/working/MV2H/bin"

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
run_name='default'

option="evaluate"  # "train", "test" or "evaluate"
model_type="proposed"  # "note_sequence", "baseline" or "proposed"  | default: "proposed"
resume_training=1  # 1 or 0 | default: 0
plot_results=0  # 1 or 0 | default: 0 | for evaluation only

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

# options in note sequence model
downbeats=1  # 1 or 0 | default: 1
tempos=0  # 1 or 0 | default: 0
reverse_link=1  # 1 or 0 | default: 1

## proposed model version
proposed_model_version=1  # 1 or 2 | default: 1

## multiprocessing and data-parallel
workers=4  # default: 8, debug: 0
gpus=4  # default: 4, debug: 1

# baseline model final version
# model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/MIDI-quantisation/checkpoints/baseline-epoch=40-val_f_beat=0.6912.ckpt"
# note sequence model final version
# model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/7/e32c1cdaa41b4733959998d34783deeb/checkpoints/epoch=251-val_f_beat=0.8962.ckpt"
# proposed model final version
# v1
model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/9/5687f6cc9b8b4546a33c8e1fda64c11e/checkpoints/epoch=105-val_f_beat=0.8682.ckpt"
# v2
# model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/9/08f633f558e34afea9ba4f204c8dd9ef/checkpoints/last.ckpt"

echo ">>> Option: "$option

python3 quantmidi/main.py \
    --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
    --workspace $WORKSPACE \
    --MV2H_path $MV2H_path \
    --experiment_name $experiment_name \
    --run_name $run_name \
    --option $option \
    --model_type $model_type \
    --resume_training $resume_training \
    --plot_results $plot_results \
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
    --downbeats $downbeats \
    --tempos $tempos \
    --reverse_link $reverse_link \
    --proposed_model_version $proposed_model_version \
    --workers $workers \
    --gpus $gpus \
    --model_checkpoint $model_checkpoint \
    --verbose

