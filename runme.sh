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

experiment_name="Default"

option="train"
model_type="baseline"  # "note_sequence" or "baseline"  | default: "note_sequence"

features="pitch onset duration velocity"  # default: "pitch onset duration velocity"
pitch_encoding="midi"  # "midi" or "chroma" | best: "midi"
onset_encoding="shift-onehot"  # "shift-onehot", "shift-raw", "absolute-onehot" or "absolute-raw" | best: "shift-onehot"
duration_encoding="raw"  # "raw" or "onehot" | best: "raw"

# run_name=$pitch_encoding"."$onset_encoding"."$duration_encoding  # input_encoding_experiments run_name
# run_name="no_duration"  # input_ablation_study run_name
run_name='baseline_model'  # model_type=baseline run_name

workers="0"  # default: "8", debug: "0"
gpus="4"  # default: "4", debug: "1"

model_checkpoint="/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/mlruns/2/3b46be0fb292454b9fc8ee649a2cfee5/checkpoints/epoch=107-val_f1=0.8772.ckpt"

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

