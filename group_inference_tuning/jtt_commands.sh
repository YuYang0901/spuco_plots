#!/bin/bash

# Define function to run command in tmux session
run_command() {
    tmux new-session -d -s $1 "$2"
}

# Set the path to your script
script_path="/home/sjoshi/spuco_experiments/group_inference_tuning"

# Run commands for GPU 1
run_command "gpu_1_1" "python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=0.1 --infer_num_epochs=40 --batch_size=64 --gpu 1 && python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=0.01 --infer_num_epochs=40 --batch_size=64 --gpu 1"
run_command "gpu_1_2" "python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=1 --infer_num_epochs=40 --batch_size=64 --gpu 1 && python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=0.01 --infer_num_epochs=40 --batch_size=64 --gpu 1"
run_command "gpu_1_3" "python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=1 --infer_num_epochs=40 --batch_size=64 --gpu 1 && python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=0.01 --infer_num_epochs=40 --batch_size=64 --gpu 1"
run_command "gpu_1_4" "python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=1 --infer_num_epochs=40 --batch_size=64 --gpu 1 && python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=0.1 --infer_num_epochs=40 --batch_size=64 --gpu 1"

# Run commands for GPU 2
run_command "gpu_2_1" "python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=0.01 --infer_num_epochs=50 --batch_size=64 --gpu 2 && python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=0.1 --infer_num_epochs=40 --batch_size=64 --gpu 2"
run_command "gpu_2_2" "python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=0.01 --infer_num_epochs=50 --batch_size=64 --gpu 2 && python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=0.1 --infer_num_epochs=60 --batch_size=64 --gpu 2"
run_command "gpu_2_3" "python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=0.01 --infer_num_epochs=50 --batch_size=64 --gpu 2 && python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=0.1 --infer_num_epochs=60 --batch_size=64 --gpu 2"
run_command "gpu_2_4" "python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=0.1 --infer_num_epochs=50 --batch_size=64 --gpu 2 && python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=1 --infer_num_epochs=60 --batch_size=64 --gpu 2"
run_command "gpu_2_5" "python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=0.1 --infer_num_epochs=50 --batch_size=64 --gpu 2 && python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=1 --infer_num_epochs=60 --batch_size=64 --gpu 2 && python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=1 --infer_num_epochs=60 --batch_size=64 --gpu 2"

# Run commands for GPU 3
run_command "gpu_3_1" "python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=0.1 --infer_num_epochs=50 --batch_size=64 --gpu 3 && python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=0.01 --infer_num_epochs=60 --batch_size=64 --gpu 3"
run_command "gpu_3_2" "python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=1 --infer_num_epochs=50 --batch_size=64 --gpu 3 && python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=0.01 --infer_num_epochs=60 --batch_size=64 --gpu 3"
run_command "gpu_3_3" "python waterbirds_jtt.py --pretrained --infer_lr=0.0001 --infer_weight_decay=1 --infer_num_epochs=50 --batch_size=64 --gpu 3 && python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=0.01 --infer_num_epochs=60 --batch_size=64 --gpu 3"
run_command "gpu_3_4" "python waterbirds_jtt.py --pretrained --infer_lr=1e-05 --infer_weight_decay=1 --infer_num_epochs=50 --batch_size=64 --gpu 3 && python waterbirds_jtt.py --pretrained --infer_lr=0.001 --infer_weight_decay=0.1 --infer_num_epochs=60 --batch_size=64 --gpu 3"

# Inform the user that the tmux sessions have been started
echo "Tmux sessions have been started."
