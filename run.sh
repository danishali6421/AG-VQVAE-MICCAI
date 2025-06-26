#!/bin/bash

# Define arguments
DATA_PATH="../dataset/brats2023/"
MODALITIES="t1n t2w t1c t2f"
CROP_SIZE="128 128 128"
BATCH_SIZE=2
NUM_WORKERS=8
RESUME="--resume"  # Add this flag if you want to resume training; remove it if you are starting from scratch
VQVAETRAINING="--vqvae_training"
LDMTRAINING="--ldmtraining"
CHECKPOINT_DIR="./model/checkpoints"  # Directory for saving and loading checkpoints
VQVAE="--VQVAE"
LDM="--LDM"
COND="--COND"
CONDTRAINING="--cond_training"
LMUNET="--LMUNET"
LMUNETTRAINING="--lmunet_training"

# Run the Python script with the arguments
    python main.py \
    --data_path $DATA_PATH \
    --modalities $MODALITIES \
    --crop_size $CROP_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $COND \
    # $CONDTRAINING \
    # $RESUME \
    # #--checkpoint_dir $CHECKPOINT_DIR
