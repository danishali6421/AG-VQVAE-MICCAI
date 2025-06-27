## Attention-Guided Vector Quantized Variational Autoencoder for Brain Tumor Segmentation

An Attention-Guided Vector Quantized Variational Autoencoder (AG-VQ-VAE) — a two-stage network specifically designed for boundary-focused tumor segmentation. Stage 1 comprises a VQ-VAE which learns a compact, discrete latent representation of segmentation masks. In stage 2, a conditional network extracts contextual features from MRI scans and aligns them with discrete mask embeddings to facilitate precise structural correspondence and improved segmentation fidelity.

## 🧠 Model Architecture

![Architecture Diagram](output/architecture.png)

## 📊 Ablation Study: Effect of Attention Scaling (AS) and Soft Masking (SM)

Ablation study on Attention Scaling (AS) and Soft Masking (SM) to evaluate the performance of the single-stage **AG-UNet** and two-stage **AG-VQ-VAE**.

| Model                   | AS ✔ | SM ✔ | HD95 ↓ WT | HD95 ↓ TC | HD95 ↓ ET | Dice ↑ WT | Dice ↑ TC | Dice ↑ ET |
|-------------------------|:----:|:----:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| AG-UNet (Single-stage)  | ✔    | ✔    | 6.15      | 5.39      | 4.37      | **93.19** | 88.26     | **85.09** |
| AG-VQ-VAE (Two-stage)   | ✔    | ✔    | **5.01**  | **4.10**  | **3.74**  | 92.64     | **89.05** | 82.25     |
| AG-VQ-VAE (Two-stage)   | ✔    | ✖    | 5.87      | 4.64      | 4.20      | 91.35     | 88.51     | 80.27     |
| AG-VQ-VAE (Two-stage)   | ✖    | ✔    | 5.57      | 4.99      | 4.43      | 91.74     | 87.88     | 80.41     |


## 🚀 Getting Started

Follow these steps to set up and run the project.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/danishali6421/AG-VQVAE-MICCAI.git
cd AG-VQVAE-MICCAI
```



### 2️⃣ Set Up Environment (Custom Name)

Create the environment with a name of your choice (e.g., `vqvae_env`):

```bash
conda env create -f environment.yml -n vqvae_env
conda activate vqvae_env
```

### 3️⃣ Train the Model

Use the following command inside your `run.sh` to train either **Stage 1** (VQ-VAE) or **Stage 2** (Conditional Network). You will manually **modify the variables** depending on which stage you're training.

```bash
STAGE 1 Training
python main.py \
    --data_path $DATA_PATH \
    --modalities $MODALITIES \
    --crop_size $CROP_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $VQVAE \
    $VQVAETRAINING \
    #$RESUME
STAGE2 Training
python main.py \
    --data_path $DATA_PATH \
    --modalities $MODALITIES \
    --crop_size $CROP_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $COND \
    $CONDTRAINING \
    #$RESUME
``` ```
```
### 4️⃣ Run Inference

Once all **three VQ-VAE modules** and their corresponding **three conditional segmentation networks** have been trained for each tumor region (e.g., WT, TC, ET), you can perform inference using the trained conditional networks.

### 🛠️ Modify `run.sh` for Inference

In your `run.sh`, configure the variables as follows:

```bash
python main.py \
    --data_path $DATA_PATH \
    --modalities $MODALITIES \
    --crop_size $CROP_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $COND \
```

