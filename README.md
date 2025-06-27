## Attention-Guided Vector Quantized Variational Autoencoder for Brain Tumor Segmentation

An Attention-Guided Vector Quantized Variational Autoencoder (AG-VQ-VAE) — a two-stage network specifically designed for boundary-focused tumor segmentation. Stage 1 comprises a VQ-VAE which learns a compact, discrete latent representation of segmentation masks. In stage 2, a conditional network extracts contextual features from MRI scans and aligns them with discrete mask embeddings to facilitate precise structural correspondence and improved segmentation fidelity.

## 🧠 Model Architecture

![Architecture Diagram](output/architecture.png)

## 🚀 Getting Started

Follow these steps to set up and run the project.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/danishali6421/AG-VQVAE-MICCAI.git
cd AG-VQVAE-MICCAI

### 2️⃣ Set Up Environment (Custom Name)

Create the environment with a name of your choice (e.g., `vqvae_env`):

```bash
conda env create -f environment.yml -n vqvae_env
conda activate vqvae_env

