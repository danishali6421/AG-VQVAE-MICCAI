**Attention-Guided Vector Quantized Variational Autoencoder for Brain Tumor Segmentation**

An Attention-Guided Vector Quantized Variational Autoencoder (AG-VQ-VAE) — a two-stage network specifically designed for boundary-focused tumor segmentation. Stage 1 comprises a VQ-VAE which learns a compact, discrete latent representation of segmentation masks. In stage 2, a conditional network extracts contextual features from MRI scans and aligns them with discrete mask embeddings to facilitate precise structural correspondence and improved segmentation fidelity.

## 🧠 Model Architecture

![Architecture Diagram](output/architecture.png)

