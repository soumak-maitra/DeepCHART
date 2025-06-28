# DeepCHART
Deep learning for Cosmological Heterogeneity and Astrophysical Reconstruction via Tomography


This project implements a 3D tomographic reconstruction pipeline using Ly\(\alpha\) forest skewers and galaxy positions as input to reconstruct the dark matter (DM) density field. It is built using PyTorch and leverages a variational autoencoder (VAE) with a U-Net-based 3D convolutional neural network.

---

## 📦 Project Structure

```
tomography/
├── config.py              # Configuration for training, model, and paths
├── dataset.py             # Custom PyTorch Dataset class for tau, galaxy, DM
├── model.py               # 3D VAE-UNet model with anisotropic kernels
├── train.py               # Training pipeline
├── requirements.txt       # Python dependencies (optional)
└── README.md              # Project description (you are here)
```

---

## 🧠 Core Features

- **Two-channel input**: sparse Ly\(\alpha\) forest and galaxy CIC fields
- **Output**: log-scaled 3D dark matter density
- **Anisotropic volume support**: reconstructs volumes of shape \(N_\text{section} \times N_\text{section} \times Z_\text{section}\)
- **Realistic noise model**: SNR-based Gaussian noise injection in the flux
- **Modular VAE-UNet**: residual blocks, skip connections, and latent sampling

---

## 📐 Volume Specifications

- Input & output volumes are extracted as:
  ```python
  shape = (N_SECTION, N_SECTION, Z_SECTION)  # typically 96 x 96 x 288
  ```
- Sliced from parent cubes of size `N = 128` (configurable)

---

## 🚀 Getting Started

### 1. Configure your setup
Edit `config.py` to set:
- Data paths
- Network hyperparameters
- Volume shape (`N_SECTION`, `Z_SECTION`)
- SNR distribution and augmentation parameters

### 2. Run training
```bash
python train.py
```

### 3. Outputs
- Model checkpoint: `model_tomography_...pth`
- Training log: `Output_F_galaxy_DM_...txt`
- Loss plot: `training_history.png`

---

## 🔧 Requirements

Minimal requirements :
```
numpy
scipy
torch
matplotlib
```


---

## 📊 Future Enhancements

- Add evaluation/validation metrics on test slices
- Extend for real observational masking
- Use transformer-based latent regularization

---

## 📄 Citation
If you use this code, please consider citing:

> Maitra et al., *in prep*, 2025

---

## 📝 License
Specify your license here (e.g., MIT, GPLv3, etc.).
