# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d, gaussian_filter, zoom
from config import SNR_MIN, SNR_MAX, ALPHA, N, N_SECTION, Z_SECTION, GAPS, L_SMOOTH

def generate_snr(size, SNR_min, SNR_max, alpha):
    """Generate SNR samples using inverse transform sampling."""
    exp = alpha + 1
    u = np.random.uniform(0, 1, size)
    SNR = ((u * (SNR_max**exp - SNR_min**exp)) + SNR_min**exp)**(1/exp)
    return SNR

class CustomDataset(Dataset):
    def __init__(self, file_paths1, file_paths2, file_paths3, transform=None):
        self.file_paths1 = file_paths1
        self.file_paths2 = file_paths2
        self.file_paths3 = file_paths3
        self.transform = transform

    def __len__(self):
        return len(self.file_paths1)

    def __getitem__(self, idx):
        tau = np.load(self.file_paths1[idx]).astype(np.float32)
        dm = np.load(self.file_paths2[idx]).astype(np.float32)
        galaxy = np.load(self.file_paths3[idx]).astype(np.float32)

        flux = np.exp(-tau)
        flux = zoom(flux, (1, 1, N*3/1024), order=1)
        flux = gaussian_filter1d(flux, sigma=3.0/2.355, order=0, mode='wrap', axis=2)

        SNR = np.repeat(generate_snr(N*N, SNR_MIN, SNR_MAX, ALPHA).reshape(N, N, -1), repeats=N*3, axis=2)
        noise = np.random.normal(0, 1/SNR, flux.shape)
        flux = flux + noise * flux
        flux = 1 - flux



        dm = gaussian_filter(dm, sigma=L_SMOOTH, mode='wrap')

        dm = np.log10(dm)


        start_x = np.random.randint(0, N - N_SECTION + 1)
        start_y = np.random.randint(0, N - N_SECTION + 1)
        start_z = np.random.randint(0, 3*N - Z_SECTION + 1)

        flux = flux[start_x:start_x + N_SECTION, start_y:start_y + N_SECTION, start_z:start_z + Z_SECTION]
        galaxy = galaxy[start_x:start_x + N_SECTION, start_y:start_y + N_SECTION, start_z:start_z + Z_SECTION]
        dm = dm[start_x:start_x + N_SECTION, start_y:start_y + N_SECTION, start_z:start_z + Z_SECTION]

        sparse_flux = np.zeros_like(flux)
        x_rand = np.random.choice(N_SECTION, size=int((N_SECTION/GAPS)**2))
        y_rand = np.random.choice(N_SECTION, size=int((N_SECTION/GAPS)**2))
        sparse_flux[x_rand, y_rand, :] = flux[x_rand, y_rand, :]

        inputs = np.stack([sparse_flux, galaxy], axis=0)
        target = dm[np.newaxis, :, :, :]

        inputs = torch.tensor(inputs, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        if self.transform:
            inputs = self.transform(inputs)
            target = self.transform(target)

        return inputs, target

