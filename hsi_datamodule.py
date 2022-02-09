from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from models.types_ import *
from scipy import io as sio
import numpy as np
import pandas as pd
import torch


class HSIDataset(Dataset):
    def __init__(self, spectra: Tensor, labels: Tensor):
        self.spectra = spectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        spectra = self.spectra[idx]
        sample = {"Spectrum": spectra, "Class": label}
        return sample


class HSIDataModule(LightningDataModule):
    def __init__(
        self, data_path: str, threshold: float, batch_size: int, normalize=False,
    ):
        super().__init__()
        self.path = data_path
        self.threshold = threshold
        self.batch_size = batch_size
        self.rows = None
        self.cols = None
        self.labels = None
        self.spectra = None
        self.hsi_full = None
        self.hsi_train = None
        self.gt = None
        self.num_pixels = 0
        self.abundances = None
        self.abundance_maps = None
        self.hsi_shape = None
        self.normalize = normalize

        self.prepare_data()
        self.setup("fit")

    def load_abundances(self, path):
        data = sio.loadmat(path)
        A = data["A"]
        self.abundances = np.reshape(A, (307, 307, 4))

    def prepare_data(self):
        data = sio.loadmat(self.path)
        spectra = data["Y"].astype(float)
        if self.abundances == None:
            abundances = data["S_GT"].astype(float)
            self.abundance_maps = np.transpose(abundances, [1, 0, 2])
        self.hsi_shape = (
            int(data["lines"][0]),
            int(data["cols"][0]),
            np.min(spectra.shape),
        )
        self.gt = data["GT"].astype(float)
        # self.gt = self.gt/np.max(self.gt)
        self.abundances = np.reshape(
            self.abundance_maps,
            (
                self.abundance_maps.shape[0] * self.abundance_maps.shape[1],
                abundances.shape[-1],
            ),
        )
        spectra = np.transpose(spectra)
        if self.normalize:
            spectra = spectra / np.max(spectra)
        self.num_pixels = np.max(spectra.shape)
        self.rows, self.cols = np.where(self.abundances > self.threshold)
        self.spectra = torch.FloatTensor(spectra[self.rows, :])
        self.labels = F.one_hot(
            torch.LongTensor(self.cols), num_classes=self.cols.max() + 1
        )
        self.gt = torch.FloatTensor(self.gt)
        self.hsi_full = HSIDataset(self.spectra, self.labels)

    def setup(self, stage):
        length = self.hsi_full.__len__()
        train_size = int(0.8 * length)
        validate_size = length - train_size
        self.hsi_train, self.hsi_validate = random_split(
            self.hsi_full, [train_size, validate_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.hsi_train, batch_size=self.batch_size, drop_last=True, num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.hsi_validate,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=1,
        )

    def teardown(self, stage):
        del self.hsi_full
        del self.hsi_train


# DM = HSIDataModule("./data/Urban4.mat", "Urban4", 0.8, 10)
# DM.setup("fit")
# print(list(next(iter(DM.train_dataloader())).values()))
