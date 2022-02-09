import numpy as np
import torch
from hsi_datamodule import HSIDataModule
from models.types_ import *
from models import BaseVAE
from scipy.spatial import cKDTree
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Sampler(object):
    def __init__(
        self, model: BaseVAE, data: HSIDataModule, normed_radius: float, sigma: float
    ) -> None:
        super(Sampler).__init__()
        self.model = model
        self.data = data
        self.R = normed_radius
        self.num_pixels = data.num_pixels
        self.sigma = sigma
        self.codes, _ = self.model.encode(
            data.hsi_full[:]["Spectrum"], labels=data.hsi_full[:]["Class"]
        )
        self.abundances = data.abundances
        self.gt = self.model.encode(data.gt)[0].detach().numpy()
        self.num_endmembers = np.min(data.hsi_full[:]["Class"].size())
        self.latent_dim = np.min(self.codes.shape)
        self.labels = self.__decode_one_hot(data.hsi_full[:]["Class"])
        self.trees = []
        self.codes = self.codes.detach().numpy()
        self.codes = np.concatenate([self.codes, self.gt])
        self.labels = np.concatenate([self.labels, np.arange(4)])
        self.max_dists = {}
        self.codes_witin_radius = []
        self._endmember_codes = []
        self.__split_codes_by_labels()
        self.__make_trees()
        self.__trim_trees()

    def __decode_one_hot(self, code: Tensor) -> np.array:
        decoded = torch.argmax(code, dim=1).detach().numpy()
        return decoded.astype(float)

    def __split_codes_by_labels(self):
        self._endmember_codes = []
        for i in range(self.num_endmembers):
            indices = np.where(self.labels == i)[0]
            data = self.codes[indices, :]
            self._endmember_codes.append(data)

    def __make_trees(self):
        for i in range(self.num_endmembers):
            self.trees.append(cKDTree(self._endmember_codes[i]))
            num_points = self.trees[i].n
            d, ind = self.trees[i].query(self.gt[i], k=num_points, workers=10)
            max_dist = np.max(d)
            self.max_dists[i] = np.max(d)

    def __trim_trees(self):
        for i in range(self.num_endmembers):
            indices = self.trees[i].query_ball_point(
                self.gt[i], self.R * self.max_dists[i]
            )
            points = self._endmember_codes[i][indices, :]
            self._endmember_codes[i] = points
            self.trees[i] = cKDTree(points)

    def query_tree(self, endmember_nr, num_points):
        n = self.trees[endmember_nr].n
        seeds = self.trees[endmember_nr].data[np.random.randint(0, n, num_points)]
        dd, idxs = self.trees[endmember_nr].query(
            x=seeds, k=np.random.randint(2, 4, 2)[0]
        )
        points = np.squeeze(self.trees[endmember_nr].data[idxs, :])
        mean = np.mean(points, axis=1)
        return np.reshape(mean, (num_points, self.latent_dim))

    def __add_random_noise(self, codes, sigma=0.01):
        """
        Add random noise to the input codes
        
        :param codes: the latent codes we want to add noise to
        :param sigma: The standard deviation of the noise
        :return: The return value is a tensor of size (num_pizels, code_size) containing the latent codes
        """
        noise = np.random.randn(*codes.shape) * sigma
        return torch.FloatTensor(codes + noise)

    def generate_endmembers(self):
        """
        Given a set of codes, generate the corresponding endmembers
        :return: a numpy array of shape (num_endmembers, num_pixels, num_channels).
        """
        endmembers = []
        for i in range(self.num_endmembers):
            codes = self.query_tree(i, self.num_pixels)
            codes = self.__add_random_noise(codes)
            endmembers.append(self.model.decode(codes).detach().numpy())
        endmembers = np.stack(endmembers, axis=1)
        return endmembers

    def generate_synthetic_hsi(self, fname):
        """
        Generate a synthetic hyperspectral image by multiplying the abundances by the endmembers
        :return: The hsi image
        """
        endmembers = self.generate_endmembers()
        gt = self.model.decode(torch.FloatTensor(self.gt)).cpu().detach().numpy()
        hsi = np.zeros((self.num_pixels, endmembers.shape[2]))
        for i in range(self.num_pixels):
            A = self.abundances[i, :].reshape((self.num_endmembers, 1))
            E = endmembers[i, :]
            hsi[i, :] = np.squeeze(np.matmul(E.T, A))
        lines = self.data.hsi_shape[0]
        cols = self.data.hsi_shape[1]
        sio.savemat(
            fname,
            {
                "Y": hsi,
                "S_GT": self.data.abundance_maps,
                "GT": gt,
                "lines": lines,
                "cols": cols,
            },
        )
        return hsi

