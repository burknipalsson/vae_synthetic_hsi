import torch
import io
from PIL import Image
from .types_ import *
from hsi_datamodule import HSIDataModule
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torchvision
import numpy as np


def decode_one_hot(code: Tensor) -> np.array:
    decoded = torch.argmax(code, dim=1).numpy()
    return decoded.astype(int)


""" Plot Latent space of VAE with different color for each class
    Returns an image as tensor
"""


def plot_latent_space_simple(model, datamodule: HSIDataModule, name: str, device):
    spectra = datamodule.hsi_full[:]["Spectrum"]
    labels = datamodule.hsi_full[:]["Class"]
    labels = labels.float()
    if type(model).__name__ in ["ConditionalVAE"]:
        spectra = torch.cat([spectra, labels], dim=1)
    classes = decode_one_hot(labels)
    spectra = spectra.to(device=device)
    mu, y = model.encode(spectra)
    y = np.argmax(y, axis=1)
    mu_np = mu.to(device="cpu").numpy()
    plt.figure(figsize=(7.5, 7.5))
    plt.scatter(x=mu_np[:, 0], y=mu_np[:, 1], s=1, c=classes, cmap="brg")

    plt.xlabel(r"$\mu[0]$")
    plt.ylabel(r"$\mu[1]$")
    # plt.axes('equal')
    plt.title(name + ": Latent Space of HSI")
    plt.colorbar()
    buf = io.BytesIO()

    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im, mu


def compare_gt_with_reconstruction(epoch, model, datamodule, device):
    custom_lines = [
        Line2D([0], [0], color="blue", lw=1),
        Line2D([0], [0], color="red", lw=1),
    ]
    gt = datamodule.gt
    gt = gt.to(device=device)
    recon_gt = model.generate(gt, labels=torch.eye(4).to(device))
    gt = gt.to(device="cpu").numpy().T
    recon_gt = recon_gt.to(device="cpu").numpy().T
    plt.figure()
    plt.plot(gt, "r", label="Reference")
    plt.plot(recon_gt, "b", label="Reconstruction")
    plt.xlabel("Bands")
    plt.ylabel("Normalized reflectance")
    plt.legend(custom_lines, ["Generated", "Reference"])
    # plt.title('Epoch {}'.format(epoch)+' Reference Reconstruction')
    plt.title("Reference Reconstruction")
    buf = io.BytesIO()

    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im


def plot_endmember_bundle(gt, model, num_endmembers, num: int):
    custom_lines = [
        Line2D([0], [0], color="blue", lw=1),
        Line2D([0], [0], color="red", lw=1),
    ]

    gt = model.generate(gt).detach()
    fig = plt.figure(figsize=(10, 10))
    names = ["Asphalt", "Grass", "Tree", "Roof", "Metal", "Soil"]
    spectra = model.sample(num, None)
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        for j in range(spectra.shape[0]):
            plt.plot(
                spectra[j, i, :],
                color="cornflowerblue",
                linestyle="dashed",
                linewidth=1.0,
            )
        ax.plot(gt[i, :], "r", linewidth=2.0)
        # ax.set_title(names[i],fontweight="normal", size=24)
        ax.set_title("Endmember " + str(i), fontweight="normal", size=24)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im
