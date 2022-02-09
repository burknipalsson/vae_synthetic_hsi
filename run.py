from hsi_datamodule import HSIDataModule
from sampler import Sampler
import scipy.io as sio
import torch

# run the train_vae.py file to create the model if you don't have it

# load the beta_vae model
model_filename = "model.pt"
model = torch.load(model_filename)

# load the reference dataset into a HSIDataModule object
data_module = HSIDataModule("./data/Urban4.mat", 0.3, 64, False)

# create a Sampler object to sample the endmembers and create the hsi.
# The arguments are: radius (0 to 1), and std of noise to apply to latent codes.
sampler = Sampler(model, data_module, 0.4, 0)

# create the hsi and save it to the file "test.mat". Also returns the hsi array
synthetic_hsi = sampler.generate_synthetic_hsi("test.mat")

