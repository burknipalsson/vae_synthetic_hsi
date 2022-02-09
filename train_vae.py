import yaml
import argparse
import numpy as np
import shutil, os
from vae_module import vae_module
from models import *
from hsi_datamodule import HSIDataModule
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
    default="configs/beta_vae.yaml",
)

args = parser.parse_args()
with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["logging_params"]["name"],
    version=config["logging_params"]["version"],
    default_hp_metric=False,
    log_graph=True,
)

# For reproducibility.
if True:
    torch.manual_seed(config["logging_params"]["manual_seed"])
    np.random.seed(config["logging_params"]["manual_seed"])
    cudnn.deterministic = True
    cudnn.benchmark = False

model = BetaVAE(**config["model_params"])
experiment = vae_module(model, config["exp_params"])
data_module = HSIDataModule(**config["datamodule_params"])

runner = Trainer(
    default_root_dir=f"{tb_logger.save_dir}",
    min_epochs=1,
    logger=tb_logger,
    log_every_n_steps=50,
    num_sanity_val_steps=5,
    **config["trainer_params"],
)

print(f"======= Training {config['model_params']['name']} =======")
if os.path.exists("./logs"):
    shutil.rmtree("./logs")

runner.fit(experiment, data_module)
torch.save(model, "./model.pt")

