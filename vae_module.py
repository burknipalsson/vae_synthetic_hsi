from torch import optim
from models.types_ import *
from models import BaseVAE
import pytorch_lightning as pl
from models.utils import (
    plot_latent_space_simple,
    compare_gt_with_reconstruction,
)


class vae_module(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(vae_module, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        spectra, labels = batch["Spectrum"], batch["Class"]
        self.curr_device = self.device
        results = self.forward(spectra, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params["KL_weight"],
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log(
            "train loss", value={key: val.item() for key, val in train_loss.items()}
        )

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        spectra, labels = batch["Spectrum"], batch["Class"]
        self.curr_device = self.device
        results = self.forward(spectra, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=self.params["KL_weight"],
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )
        self.log(
            "validation loss", value={key: val.item() for key, val in val_loss.items()}
        )
        return val_loss

    def validation_epoch_end(self, outs):
        tb = self.logger.experiment

        if self.current_epoch % 5 == 0:
            im = compare_gt_with_reconstruction(
                self.current_epoch, self.model, self.trainer.datamodule, self.device
            )
            tb.add_image("Reference Reconstruction", im, global_step=self.current_epoch)

        if self.current_epoch % 10 == 0:
            im, mu = plot_latent_space_simple(
                self.model,
                self.trainer.datamodule,
                type(self.model).__name__,
                self.device,
            )
            tb.add_image(
                "Latent Space of " + type(self.model).__name__,
                im,
                global_step=self.current_epoch,
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        return optimizer

