from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from .base import BaseVAE
from .types_ import *
import torch
from torch import nn
import torch.nn.functional as F


class BetaVAE(BaseVAE):
    num_iter = 0
    has_labels = False

    def __init__(
        self,
        num_bands,
        latent_dim,
        hidden_dims: List,
        beta: int = 5,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
        **kwargs
    ) -> None:
        super(BetaVAE, self).__init__()
        self.num_bands = num_bands
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.hidden_dims = hidden_dims
        self.lrelu_beta = 0.02

        # Build encoder
        layers = []
        in_units = self.num_bands
        activation = torch.nn.LeakyReLU(negative_slope=self.lrelu_beta)
        for h_dim in self.hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=in_units, out_features=h_dim),
                    # nn.BatchNorm1d(h_dim),
                    activation,
                )
            )
            in_units = h_dim

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.var = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Build decoder

        self.hidden_dims.reverse()
        in_units = latent_dim
        layers = []
        for h_dim in self.hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=in_units, out_features=h_dim),
                    # nn.BatchNorm1d(h_dim),
                    activation,
                )
            )
            in_units = h_dim
        self.decoder = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(in_units, self.num_bands), torch.nn.LeakyReLU(0.01)
        )

    def encode(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        x = self.encoder(inputs)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]

    def decode(self, inputs: Tensor) -> Tensor:
        x = self.decoder(inputs)
        x = self.output_layer(x)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs, **kwarg) -> Tensor:
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), inputs, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs["M_N"]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]
