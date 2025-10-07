import abc
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from typing import Callable, Optional

# test
import torch
import torch.nn as nn
import numpy as np

from .utils import *


def reparameterize(q_m, q_v, return_dist=True):
    dist = Normal(q_m, q_v.sqrt())
    z = dist.rsample()

    if return_dist:
        return dist, q_m, q_v, z
    return q_m, q_v, z


class known_network(nn.Module):
    def __init__(
        self,
        num_features,
        n_latent,
        decoder_hidden,
        decoder_dropout,
        data_register,
        continuous_covariate_names,
        covar_centers,
        device,
        sigma=0.2,
        n_decoders=1,
    ):
        super(known_network, self).__init__()
        self.mask_value = -999
        self.equal_value = -1
        self.continous_mask_value = -999
        self.continous_equal_value = 0
        self.n_latent = n_latent
        self.device = device
        self.data_register = data_register
        self.continuous_covariate_names = continuous_covariate_names
        self.n_decoders = n_decoders

        self.discrete_covariates_embeddings = nn.ModuleDict(
            {
                key: RegularizedEmbedding(
                    len(unique_covars),
                    self.n_latent,
                    padding_idx=(
                        np.where(
                            self.data_register["covariate_names_unique"][key]
                            == covar_centers[key]
                        )[0].item()
                        if key in covar_centers
                        else None
                    ),
                    sigma=sigma,
                )
                for key, unique_covars in self.data_register[
                    "covariate_names_unique"
                ].items()
            }
        )

        self.continuous_covariates_embeddings = nn.ModuleDict(
            {
                key: RegularizedEmbedding(
                    1, self.n_latent, padding_idx=None, sigma=sigma
                )
                for key in self.continuous_covariate_names
            }
        )

        self.decoder_list = nn.ModuleList()
        for i in range(self.n_decoders):
            self.decoder_list.append(
                Decoder(
                    n_input=n_latent,
                    n_output=num_features,
                    n_hidden=decoder_hidden,
                    dropout_rate=decoder_dropout,
                    use_batch_norm=False,
                    use_layer_norm=True,
                )
            )

    def forward(
        self,
        discrete_covariates,
        continuous_covariates,
        covars_to_add,
        use_decoder="all",
    ):
        z_cov, z_covs = self.get_z_covs(
            discrete_covariates, continuous_covariates, covars_to_add
        )
        if use_decoder == "all":
            px_scale_list = []
            for i in range(self.n_decoders):
                px_scale_list.append(self.decoder_list[i](z_cov))
            px_scale_stack = torch.stack(px_scale_list, dim=0)
            px_scale = torch.mean(px_scale_stack, dim=0)
        else:
            px_scale = self.decoder_list[use_decoder](z_cov)
        return z_covs, z_cov, px_scale

    def get_z_covs(
        self,
        discrete_covariates,
        continuous_covariates,
        covars_to_add=None,
        exclude=None,
    ):
        batch_size = discrete_covariates[0].size(0)
        continuous_zeros = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )

        z_covs_list = []
        z_covs = torch.zeros(
            discrete_covariates[0].size(0), self.n_latent, device=self.device
        )
        for idx, covar in enumerate(self.data_register["covariate_names_unique"]):
            if (covar in covars_to_add) and (covar is not exclude):
                z_covs_list.append(
                    self.discrete_covariates_embeddings[covar](
                        torch.arange(
                            self.data_register["covar_numeric_encoders"][
                                covar
                            ]._n_unique
                            + 1,
                            device=self.device,
                        )[
                            discrete_map_with_mask(
                                discrete_covariates[idx],
                                mask_value=self.mask_value,
                                equal_value=self.equal_value,
                            )
                        ]
                    )
                )
                z_covs += z_covs_list[-1]

        for idx, covar in enumerate(self.continuous_covariate_names):
            if (covar in covars_to_add) and (covar is not exclude):
                z_covs_list.append(
                    self.continuous_covariates_embeddings[covar](continuous_zeros)
                    * continuous_map_with_mask(
                        continuous_covariates[idx],
                        mask_value=self.continous_mask_value,
                        equal_value=self.continous_equal_value,
                    )
                )
                z_covs += z_covs_list[-1]

        return z_covs, z_covs_list

    def _get_concatenate_embedings(
        self,
        covars_to_add=None,
        exclude=None,
    ):

        covar_list = []
        for idx, covar in enumerate(self.data_register["covariate_names_unique"]):
            if (covar in covars_to_add) and (covar is not exclude):

                covar_list.append(
                    self.discrete_covariates_embeddings[covar].weight[:-1]
                )

        for i, covar in enumerate(self.continuous_covariate_names):
            if (covar in covars_to_add) and (covar is not exclude):
                covar_list.append(self.continuous_covariates_embeddings[covar].weight)

        return torch.concat(covar_list)


class unknown_network(nn.Module):
    def __init__(
        self,
        num_features,
        encoder_hidden,
        decoder_hidden,
        n_latent,
        decoder_dropout,
        encoder_dropout,
        n_decoders=1,
    ):
        super(unknown_network, self).__init__()

        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.n_latent = n_latent
        self.num_features = num_features
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.n_decoders = n_decoders

        self.encoder = Encoder(
            n_input=self.num_features,
            n_output=self.n_latent,
            n_hidden=self.encoder_hidden,
            dropout_rate=self.encoder_dropout,
            distribution="normal",
            var_activation=None,
            return_dist=True,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        self.decoder_list = nn.ModuleList()
        for i in range(self.n_decoders):
            self.decoder_list.append(
                Decoder(
                    n_input=self.n_latent,
                    n_output=self.num_features,
                    n_hidden=self.decoder_hidden,
                    dropout_rate=self.decoder_dropout,
                    use_batch_norm=False,
                    use_layer_norm=True,
                )
            )

    def forward(self, x, use_decoder="all"):
        x_ = torch.log1p(x)
        q_m, q_v = self.encoder(x_)
        qz, q_m, q_v, z = reparameterize(q_m, q_v, return_dist=True)
        if not self.training:
            z = qz.loc
        if use_decoder == "all":
            px_scale_list = []
            for i in range(self.n_decoders):
                px_scale_list.append(self.decoder_list[i](z))
            px_scale_stack = torch.stack(px_scale_list, dim=0)
            px_scale = torch.mean(px_scale_stack, dim=0)
        else:
            px_scale = self.decoder_list[use_decoder](z)

        return qz, q_m, q_v, z, px_scale

# adapted from https://github.com/nitzanlab/biolord/blob/main/src/biolord/_module.py
class RegularizedEmbedding(nn.Module):
    """Regularized embedding module."""
    def __init__(
        self,
        n_input: int,
        n_output: int,
        padding_idx=None,
        sigma: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_input,
            embedding_dim=n_output,
            padding_idx=padding_idx,
        )
        self.sigma = sigma

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, x):
        x_ = self.embedding(x)
        if self.training and self.sigma != 0:
            noise = torch.zeros_like(x_)
            noise.normal_(mean=0, std=self.sigma)
            x_ = x_ + noise
        return x_

class Prior(torch.nn.Module, abc.ABC):
    @abstractmethod
    def kl(self, m_q, v_q, z):
        pass

class StandardPrior(Prior):
    def kl(self, m_q, v_q, z=None):
        return kl_divergence(
            Normal(m_q, v_q.sqrt()), Normal(torch.zeros_like(m_q), torch.ones_like(v_q))
        ).sum(dim=1)

# adapted from https://github.com/scverse/scvi-tools
class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list = [],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + n_hidden + [n_out]

        self.fc_layers = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=bias),
                (
                    nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001)
                    if use_batch_norm
                    else nn.Identity()
                ),
                (
                    nn.LayerNorm(out_dim, elementwise_affine=False)
                    if use_layer_norm
                    else nn.Identity()
                ),
                nn.LeakyReLU() if use_relu else nn.Identity(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            )
            self.fc_layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layers in self.fc_layers:
            for layer in layers:
                x = layer(x)
        return x


class Encoder(nn.Module):
    """Encoder network for variational inference."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: list = [],
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.return_dist = return_dist

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden[-1],
            n_hidden=n_hidden[:-1],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        self.mean_encoder = nn.Linear(n_hidden[-1], n_output)
        self.var_encoder = nn.Linear(n_hidden[-1], n_output)

        self.var_activation = torch.exp if var_activation is None else var_activation
        self.z_transformation = (
            nn.Softmax(dim=-1) if distribution == "ln" else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps

        return q_m, q_v


class Decoder(nn.Module):
    """Decoder network for variational autoencoder."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: list = [],
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden[-1],
            n_hidden=n_hidden[:-1],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.px_scale_decoder = nn.Linear(n_hidden[-1], n_output, bias=True)

    def forward(self, z: torch.Tensor):
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)

        return px_scale


class Classifier(nn.Module):
    """Basic fully-connected NN classifier."""

    def __init__(
        self,
        n_input: int,
        n_hidden,
        n_labels: int = 5,
        dropout_rate: float = 0.1,
        logits: bool = False,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_hidden=[n_hidden],
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            ),
            nn.Linear(n_hidden, n_labels),
            nn.Identity() if logits else nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ContinuousRegressor(nn.Module):
    """Basic fully-connected NN regressor for continuous outputs."""

    def __init__(
        self,
        n_input: int,
        n_hidden,
        dropout_rate: float,
    ):
        super().__init__()
        self.regressor = nn.Sequential(
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_hidden=[n_hidden],
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            ),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)
