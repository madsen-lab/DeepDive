import time
from collections import defaultdict
from typing import Any, Optional


import torch
from anndata import AnnData

from torch import nn
from torch.distributions import Normal, Poisson

from tqdm import tqdm

from .data import *
from .utils import *
from .losses import *
from .layers import *
from .metrics import *
from .utils import *
from .base_model import BaseMixin


class DeepDive(nn.Module, BaseMixin):
    """
    DeepDive: A conditional variational autoencoder with adversarial 
    disentanglement for single-cell epigenomes. 

    This model learns a latent representation of chromatin accessibility (or 
    other single-cell omics data) while disentangling the effects of known and 
    unknown covariates. It supports adversarial training to remove confounding 
    variation, counterfactual prediction, and flexible handling of both discrete 
    and continuous covariates.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells × features, used as input for training.
    discrete_covariate_names : list[str], optional
        Names of discrete covariates to be included in the model.
    continuous_covariate_names : list[str], optional
        Names of continuous covariates to be included in the model.
    unknown_keys : dict[str, Any], optional
        Keys for covariates and associated labels treated as "unknown" factors.
    continous_mask_value : dict[torch.nan, int], default=-999
        Value to use when masking missing continuous covariates.
    covar_centers : dict[str, Any], optional
        Dictionary specifying centering values for covariates. Used to normalize 
        continuous covariates around reference points or setting a label in discrete 
        covariates as the reference point.
    n_latent : int, default=32
        Dimensionality of the latent space.
    unknown_encoder_hidden : list[str], optional
        Hidden layer sizes for the encoder of unknown covariates.
    unknown_decoder_hidden : list[str], optional
        Hidden layer sizes for the decoder reconstructing unknown covariates.
    known_decoder_hidden : list[str], optional
        Hidden layer sizes for the decoder reconstructing known covariates.
    n_hidden_adv : int, default=16
        Number of hidden units in adversarial networks.
    unknown_encoder_dropout : float, default=0.1
        Dropout rate applied to the unknown covariate encoder.
    unknown_decoder_dropout : float, default=0.1
        Dropout rate applied to the unknown covariate decoder.
    known_decoder_dropout : float, default=0.1
        Dropout rate applied to the known covariate decoder.
    dropout_rate_adv : float, default=0.3
        Dropout rate used in adversarial networks.
    reg_sigma : float, default=0.2
        Regularization strength for Gaussian noise in known latent representation.
    patience : int, default=500
        Early stopping patience during training.
    n_epochs_pretrain_ae : int, default=50
        Number of epochs for pretraining the autoencoder before adversarial training.
    adversary_steps : int, default=3
        Number of adversary update steps per autoencoder step.
    penalty_adversary : float, default=50.0
        Weight applied to the adversarial penalty in the loss function.
    reg_adversary : float, default=10.0
        Regularization applied to the adversary during training.
    autoencoder_lr : float, default=1e-4
        Learning rate for the autoencoder.
    autoencoder_wd : float, default=1e-3
        Weight decay for the autoencoder optimizer.
    adversary_lr : float, default=3e-4
        Learning rate for the adversarial networks.
    adversary_wd : float, default=4e-7
        Weight decay for the adversary optimizer.
    class_loss : str, default="ce" ['ce', 'focal']
        Loss function for discrete covariates ("ce" = cross-entropy, 'focal' = focal loss).
    n_decoders : int, default=1
        Number of decoders for reconstruction tasks.
    device : str, default="cuda"
        Device to run the model on ("cuda" or "cpu").
    seed : int, default=0
        Random seed for reproducibility.
    pin_memory : bool, optional
        Whether to use pinned memory in DataLoader for faster GPU transfer.
    **kwargs
        Additional arguments passed to submodules or training routines.

    Notes
    -----
    - DeepDive disentangles biological variation from known and unknown 
      covariates, making it suitable for counterfactual predictions in 
      single-cell data.
    - The adversarial component penalizes leakage of known covariates into the 
      latent representation.
    - Can be applied to tasks such as identifying regulatory changes, 
      harmonizing across donors, or predicting cell states under 
      hypothetical covariate settings.
    """
    def __init__(
        self,
        adata,
        discrete_covariate_names: Optional[list[str]] = None,
        continuous_covariate_names: Optional[list[str]] = None,
        unknown_keys: Optional[dict[str, Any]] = None,
        continous_mask_value: Optional[dict[torch.nan, int]] = -999,
        covar_centers: Optional[dict[str, Any]] = None,
        n_latent: int = 32,
        unknown_encoder_hidden: Optional[list[str]] = None,
        unknown_decoder_hidden: Optional[list[str]] = None,
        known_decoder_hidden: Optional[list[str]] = None,
        n_hidden_adv: int = 16,
        unknown_encoder_dropout: float = 0.1,
        unknown_decoder_dropout: float = 0.1,
        known_decoder_dropout: float = 0.1,
        dropout_rate_adv: float = 0.3,
        reg_sigma: float = 0.2,
        patience: int = 500,
        n_epochs_pretrain_ae: int = 50,
        adversary_steps: int = 3,
        penalty_adversary: float = 50.0,
        reg_adversary: float = 10.0,
        autoencoder_lr: float = 1e-4,
        autoencoder_wd: float = 1e-3,
        adversary_lr: float = 3e-4,
        adversary_wd: float = 4e-7,
        class_loss: str = "ce",
        n_decoders: int = 1,
        device: str = "cuda",
        seed: int = 0,
        pin_memory: Optional[bool] = None, 
        **kwargs,
    ) -> None:
        self.init_params = self._get_init_params(locals())
        super(DeepDive, self).__init__()
        BaseMixin.__init__(self, **self.init_params)

        _ = self.init_params.pop("adata")
        _ = self.init_params.pop("kwargs")

        # set generic attributes
        self._set_generic_attributes()
        self._setup_from_anndata(adata)

        # set models
        self._initialize_models()
        # self._setup_best_states()

        # optimizers and schedulers
        self._initialize_optimizers_and_schedulers()

        # memory and workers
        if pin_memory is None:
            self.pin_memory = True if self.device == "cuda" else False

        self.history = {"epoch": []}
        self.is_trained = False
        self.to(self.device)

    def _set_generic_attributes(self, *args):

        # early-stopping
        self.best_score = 9e10
        self.patience_trials = 0

        # Scheduler parameters
        self.lr_patience = 30
        self.lr_factor = 0.6
        self.lr_threshold = 0.0
        self.lr_min = 0.0

        # kl varmup
        self.n_epochs_kl_warmup = self.n_epochs_pretrain_ae

        # Handle defaults
        self.unknown_encoder_hidden = (
            self.unknown_encoder_hidden
            if self.unknown_encoder_hidden is not None
            else [128, 64]
        )
        self.unknown_decoder_hidden = (
            self.unknown_decoder_hidden
            if self.unknown_decoder_hidden is not None
            else [64, 128]
        )

        self.known_decoder_hidden = (
            self.known_decoder_hidden
            if self.known_decoder_hidden is not None
            else [64, 128]
        )
        self.covar_centers = (
            self.covar_centers if self.covar_centers is not None else {}
        )
        self.unknown_keys = self.unknown_keys if self.unknown_keys is not None else {}
        self.discrete_covariate_names = (
            self.discrete_covariate_names
            if self.discrete_covariate_names is not None
            else []
        )
        self.continuous_covariate_names = (
            self.continuous_covariate_names
            if self.continuous_covariate_names is not None
            else []
        )
        self.covars_to_add = (
            self.discrete_covariate_names + self.continuous_covariate_names
        )

    def _initialize_models(self):

        self.known = known_network(
            self.num_features,
            self.n_latent,
            self.known_decoder_hidden,
            self.known_decoder_dropout,
            self.data_register,
            self.continuous_covariate_names,
            self.covar_centers,
            self.device,
            sigma=self.reg_sigma,
            n_decoders=self.n_decoders,
        )
        self.unknown = unknown_network(
            num_features=self.num_features,
            encoder_hidden=self.unknown_encoder_hidden,
            decoder_hidden=self.unknown_decoder_hidden,
            n_latent=self.n_latent,
            decoder_dropout=self.unknown_decoder_dropout,
            encoder_dropout=self.unknown_encoder_dropout,
            n_decoders=self.n_decoders,
        )
        self.prior = StandardPrior()
        self.px_scale_activation = nn.Softmax(dim=-1)

        self.px_loc = torch.nn.Parameter(torch.randn(self.num_features))
        self._initialize_disentanglement_regularizers()

    def _initialize_disentanglement_regularizers(self):
        def get_loss_f(x):  
            if x == 'focal':
                return MaskedFocalLoss()
            elif x == 'ce':
                return CrossEntropyWithMasking()
            else: 
                raise NotImplementedError(
                        f"Classifier error has to be one of ['focal', 'ce']"
                    )

        if self.num_discrete_covariates == [0]:
            self.covariates_embeddings = None
            self.adversary_covariates = None
            self.loss_adversary_covariates = None
        else:
            assert 0 not in self.num_discrete_covariates
            if not isinstance(self.covar_centers, dict):
                raise TypeError(f"covar_centers has to be a dict")

            self.unknown_loss_adversary_discrete_covariates = [
                (get_loss_f(self.class_loss))
                for _ in self.data_register["covariate_names_unique"].items()
            ]
            self.unknown_adversary_discrete_covariates = [
                Classifier(
                    n_input=self.n_latent,
                    n_labels=len(unique_covars)
                    - (1 if key in self.unknown_keys else 0),
                    n_hidden=self.n_hidden_adv,
                    dropout_rate=self.dropout_rate_adv,
                    logits=False,
                )
                for key, unique_covars in self.data_register[
                    "covariate_names_unique"
                ].items()
            ]
            self.unknown_loss_adversary_continuous_covariates = [
                MSELossWithMask(mask_value=self.continous_mask_value)
                for _ in self.continuous_covariate_names
            ]
            self.unknown_adversary_continuous_covariates = [
                ContinuousRegressor(
                    n_input=self.n_latent,
                    n_hidden=self.n_hidden_adv,
                    dropout_rate=self.dropout_rate_adv,
                )
                for key in self.continuous_covariate_names
            ]

            self.known_loss_adversary_discrete_covariates = [
                (get_loss_f(self.class_loss))
                for _ in self.data_register["covariate_names_unique"].items()
            ]
            self.known_adversary_discrete_covariates = [
                Classifier(
                    n_input=self.n_latent,
                    n_labels=len(unique_covars)
                    - (1 if key in self.unknown_keys else 0),
                    n_hidden=self.n_hidden_adv,
                    dropout_rate=self.dropout_rate_adv,
                    logits=False,
                )
                for key, unique_covars in self.data_register[
                    "covariate_names_unique"
                ].items()
            ]
            self.known_loss_adversary_continuous_covariates = [
                MSELossWithMask(mask_value=self.continous_mask_value)
                for _ in self.continuous_covariate_names
            ]
            self.known_adversary_continuous_covariates = [
                ContinuousRegressor(
                    n_input=self.n_latent,
                    n_hidden=self.n_hidden_adv,
                    dropout_rate=self.dropout_rate_adv,
                )
                for key in self.continuous_covariate_names
            ]

    def _initialize_optimizers_and_schedulers(self):
        self.optimizer_unknown_list = []
        for i in range(self.n_decoders):
            _parameters = self._get_unknown_parameters(i)
            self.optimizer_unknown_list.append(
                torch.optim.AdamW(
                    _parameters,
                    lr=self.autoencoder_lr,
                    weight_decay=self.autoencoder_wd,
                )
            )

        self.optimizer_known_list = []
        for i in range(self.n_decoders):
            _parameters = self._get_known_parameters(i)
            self.optimizer_known_list.append(
                torch.optim.AdamW(
                    _parameters,
                    lr=self.autoencoder_lr,
                    weight_decay=self.autoencoder_wd,
                )
            )
        has_covariates = self.num_discrete_covariates[0] > 0
        has_continuous_covariates = self.num_continuous_covariates > 0
        self.has_covariates = has_covariates
        self.has_continuous_covariates = has_continuous_covariates

        if has_covariates or has_continuous_covariates:
            _parameters = self._get_unknown_adversary_parameters()
            self.optimizer_unknown_adversaries = torch.optim.AdamW(
                _parameters, lr=self.adversary_lr, weight_decay=self.adversary_wd
            )

            _parameters = self._get_known_adversary_parameters()
            self.optimizer_known_adversaries = torch.optim.AdamW(
                _parameters, lr=self.adversary_lr, weight_decay=self.adversary_wd
            )

        self.scheduler_unknown_list = []
        for i in range(self.n_decoders):
            self.scheduler_unknown_list.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_unknown_list[i],
                    patience=self.lr_patience,
                    factor=self.lr_factor,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                    mode="max",
                    threshold_mode="abs",
                    verbose=True,
                )
            )

        self.scheduler_known_list = []
        for i in range(self.n_decoders):
            self.scheduler_known_list.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_known_list[i],
                    patience=self.lr_patience,
                    factor=self.lr_factor,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                    mode="max",
                    threshold_mode="abs",
                    verbose=True,
                )
            )
        if has_covariates or has_continuous_covariates:
            self.scheduler_unknown_adversary = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_unknown_adversaries,
                    patience=self.lr_patience,
                    factor=self.lr_factor,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                    mode="min",
                    threshold_mode="abs",
                    verbose=True,
                )
            )

            self.scheduler_known_adversary = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_known_adversaries,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                mode="min",
                threshold_mode="abs",
                verbose=True,
            )

    def _get_unknown_parameters(self, use_decoder):
        get_params = lambda model, cond: list(model.parameters()) if cond else []

        _parameters = get_params(self.unknown.encoder, True) + get_params(
            self.unknown.decoder_list[use_decoder], True
        )
        return _parameters

    def _get_unknown_adversary_parameters(self):
        has_covariates = self.num_discrete_covariates[0] > 0
        has_continuous_covariates = self.num_continuous_covariates > 0

        _parameters = []
        if has_covariates:
            for adv in self.unknown_adversary_discrete_covariates:
                _parameters.extend(list(adv.parameters()))
        if has_continuous_covariates:
            for adv in self.unknown_adversary_continuous_covariates:
                _parameters.extend(list(adv.parameters()))
        return _parameters

    def _get_known_parameters(self, use_decoder):
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        has_covariates = self.num_discrete_covariates[0] > 0
        has_continuous_covariates = self.num_continuous_covariates > 0

        _parameters = get_params(self.known.decoder_list[use_decoder], True) + [
            self.px_loc
        ]

        if has_covariates:
            for emb in self.known.discrete_covariates_embeddings.values():
                _parameters.extend(list(emb.parameters()))
        if has_continuous_covariates:
            for emb in self.known.continuous_covariates_embeddings.values():
                _parameters.extend(list(emb.parameters()))
        return _parameters

    def _get_known_adversary_parameters(self):
        has_covariates = self.num_discrete_covariates[0] > 0
        has_continuous_covariates = self.num_continuous_covariates > 0

        _parameters = []
        if has_covariates:
            for adv in self.known_adversary_discrete_covariates:
                _parameters.extend(list(adv.parameters()))
        if has_continuous_covariates:
            for adv in self.known_adversary_continuous_covariates:
                _parameters.extend(list(adv.parameters()))
        return _parameters

    def loss(
        self,
        x,
        qz,
        z,
        q_m,
        q_v,
        px,
        distribution="Poisson",
        calculate_kl=True,
    ):

        if distribution == "Poisson":
            recon_loss = -Poisson(px).log_prob(x).sum(dim=-1).mean()
        else:
            recon_loss = (
                -Normal(px, torch.exp(self.px_loc)).log_prob(x).sum(dim=-1).mean()
            )

        if calculate_kl:
            kl_loss = self.prior.kl(m_q=q_m, v_q=q_v, z=z).mean()
        else:
            kl_loss = torch.zeros(1, device=self.device)

        return recon_loss, kl_loss

    def batch_predict(
        self,
        x,
        covariates,
        continuous_covariates,
        covars_to_add=None,
        return_predictions=False,
        add_unknown=True,
        use_decoder = 0,
        library = None,
        predict_mode = 'selected'
    ):

        x_ = torch.log1p(x)
        batch_size = x.size(0)

        if library is None:
            library = torch.log(x.sum(1)).unsqueeze(1)

        if covars_to_add is None:
            covars_to_add = (
                self.discrete_covariate_names + self.continuous_covariate_names
            )

        if add_unknown:
            qz, q_m, q_v, z, unknown = self.unknown(x_, use_decoder)
            z_unknown = qz.loc
            if len(covars_to_add) > 0:
                z_covs, z_cov, known = self.known(
                    covariates, continuous_covariates, covars_to_add, use_decoder
                )
                summed_scale = known + unknown
            else:
                summed_scale = unknown
                z_cov = torch.zeros(batch_size, self.n_latent, device=self.device)
                known = torch.zeros(batch_size, self.num_features, device=self.device)
        else:
            z_covs, z_cov, known = self.known(
                covariates, continuous_covariates, covars_to_add, use_decoder
            )
            summed_scale = known
            z_unknown = torch.zeros(batch_size, self.n_latent, device=self.device)
            unknown = torch.zeros(batch_size, self.num_features, device=self.device)

        if return_predictions:
            if predict_mode == 'fraction_all':
                all_covars = self.discrete_covariate_names + self.continuous_covariate_names
                _, _, known_max = self.known(covariates, continuous_covariates, all_covars, use_decoder) 
                _, _, _, _, unknown_max = self.unknown(x_, use_decoder)
                max_logits = known_max + unknown_max
                px_scale = self.px_scale_activation(max_logits)
                if add_unknown and len(covars_to_add) > 0:
                    ratio = (torch.exp(known) + torch.exp(unknown)) / (torch.exp(known_max) + torch.exp(unknown_max))
                elif add_unknown and len(covars_to_add) == 0:
                    ratio = (torch.exp(unknown)) / (torch.exp(known_max) + torch.exp(unknown_max))
                else:
                    ratio = (torch.exp(known)) / (torch.exp(known_max) + torch.exp(unknown_max))
                px_scale = px_scale * ratio
                px_rate = torch.exp(library) * px_scale
            elif predict_mode == 'fraction_selected':
                all_covars = self.discrete_covariate_names + self.continuous_covariate_names
                _, _, known_max = self.known(covariates, continuous_covariates, all_covars, use_decoder) 
                _, _, _, _, unknown_max = self.unknown(x_, use_decoder)
                max_logits = known_max + unknown_max
                px_scale = self.px_scale_activation(summed_scale)
                if add_unknown and len(covars_to_add) > 0:
                    ratio = (torch.exp(known) + torch.exp(unknown)) / (torch.exp(known_max) + torch.exp(unknown_max))
                elif add_unknown and len(covars_to_add) == 0:
                    ratio = (torch.exp(unknown)) / (torch.exp(known_max) + torch.exp(unknown_max))
                else:
                    ratio = (torch.exp(known)) / (torch.exp(known_max) + torch.exp(unknown_max))
                px_scale = px_scale * ratio
                px_rate = torch.exp(library) * px_scale
            else:
                px_scale = self.px_scale_activation(summed_scale)
                px_rate = torch.exp(library) * px_scale
                ratio = (torch.exp(known) * 0) + 1
            return {
                "px_scale": px_scale,
                "px_rate": px_rate,
                "z_unknown": z_unknown,
                "z_known": z_cov,
                "ratio": ratio,
            }
        else:
            px_scale = self.px_scale_activation(summed_scale)
            px_rate = torch.exp(library) * px_scale
            covar_concat = torch.concat([x.view(-1, 1) for x in covariates], axis=1)
            if add_unknown:
                return (z, qz, q_m, q_v, px_rate, covar_concat)
            else:
                return (None, None, None, None, px_rate, covar_concat)

    def update(
        self,
        x,
        covariates,
        continuous_covariates,
        kl_loss_weight=1,
        update_weights=True,
        distribution="Poisson",
        use_decoder=0,
    ):

        add_unknown = self.epoch >= int(self.n_epochs_pretrain_ae / 2)
        (z, qz, q_m, q_v, px, covar_concat) = self.batch_predict(
            x,
            covariates,
            continuous_covariates,
            covars_to_add=None,
            add_unknown=add_unknown,
            use_decoder=use_decoder,
        )

        recon_loss, kl_loss = self.loss(
            x, qz, z, q_m, q_v, px, distribution, add_unknown
        )

        r2 = r2_metric(x, px, covar_concat)

        train_adv = self.iteration % self.adversary_steps == 0
        finish_pretrain = self.epoch > self.n_epochs_pretrain_ae
        pretrain = self.epoch <= self.n_epochs_pretrain_ae

        if add_unknown and not finish_pretrain:
            z = z.detach().requires_grad_(True)

        (
            known_adversary_discrete_covariates_loss,
            known_adversary_discrete_covariate_predictions,
            known_adversary_continuous_covariates_loss,
            known_adversary_continuous_covariate_predictions,
        ) = self._calculate_known_covariate_adversary_loss(
            covariates, continuous_covariates
        )

        if add_unknown:
            (
                unknown_adversary_discrete_covariates_loss,
                unknown_adversary_discrete_covariate_predictions,
                unknown_adversary_continuous_covariates_loss,
                unknown_adversary_continuous_covariate_predictions,
            ) = self._calculate_unknown_covariate_adversary_loss(
                z,
                covariates,
                continuous_covariates,
            )
        else:
            unknown_adversary_discrete_covariates_loss = torch.zeros(
                1, device=self.device
            )
            unknown_adversary_continuous_covariates_loss = torch.zeros(
                1, device=self.device
            )

        if update_weights:
            self._train_scheduler(
                recon_loss,
                kl_loss,
                known_adversary_discrete_covariates_loss,
                known_adversary_continuous_covariates_loss,
                unknown_adversary_discrete_covariates_loss,
                unknown_adversary_continuous_covariates_loss,
                kl_loss_weight,
                finish_pretrain,
                train_adv,
                pretrain,
                use_decoder,
            )

        return self._get_update_stats(
            recon_loss,
            kl_loss,
            known_adversary_discrete_covariates_loss,
            known_adversary_continuous_covariates_loss,
            unknown_adversary_discrete_covariates_loss,
            unknown_adversary_continuous_covariates_loss,
            r2,
            update_weights,
        )

    def _calculate_known_covariate_adversary_loss(
        self, discrete_covariates, continuous_covariates
    ):
        adversary_discrete_covariates_loss = torch.tensor([0.0], device=self.device)
        adversary_discrete_covariate_predictions = []
        if self.num_discrete_covariates[0] > 0:
            for i, adv in enumerate(self.known_adversary_discrete_covariates):
                adv = adv.to(self.device)

                exclude = list(self.data_register["covariate_names_unique"].keys())[i]
                z_covs, _ = self.known.get_z_covs(
                    discrete_covariates,
                    continuous_covariates,
                    covars_to_add=self.covars_to_add,
                    exclude=exclude,
                )
                pred = adv(z_covs)
                adversary_discrete_covariate_predictions.append(pred)
                adversary_discrete_covariates_loss += (
                    self.known_loss_adversary_discrete_covariates[i](
                        pred, discrete_covariates[i]
                    )
                )
            adversary_discrete_covariates_loss = (
                adversary_discrete_covariates_loss / len(self.num_discrete_covariates)
            )

        adversary_continuous_covariates_loss = torch.tensor([0.0], device=self.device)
        adversary_continuous_covariate_predictions = []
        if self.num_continuous_covariates > 0:
            for i, adv in enumerate(self.known_adversary_continuous_covariates):
                adv = adv.to(self.device)

                exclude = self.continuous_covariate_names[i]
                z_covs, _ = self.known.get_z_covs(
                    discrete_covariates,
                    continuous_covariates,
                    covars_to_add=self.covars_to_add,
                    exclude=exclude,
                )

                pred = adv(z_covs)
                adversary_continuous_covariate_predictions.append(pred)
                adversary_continuous_covariates_loss += (
                    self.known_loss_adversary_continuous_covariates[i](
                        pred, continuous_covariates[i]
                    )
                )
            adversary_continuous_covariates_loss = (
                adversary_continuous_covariates_loss / self.num_continuous_covariates
            )

        return (
            adversary_discrete_covariates_loss,
            adversary_discrete_covariate_predictions,
            adversary_continuous_covariates_loss,
            adversary_continuous_covariate_predictions,
        )

    def _calculate_unknown_covariate_adversary_loss(
        self, z, discrete_covariates, continuous_covariates
    ):
        adversary_discrete_covariates_loss = torch.tensor([0.0], device=self.device)
        adversary_discrete_covariate_predictions = []
        if self.num_discrete_covariates[0] > 0:
            for i, adv in enumerate(self.unknown_adversary_discrete_covariates):
                adv = adv.to(self.device)

                pred = adv(z)
                adversary_discrete_covariate_predictions.append(pred)
                adversary_discrete_covariates_loss += (
                    self.unknown_loss_adversary_discrete_covariates[i](
                        pred, discrete_covariates[i]
                    )
                )
            adversary_discrete_covariates_loss = (
                adversary_discrete_covariates_loss / len(self.num_discrete_covariates)
            )

        adversary_continuous_covariates_loss = torch.tensor([0.0], device=self.device)
        adversary_continuous_covariate_predictions = []
        if self.num_continuous_covariates > 0:
            for i, adv in enumerate(self.unknown_adversary_continuous_covariates):
                adv = adv.to(self.device)

                pred = adv(z)
                adversary_continuous_covariate_predictions.append(pred)
                adversary_continuous_covariates_loss += (
                    self.unknown_loss_adversary_continuous_covariates[i](
                        pred, continuous_covariates[i]
                    )
                )
            adversary_continuous_covariates_loss = (
                adversary_continuous_covariates_loss / self.num_continuous_covariates
            )

        return (
            adversary_discrete_covariates_loss,
            adversary_discrete_covariate_predictions,
            adversary_continuous_covariates_loss,
            adversary_continuous_covariate_predictions,
        )

    def _train_scheduler(
        self,
        recon_loss,
        kl_loss,
        known_adversary_discrete_covariates_loss,
        known_adversary_continuous_covariates_loss,
        unknown_adversary_discrete_covariates_loss,
        unknown_adversary_continuous_covariates_loss,
        kl_loss_weight,
        finish_pretrain,
        train_adv,
        pretrain,
        use_decoder,
    ):

        if pretrain:
            if self.epoch < int(self.n_epochs_pretrain_ae / 2):
                self.pretrain_known(
                    recon_loss,
                    kl_loss,
                    known_adversary_discrete_covariates_loss
                    + known_adversary_continuous_covariates_loss,
                    unknown_adversary_discrete_covariates_loss
                    + unknown_adversary_continuous_covariates_loss,
                    use_decoder,
                )
            else:
                self.pretrain_unknown(
                    recon_loss,
                    kl_loss,
                    known_adversary_discrete_covariates_loss
                    + known_adversary_continuous_covariates_loss,
                    unknown_adversary_discrete_covariates_loss
                    + unknown_adversary_continuous_covariates_loss,
                    use_decoder,
                )
        else:
            if self.iteration % 3 == 0:
                self.train_known_adversary(
                    recon_loss,
                    kl_loss,
                    known_adversary_discrete_covariates_loss
                    + known_adversary_continuous_covariates_loss,
                    unknown_adversary_discrete_covariates_loss
                    + unknown_adversary_continuous_covariates_loss,
                )
            else:
                self.finetune_known(
                    recon_loss,
                    kl_loss,
                    known_adversary_discrete_covariates_loss
                    + known_adversary_continuous_covariates_loss,
                    unknown_adversary_discrete_covariates_loss
                    + unknown_adversary_continuous_covariates_loss,
                    use_decoder,
                )

    def pretrain_known(
        self,
        recon_loss,
        kl_loss,
        known_adversary_loss,
        unknown_adversary_loss,
        use_decoder,
    ):
        self.optimizer_known_list[use_decoder].zero_grad()
        total_loss = recon_loss
        total_loss.backward()
        self.optimizer_known_list[use_decoder].step()

        self.optimizer_known_adversaries.zero_grad()
        adv_loss = known_adversary_loss
        adv_loss.backward()
        self.optimizer_known_adversaries.step()

    def pretrain_unknown(
        self,
        recon_loss,
        kl_loss,
        known_adversary_loss,
        unknown_adversary_loss,
        use_decoder,
    ):
        self.optimizer_unknown_list[use_decoder].zero_grad()
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        self.optimizer_unknown_list[use_decoder].step()

        self.optimizer_unknown_adversaries.zero_grad()
        adv_loss = unknown_adversary_loss
        adv_loss.backward()
        self.optimizer_unknown_adversaries.step()

    def finetune_known(
        self,
        recon_loss,
        kl_loss,
        known_adversary_loss,
        unknown_adversary_loss,
        use_decoder,
    ):
        self.optimizer_known_list[use_decoder].zero_grad()
        total_loss = recon_loss - known_adversary_loss
        total_loss.backward()
        self.optimizer_known_list[use_decoder].step()

    def train_known_adversary(
        self, recon_loss, kl_loss, known_adversary_loss, unknown_adversary_loss
    ):
        self.optimizer_known_adversaries.zero_grad()
        adv_loss = known_adversary_loss
        adv_loss.backward()
        self.optimizer_known_adversaries.step()

    def train_model(
        self,
        train_adata: AnnData,
        validation_adata: Optional[AnnData] = None,
        max_epoch: int = 500,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 10,
        verbose: bool = True,
        distribution: str = "Poisson",
        kl_loss_weight: float = 1.0,
    ) -> None:

        discrete_covariate_names = self.discrete_covariate_names
        continuous_covariate_names = self.continuous_covariate_names
        self.max_epoch = max_epoch

        train_loader = self._create_data_loader(
            train_adata,
            discrete_covariate_names,
            continuous_covariate_names,
            batch_size,
            shuffle,
            num_workers,
        )
        if validation_adata is not None:
            validation_loader = self._create_data_loader(
                validation_adata,
                discrete_covariate_names,
                continuous_covariate_names,
                batch_size,
                shuffle,
                num_workers,
            )
            self.use_validation = True
        else:
            self.use_validation = False

        if verbose:
            pbar = tqdm(total=len(train_loader))
        else:
            pbar = None

        if validation_adata is not None:
            self.total_iterations = (
                len(train_loader) + len(validation_loader)
            ) * self.max_epoch
        else:
            self.total_iterations = (len(train_loader)) * self.max_epoch
        self.iteration_total = 0

        self.start_time = time.time()

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.iteration = 0

            if verbose:
                pbar.refresh()
                pbar.reset()

            self.train()
            epoch_training_stats = self._run_epoch(
                train_loader,
                kl_loss_weight,
                verbose,
                distribution,
                update_weights=True,
                pbar=pbar,
            )

            if validation_adata is not None:
                self.eval()
                epoch_validation_stats = self._run_epoch(
                    validation_loader,
                    kl_loss_weight,
                    False,
                    distribution,
                    update_weights=False,
                    pbar=None,
                )

                self._update_history(
                    epoch_training_stats,
                    epoch_validation_stats,
                    epoch,
                    len(train_loader),
                    len(validation_loader),
                    True,
                )
            else:
                self._update_history(
                    epoch_training_stats,
                    None,
                    epoch,
                    len(train_loader),
                    None,
                    False,
                )

        self.history["elapsed_time_min"] = (time.time() - self.start_time) / 60
        self.is_trained = True

    def _run_epoch(
        self,
        data_loader,
        kl_loss_weight,
        verbose,
        distribution,
        update_weights=True,
        pbar=None,
    ):
        prefix = "Train" if update_weights else "Validation"
        epoch_stats = defaultdict(float)

        for data in data_loader:
            (
                x,
                covariates,
                continuous_covariates,
            ) = self.move_inputs_(data)
            use_decoder = np.random.randint(0, self.n_decoders)
            minibatch_stats = self.update(
                x,
                covariates,
                continuous_covariates,
                kl_loss_weight=self._compute_kl_weight(max_kl_weight=kl_loss_weight),
                update_weights=update_weights,
                distribution=distribution,
                use_decoder=use_decoder,
            )
            self.iteration += 1
            self.iteration_total += 1

            for key, val in minibatch_stats.items():
                epoch_stats[key] += val
            if self.iteration % 5 == 0:
                self.update_time()

            if verbose:
                self.update_pbar(pbar, prefix, minibatch_stats)

        return epoch_stats
