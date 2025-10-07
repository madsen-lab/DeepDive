import torch
import torch.nn as nn

from collections import OrderedDict
from anndata import AnnData
import json
import os
import time
import numpy as np
from scipy.stats import chi2, spearmanr
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy.stats import norm
from torch.distributions import Poisson
from .data import *
from .utils import *


class BaseMixin:
    def __init__(self, **kwargs):
        self.set_attributes(**kwargs)

    def _get_init_params(self, local_vars):
        """
        Extracts only the parameters defined in the `__init__` signature
        of the child class, excluding *args and **kwargs.
        """
        init = self.__class__.__init__
        sig = inspect.signature(init)
        parameters = sig.parameters.values()
        init_params = {p.name for p in parameters}

        non_kwargs = {
            key: value
            for key, value in local_vars.items()
            if key in init_params and key != "self"
        }
        return non_kwargs

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _calc_metric(self, rec, adata, metric, axis=None):
        adata_copy = adata.copy()
        if type(adata_copy.X) != np.ndarray:
            adata_copy.X = adata_copy.X.todense()

        # Calculate metric
        if metric == "mse":
            val = np.sqrt(np.mean(np.array(rec.X - adata_copy.X) ** 2, axis=axis))
        elif metric == "nll":
            val = (
                -Poisson(torch.tensor(rec.X))
                .log_prob(torch.tensor(adata_copy.X))
                .mean(axis)
                .numpy()
            )
        else:
            rowwise_corr = []
            for row1, row2 in zip(np.array(rec.X), np.array(adata_copy.X)):
                corr, _ = spearmanr(row1, row2)
                rowwise_corr.append(corr)
            val = np.array(rowwise_corr)
            if axis is None:
                val = np.mean(val)
        return val

    def predict_missing(
        self,
        adata,
        covar_importance=None,
        group_by=None,
        vote_for=[],
        majority_vote=False,
        metric="nll",
        mode="ablate",
        covariates=None,
        n_steps=20,
        predict_mode="selected",
        add_unknown=False,
        covars_to_add=None,
        verbose=True,
        continous_window=1.2,
    ):
        # Take a copy
        adata = adata.copy()

        # Covariate importance
        if covar_importance is None:
            if verbose:
                print("Calculating covariate importance.")
            covar_importance = self.covariate_importance()

        # Covariates to predict
        if covariates is None:
            covariates = self.discrete_covariate_names + self.continuous_covariate_names

        # Subset covariance importance by selected covariants
        covar_importance = covar_importance[
            covar_importance["Covariate"].isin(covariates)
        ]

        # Loop across covariates
        for covar in covar_importance["Covariate"]:
            covar_index = np.argwhere(adata.obs.columns == covar)[0][0]
            if covar in self.discrete_covariate_names:
                if covar in list(self.unknown_keys.keys()):
                    mask_key = self.unknown_keys[covar]
                else:
                    mask_key = "Training_mask"
                levels = list(
                    set(self.data_register["covariate_names_unique"][covar])
                    - set([mask_key])
                )
            else:
                mask_key = self.continous_mask_value
                levels = list(
                    np.linspace(
                        self.data_register["continuous_covariate_scalers"][
                            covar
                        ].min.numpy()
                        / continous_window,
                        self.data_register["continuous_covariate_scalers"][
                            covar
                        ].max.numpy()
                        * continous_window,
                        num=n_steps,
                    )[:, 0]
                )

            if np.sum(adata.obs[covar] == mask_key) > 0:
                indices = list(np.argwhere((adata.obs[covar] == mask_key).values)[:, 0])
                if verbose:
                    print(
                        "Predicting "
                        + str(covar)
                        + " for "
                        + str(len(indices))
                        + " cells missing that covariate."
                    )
                adata_cell = adata[indices, :].copy()
                counter = 0
                for level in levels:
                    adata_cell.obs.iloc[:, covar_index] = level
                    rec = self.predict(
                        adata_cell,
                        predict_mode=predict_mode,
                        add_unknown=add_unknown,
                        covars_to_add=covars_to_add,
                    )
                    err_level = self._calc_metric(rec, adata_cell, metric, axis=1)[
                        :, np.newaxis
                    ]
                    if counter == 0:
                        err = err_level
                    else:
                        err = np.concatenate((err, err_level), axis=1)
                    counter += 1

                inferred = np.argmin(err, axis=1)
                mapped = np.array([levels[i] for i in inferred])
                for indice in range(len(indices)):
                    adata.obs.iloc[indices[indice], covar_index] = mapped[indice]

                if majority_vote:
                    if covar in vote_for:
                        if verbose:
                            print(
                                "Majority voting for "
                                + str(covar)
                                + " grouping by "
                                + str(group_by)
                                + "."
                            )
                        adata = self.majority_voting(adata, group_by, [covar])
        return adata

    def majority_voting(self, adata, group_by, vote_for):
        # Take a copy of the data
        adata = adata.copy()

        # Only perform majority voting for covariates in the model
        covariates = set(
            self.discrete_covariate_names + self.continuous_covariate_names
        )
        vote_for = list(set(vote_for).intersection(covariates))

        # Get the levels of the group_by covariate
        levels = list(np.unique(adata.obs[group_by]))

        # Loop through vote_for covariates
        for covar in vote_for:
            covar_index = np.argwhere(adata.obs.columns == covar)[0][0]
            for level in levels:
                indices = list(np.argwhere(adata.obs[group_by] == level)[:, 0])
                if covar in self.discrete_covariate_names:
                    covar_levels, covar_counts = np.unique(
                        adata.obs.iloc[indices, covar_index], return_counts=True
                    )
                    adata.obs.iloc[indices, covar_index] = covar_levels[
                        np.argmax(covar_counts)
                    ]
                else:
                    covar_level = np.median(adata.obs.iloc[indices, covar_index])
                    adata.obs.iloc[indices, covar_index] = covar_level

        return adata

    def covariate_importance(self):
        # Set model into evaluation mode
        self.eval()

        # Get all covariates
        all_covars = self.discrete_covariate_names + self.continuous_covariate_names

        # Loop across covariates in the model
        counter = 0
        for covar in all_covars:
            # Loop across decoders
            importance = []
            for decoder in range(self.n_decoders):
                if covar in self.discrete_covariate_names:
                    discrete = True
                    logits = (
                        self.known.decoder_list[decoder](
                            self.known.discrete_covariates_embeddings[covar].weight
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    nlevels = (
                        len(self.data_register["covariate_names_unique"][covar]) - 1
                    )
                    logits = logits[:nlevels, :]
                else:
                    discrete = False
                    weights = self.known.continuous_covariates_embeddings[covar].weight
                    high_logits = (
                        self.known.decoder_list[decoder](weights).detach().cpu().numpy()
                    )
                    low_logits = (
                        self.known.decoder_list[decoder](torch.zeros_like(weights))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    logits = np.concatenate((high_logits, low_logits))
                importance.append(logits[:2, :].max(axis=0) - logits[:2, :].min(axis=0))

            # Average across multiple decoder or extract a single decoder
            if len(importance) > 1:
                importance = np.mean(np.abs(np.vstack(importance).mean(axis=0)))
            else:
                importance = np.mean(np.abs(importance[0]))

            # Setup results
            res_tmp = pd.DataFrame(
                {"Covariate": covar, "Discrete": discrete, "Importance": importance},
                index=[counter],
            )

            # Combine results
            if counter == 0:
                res = res_tmp
            else:
                res = pd.concat((res, res_tmp))
            counter += 1

        res.sort_values(by="Importance", ascending=False, inplace=True)
        return res

    def move_inputs_(
        self,
        data,
    ):
        """
        Move minibatch tensors to CPU/GPU.
        """
        (
            x,
            covariates,
            continuous_covariates,
        ) = data
        if isinstance(x, list):
            x = x[0]
        if x.device.type != self.device:
            x = x.to(self.device, non_blocking=True)
            if covariates is not None:
                covariates = [
                    cov.to(self.device, non_blocking=True) for cov in covariates
                ]
            if continuous_covariates is not None:
                continuous_covariates = [
                    cov.to(self.device, non_blocking=True)
                    for cov in continuous_covariates
                ]
        return (
            x,
            covariates,
            continuous_covariates,
        )

    def _setup_best_states(self):
        self.best_encoder = self.encoder
        self.best_decoder = self.decoder

        self.best_adversary_covariates = self.adversary_covariates
        self.best_adversary_continuous_covariates = self.adversary_continuous_covariates
        self.best_continuous_covariates_embeddings = (
            self.continuous_covariates_embeddings
        )
        self.best_covariates_embeddings = self.covariates_embeddings
        self.best_prior = self.prior

    def early_stopping(self, score, adv_score):
        self.scheduler_autoencoder.step(score)
        if self.has_covariates or self.has_continuous_covariates:
            self.scheduler_adversary.step(adv_score)

        if score <= self.best_score:
            self.best_score = score
            self.patience_trials = 0
            self.best_epoch = self.epoch

            self.best_encoder = self.encoder
            self.best_decoder = self.decoder

            self.best_adversary_covariates = self.adversary_covariates
            self.best_adversary_continuous_covariates = (
                self.adversary_continuous_covariates
            )
            self.best_continuous_covariates_embeddings = (
                self.continuous_covariates_embeddings
            )
            self.best_covariates_embeddings = self.covariates_embeddings
            self.best_prior = self.prior

        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def _compute_gradient_penalty(self, input, predictions):
        penalty = torch.tensor([0.0], device=self.device)
        for pred in predictions:
            grads = torch.autograd.grad(
                pred.sum(),
                input,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            penalty += grads.pow(2).mean()
        return penalty

    def _get_update_stats(
        self,
        recon_loss,
        kl_loss,
        known_adversary_discrete_covariates_loss,
        known_adversary_continuous_covariates_loss,
        unknown_adversary_discrete_covariates_loss,
        unknown_adversary_continuous_covariates_loss,
        r2,
        update_weights,
    ):
        prefix = "Train" if update_weights else "Validation"
        return {
            f"{prefix}_loss_reconstruction": recon_loss.item(),
            f"{prefix}_loss_kl": kl_loss.item(),
            f"{prefix}_known_adversary_discrete_covariates_loss": known_adversary_discrete_covariates_loss.item(),
            f"{prefix}_known_adversary_continuous_covariates_loss": known_adversary_continuous_covariates_loss.item(),
            f"{prefix}_unknown_adversary_discrete_covariates_loss": unknown_adversary_discrete_covariates_loss.item(),
            f"{prefix}_unknown_adversary_continuous_covariates_loss": unknown_adversary_continuous_covariates_loss.item(),
            f"{prefix}_r2": r2,
        }

    def _create_data_loader(
        self,
        adata,
        discrete_covariate_names,
        continuous_covariate_names,
        batch_size,
        shuffle,
        num_workers,
    ):
        data = dataset(
            adata,
            discrete_covariate_names=discrete_covariate_names,
            continuous_covariate_names=continuous_covariate_names,
            data_register=self.data_register,
        )
        sampler = torch.utils.data.sampler.BatchSampler(
            (
                torch.utils.data.sampler.RandomSampler(
                    data, generator=torch.Generator(device="cpu")
                )
                if shuffle == True
                else torch.utils.data.sampler.SequentialSampler(data)
            ),
            batch_size=batch_size,
            drop_last=False,
        )
        return torch.utils.data.DataLoader(
            data,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_fn,
            sampler=sampler,
        )

    def _update_history(
        self,
        train_stats,
        val_stats,
        epoch,
        train_loader_len,
        val_loader_len,
        use_validation,
    ):
        for key, val in train_stats.items():
            self.history.setdefault(key, []).append(val / train_loader_len)
        if use_validation:
            for key, val in val_stats.items():
                self.history.setdefault(key, []).append(val / val_loader_len)
        self.history["epoch"].append(epoch)

    def _compute_kl_weight(self, max_kl_weight=1.0, min_kl_weight=0.0):

        slope = max_kl_weight - min_kl_weight
        return min(
            max_kl_weight,
            slope * (self.epoch / self.n_epochs_kl_warmup) + min_kl_weight,
        )

    def _setup_from_anndata(self, data):
        self.var_names = data.var_names
        self.num_features = data.n_vars
        self.num_continuous_covariates = len(self.continuous_covariate_names)

        data_register = {
            "covariate_names_unique": OrderedDict(),
            "covar_numeric_encoders": OrderedDict(),
            "covars_dict": OrderedDict(),
            "continuous_covariate_scalers": OrderedDict(),
        }
        # Handle discrete
        if (
            isinstance(self.discrete_covariate_names, list)
            and self.discrete_covariate_names
        ):
            if not len(self.discrete_covariate_names) == len(
                set(self.discrete_covariate_names)
            ):
                raise ValueError(
                    f"Duplicate keys were given in: {self.discrete_covariate_names}"
                )

            for covar in self.discrete_covariate_names:
                if not covar in data.obs.columns:
                    raise KeyError(f"{covar} not found in adata")
                if not data.obs[covar].map(lambda x: isinstance(x, str)).all():
                    raise TypeError(f"Discrete covar: {covar} has to be strings")

            covariate_names = OrderedDict()
            for cov in self.discrete_covariate_names:
                if not isinstance(self.unknown_keys, dict):
                    raise TypeError(f"unknown_keys has to be a dictionary")

                if cov in self.unknown_keys:
                    unknown_key = self.unknown_keys[cov]
                else:
                    unknown_key = None

                covariate_names[cov] = np.array(data.obs[cov].values)
                data_register["covariate_names_unique"][cov] = np.unique(
                    covariate_names[cov]
                )

                if unknown_key is not None:
                    # Move the unknown cov last
                    data_register["covariate_names_unique"][cov] = np.append(
                        data_register["covariate_names_unique"][cov][
                            data_register["covariate_names_unique"][cov] != unknown_key
                        ],
                        [unknown_key],
                    )
                else:
                    # Make an unknown category for masking in training and put it last
                    data_register["covariate_names_unique"][cov] = np.append(
                        data_register["covariate_names_unique"][cov], ["Training_mask"]
                    )
                    unknown_key = "Training_mask"

                names = data_register["covariate_names_unique"][cov]
                names_no_unkown = names[names != unknown_key]

                encoder_cov = NumericEncoder(
                    handle_unknown="neg_int", unknown_key=unknown_key
                )
                encoder_cov.fit(names_no_unkown)
                data_register["covar_numeric_encoders"][cov] = encoder_cov

                data_register["covars_dict"][cov] = OrderedDict(
                    zip(list(names), encoder_cov.transform(names))
                )
        elif not isinstance(self.discrete_covariate_names, list):
            raise TypeError(f"discrete_covariate_names has to be a list")

        # Handle continous
        if (
            isinstance(self.continuous_covariate_names, list)
            and self.continuous_covariate_names
        ):
            if not len(self.continuous_covariate_names) == len(
                set(self.continuous_covariate_names)
            ):
                raise ValueError(
                    f"Duplicate keys were given in: {self.continuous_covariate_names}"
                )

            for covar in self.discrete_covariate_names:
                if not covar in data.obs.columns:
                    raise KeyError(f"{covar} not found in adata")

            for cov in self.continuous_covariate_names:
                data_register["continuous_covariate_scalers"][cov] = MinMaxScaler(
                    masked=self.continous_mask_value
                )
                data_register["continuous_covariate_scalers"][cov].fit(
                    torch.tensor(np.array(data.obs[cov].values), dtype=torch.float32)
                )

        elif not isinstance(self.continuous_covariate_names, list):
            raise TypeError(f"continuous_covariate_names has to be a list")

        self.num_discrete_covariates = [
            len(names) for names in data_register["covariate_names_unique"].values()
        ]
        self.data_register = data_register

    def save(self, dir_path: str) -> None:
        """Save the model

        Args:
            dir_path (str): _description_
            save_anndata (bool, optional): _description_. Defaults to False.
        """

        ## Save params
        params_path = f"{dir_path}/params.json"
        json_object = json.dumps(self.init_params, indent=4, cls=JsonEncoder)
        with open(params_path, "w") as f:
            f.write(json_object)

        ## Save weights
        model_path = f"{dir_path}/model.pt"
        state_dict = self.state_dict()
        torch.save(state_dict, model_path)

        ## Save registers
        register_path = f"{dir_path}/register.json"
        data_reg = self.data_register
        json_object = json.dumps(data_reg, indent=4, cls=JsonEncoder)
        with open(register_path, "w") as f:
            f.write(json_object)

        ##Save history
        history_path = f"{dir_path}/history.json"
        history = self.history
        json_object = json.dumps(history, indent=4)
        with open(history_path, "w") as f:
            f.write(json_object)

        print(f"DeepDIVE model saved at: {dir_path}")

    @classmethod
    def load(cls, adata, dir_path: str):

        ## Load params
        params_path = f"{dir_path}/params.json"
        with open(params_path, "r") as f:
            params = json.load(f)

        ## Initialize model
        model = cls(adata, **params)
        model.is_trained = True

        ## Load history
        history_path = os.path.join(dir_path, "history.json")
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                model.history = json.load(f)
        else:
            print(f"The history file `{history_path}` was not found")

        ## Load weights
        model_path = f"{dir_path}/model.pt"
        model.load_state_dict(torch.load(model_path))

        ## Load registers
        register_path = f"{dir_path}/register.json"
        with open(register_path, "r") as f:
            model.data_register = json.load(f)

            model.data_register["covariate_names_unique"] = OrderedDict(
                (x, np.array(model.data_register["covariate_names_unique"][x]))
                for x in model.data_register["covariate_names_unique"].keys()
            )

            # covar_numeric_encoders
            model.data_register["covar_numeric_encoders"] = OrderedDict(
                (
                    x,
                    NumericEncoder.load(
                        **model.data_register["covar_numeric_encoders"][x]
                    ),
                )
                for x in model.data_register["covar_numeric_encoders"].keys()
            )

            # covar_numeric_encoders
            model.data_register["continuous_covariate_scalers"] = OrderedDict(
                (
                    x,
                    MinMaxScaler.load(
                        **model.data_register["continuous_covariate_scalers"][x]
                    ),
                )
                for x in model.data_register["continuous_covariate_scalers"].keys()
            )

        return model

    def __repr__(self):
        summary_string = ""
        summary_string += f"Encoder size: {self.encoder_hidden}\n"
        summary_string += f"Decoder size: {self.decoder_hidden}\n"
        summary_string += f"Latent size: {self.n_latent}\n"
        summary_string += f"\n"
        summary_string += (
            f"Adversarial networks for the following covariates: {self.covars_to_add}\n"
        )
        summary_string += f"\n"
        summary_string += f"Total parameters: {_get_n_params(self)}"
        summary_string += f"\n"
        summary_string += (
            f"Training status: {'Trained' if self.is_trained else 'Untrained'}"
        )

        print(summary_string)
        return ""

    def predict(
        self,
        adata,
        covars_to_add=None,
        batch_size=256,
        num_workers=10,
        add_unknown=True,
        use_decoder="all",
        library_size="observed",
        predict_mode="selected",
    ):
        self.eval()
        discrete_covariate_names = self.discrete_covariate_names
        continuous_covariate_names = self.continuous_covariate_names
        torch.manual_seed(self.seed)

        data_loader = self._create_data_loader(
            adata,
            discrete_covariate_names,
            continuous_covariate_names,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        recons = []

        # Handle library size
        if library_size == "observed" or library_size is None:
            library_size = None
        elif library_size == "mean" or library_size == "median":
            ls = []
            for data in data_loader:
                (
                    x,
                    covariates,
                    continuous_covariates,
                ) = data
                ls.append(x.sum(1).unsqueeze(1).numpy())
            ls = np.vstack(ls)
            if library_size == "mean":
                library_size = np.mean(ls)
            else:
                library_size = np.median(ls)
            library_size = torch.log(torch.tensor(library_size, device=self.device))
        elif (isinstance(library_size, int) and library_size >= 0) or (
            isinstance(library_size, float) and library_size >= 0
        ):
            library_size = torch.log(torch.tensor(library_size, device=self.device))
        else:
            print(
                "Library size must either None, 'observed', 'mean', 'median' or a positive number. Defaulting to 'observed'"
            )
            library_size = None

        # Handle edge case where both no covariates are added and unknown is not added
        if covars_to_add is not None:
            if not add_unknown and len(covars_to_add) == 0:
                print(
                    "Have to add either covariates or unknown. Defaulting to adding all known covariates."
                )
                covars_to_add = (
                    self.discrete_covariate_names + self.continuous_covariate_names
                )

        for data in data_loader:
            (
                x,
                covariates,
                continuous_covariates,
            ) = self.move_inputs_(data)
            with torch.no_grad():
                pred = self.batch_predict(
                    x,
                    covariates,
                    continuous_covariates,
                    covars_to_add=covars_to_add,
                    return_predictions=True,
                    add_unknown=add_unknown,
                    use_decoder=use_decoder,
                    library=library_size,
                    predict_mode=predict_mode,
                )

            recons.append(pred["px_rate"].cpu().numpy())

        reconstruction = np.concatenate(recons, axis=0)
        reconstruction_ann = AnnData(
            X=reconstruction, obs=adata.obs.copy(), var=adata.var.copy()
        )

        return reconstruction_ann

    def get_attributions(
        self,
        adata,
        covars_to_add=None,
        batch_size=256,
        num_workers=10,
        add_unknown=True,
        use_decoder="all",
    ):
        self.eval()
        discrete_covariate_names = self.discrete_covariate_names
        continuous_covariate_names = self.continuous_covariate_names
        torch.manual_seed(self.seed)

        data_loader = self._create_data_loader(
            adata,
            discrete_covariate_names,
            continuous_covariate_names,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        attributions = []

        # Handle edge case where both no covariates are added and unknown is not added
        if covars_to_add is not None:
            if not add_unknown and len(covars_to_add) == 0:
                print(
                    "Have to add either covariates or unknown. Defaulting to adding all known covariates."
                )
                covars_to_add = (
                    self.discrete_covariate_names + self.continuous_covariate_names
                )

        for data in data_loader:
            (
                x,
                covariates,
                continuous_covariates,
            ) = self.move_inputs_(data)
            with torch.no_grad():
                pred = self.batch_predict(
                    x,
                    covariates,
                    continuous_covariates,
                    covars_to_add=covars_to_add,
                    return_predictions=True,
                    add_unknown=add_unknown,
                    use_decoder=use_decoder,
                    library=None,
                )

            attributions.append(pred["ratio"].cpu().numpy())

        attributions = np.concatenate(attributions, axis=0)
        attributions_ann = AnnData(
            X=attributions, obs=adata.obs.copy(), var=adata.var.copy()
        )

        return attributions_ann

    def get_latent(
        self,
        adata,
        covars_to_add=None,
        batch_size=256,
        num_workers=10,
        add_unknown=True,
        use_decoder="all",
    ):
        self.eval()
        discrete_covariate_names = self.discrete_covariate_names
        continuous_covariate_names = self.continuous_covariate_names
        torch.manual_seed(self.seed)

        data_loader = self._create_data_loader(
            adata,
            discrete_covariate_names,
            continuous_covariate_names,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        zu, zk = [], []

        # Handle edge case where both no covariates are added and unknown is not added
        if covars_to_add is not None:
            if not add_unknown and len(covars_to_add) == 0:
                print(
                    "Have to add either covariates or unknown. Defaulting to adding all known covariates."
                )
                covars_to_add = (
                    self.discrete_covariate_names + self.continuous_covariate_names
                )

        for data in data_loader:
            (
                x,
                covariates,
                continuous_covariates,
            ) = self.move_inputs_(data)
            with torch.no_grad():
                pred = self.batch_predict(
                    x,
                    covariates,
                    continuous_covariates,
                    covars_to_add=covars_to_add,
                    return_predictions=True,
                    add_unknown=add_unknown,
                    use_decoder=use_decoder,
                    library=None,
                )

            zu.append(pred["z_unknown"].cpu().numpy())
            zk.append(pred["z_known"].cpu().numpy())

        zu = np.concatenate(zu, axis=0)
        zk = np.concatenate(zk, axis=0)

        if covars_to_add is None:
            covars_to_add = (
                self.discrete_covariate_names + self.continuous_covariate_names
            )

        if add_unknown and len(covars_to_add) > 0:
            z = np.concatenate((zu, zk), axis=1)
        elif add_unknown and len(covars_to_add) == 0:
            z = zu
        else:
            z = zk

        latent_ann = AnnData(X=z, obs=adata.obs.copy())

        return latent_ann

    def update_time(self):
        t2 = time.time()
        sec_per_iteration = (t2 - self.start_time) / self.iteration_total
        self.est_total_time = time.strftime(
            "%dd:%Hh:%M:m%Ss", time.gmtime(sec_per_iteration * self.total_iterations)
        )
        self.time_passed = time.strftime(
            "%dd:%Hh:%M:m%Ss", time.gmtime(t2 - self.start_time)
        )

    def update_pbar(self, pbar, prefix, minibatch_stats):
        pbar.update(1)
        pbar.set_description(f"Epoch {prefix} [{self.epoch + 1} / {self.max_epoch}]")
        pbar.set_postfix(
            ETA=(
                f"{self.time_passed}|{self.est_total_time}"
                if hasattr(self, "time_passed")
                else "00d:00h:00m:00s|XXd:XXh:XXm:XXs"
            ),
            recon_loss=minibatch_stats[f"{prefix}_loss_reconstruction"],
            kl_loss=minibatch_stats[f"{prefix}_loss_kl"],
        )

    def calculate_counter_factuals(
        self,
        adata,
        covariate,
        baseline,
        target=None,
        mode="leave-out",
        exclude_covariates=None,
        batch_size=256,
        num_workers=10,
        fdr_method="fdr_bh",
        add_unknown=True,
    ):
        ## Checks
        # Mode
        if mode not in ["leave-out", "perturb"]:
            raise ValueError("Mode must be either 'leave-out' or 'perturb'.")

        # Covariate
        all_covariates = self.discrete_covariate_names
        if covariate not in all_covariates:
            raise ValueError(f"Covariate must be one of {all_covariates}.")

        ## Baseline
        covariate_levels = list(self.data_register["covars_dict"][covariate].keys())[
            :-1
        ]
        if baseline not in covariate_levels:
            raise ValueError(f"Baseline must be one of {covariate_levels}.")

        ## Target
        if mode == "perturb":
            perturb_covariate_levels = list(set(covariate_levels) - set([baseline]))
            if target not in perturb_covariate_levels:
                raise ValueError(
                    f"When mode is 'perturb', target must be one of {perturb_covariate_levels}."
                )

        ## Subset adata to only cells in baseline
        adata_subset = adata[adata.obs[covariate] == baseline, :]

        ## Setup covariates to add
        covars_to_add = self.discrete_covariate_names + self.continuous_covariate_names
        if exclude_covariates is not None:
            covars_to_add = list(set(covars_to_add) - set(exclude_covariates))

        ## Reconstruction
        if mode == "perturb":
            # Baseline
            baseline_recon = []
            for decoder in range(self.n_decoders):
                rec, z = self.predict(
                    adata_subset,
                    covars_to_add=covars_to_add,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_decoder=decoder,
                    add_unknown=add_unknown,
                )
                baseline_recon.append(rec.X.mean(axis=0))
            baseline_recon = np.stack(baseline_recon, axis=0)

            # Target
            adata_subset.obs[covariate] = adata_subset.obs[covariate].replace(
                baseline, target
            )
            target_recon = []
            for decoder in range(self.n_decoders):
                rec, z = self.predict(
                    adata_subset,
                    covars_to_add=covars_to_add,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_decoder=decoder,
                    add_unknown=add_unknown,
                )
                target_recon.append(rec.X.mean(axis=0))
            target_recon = np.stack(target_recon, axis=0)
        else:
            # Baseline
            baseline_recon = []
            for decoder in range(self.n_decoders):
                rec, z = self.predict(
                    adata_subset,
                    covars_to_add=covars_to_add,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_decoder=decoder,
                    add_unknown=add_unknown,
                )
                baseline_recon.append(rec.X.mean(axis=0))
            baseline_recon = np.stack(baseline_recon, axis=0)

            # Target
            target_recon = []
            covars_target = list(set(covars_to_add) - set([covariate]))
            for decoder in range(self.n_decoders):
                rec, z = self.predict(
                    adata_subset,
                    covars_to_add=covars_target,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_decoder=decoder,
                    add_unknown=add_unknown,
                )
                target_recon.append(rec.X.mean(axis=0))
            target_recon = np.stack(target_recon, axis=0)

        ## Process
        # Stats per decoder
        diff, Z, P = [], [], []
        for decoder in range(self.n_decoders):
            diff_tmp = baseline_recon[decoder, :] - target_recon[decoder, :]
            Z_tmp = (diff_tmp - diff_tmp.mean()) / diff_tmp.std()
            P_tmp = 2 * (1 - norm.cdf(np.abs(Z_tmp)))
            diff.append(diff_tmp)
            Z.append(Z_tmp)
            P.append(P_tmp)

        # Stacks
        P = np.stack(P, axis=0)
        Z = np.stack(Z, axis=0)
        diff = np.stack(diff, axis=0)

        # Summary
        chi_squared_stat = -2 * np.sum(np.log(P), axis=0)
        df = 2 * self.n_decoders
        P_combined = 1 - chi2.cdf(chi_squared_stat, df)
        Z_combined = np.mean(Z, axis=0)
        diff_combined = np.mean(diff, axis=0)
        reject, pvals_corrected, _, _ = multipletests(
            P_combined, alpha=0.05, method=fdr_method
        )

        # Setup results
        test_res = pd.DataFrame(
            {
                "Feature": adata.var_names,
                "Baseline": baseline_recon.mean(axis=0),
                "Target": target_recon.mean(axis=0),
                "Difference": diff_combined,
                "Zscore": Z_combined,
                "Pvalue": P_combined,
                "FDR": pvals_corrected,
            }
        )

        return test_res

 