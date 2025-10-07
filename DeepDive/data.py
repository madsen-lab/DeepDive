import numpy as np
import torch


indx = lambda a, i: a[i] if a is not None else None


class NumericEncoder:
    def __init__(
        self, handle_unknown="neg_int", unknown_key="Unknown", unknown_val=-999
    ):
        if handle_unknown == "neg_int":
            self.handle_unknown = handle_unknown
            self.unknown_val = unknown_val
            self.unknown_key = unknown_key
        else:
            raise NotImplementedError('Currently only "neg_int" is implemented')

    def fit(self, x):
        self._unique = sorted(set(x))
        self._n_unique = len(self._unique)
        self._mapper = {x: y for x, y in zip(list(self._unique), range(self._n_unique))}
        self._inverse_mapper = {
            y: x for x, y in zip(list(self._unique), range(self._n_unique))
        }
        self._fitted = True

    def transform(self, x):
        self._unknown_cats = list(set(x).difference(self._unique))
        for i in self._unknown_cats:
            self._mapper[i] = self.unknown_val
            if len(self._unknown_cats) <= 1:
                self._inverse_mapper[self.unknown_val] = i
            else:
                self._inverse_mapper[self.unknown_val] = self.unknown_key
        return np.vectorize(self._mapper.get)(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return np.vectorize(self._inverse_mapper.get)(x)

    def to_json(self):
        assert self._fitted, "Encoder needs to be fitted in order to be serialized"
        return {
            "_type": "NumericEncoder",
            "_fitted": self._fitted,
            "_unique": self._unique,
            "_n_unique": self._n_unique,
            "_mapper": self._mapper,
            "_inverse_mapper": self._inverse_mapper,
            "handle_unknown": self.handle_unknown,
            "unknown_val": self.unknown_val,
            "unknown_key": self.unknown_key,
        }

    @classmethod
    def load(
        cls,
        _type,
        _fitted,
        _unique,
        _n_unique,
        _mapper,
        _inverse_mapper,
        handle_unknown,
        unknown_val,
        unknown_key,
    ):
        assert _type == "NumericEncoder"

        encoder = cls()
        encoder._fitted = _fitted
        encoder._unique = _unique
        encoder._n_unique = _n_unique
        encoder._mapper = _mapper
        encoder._inverse_mapper = _inverse_mapper

        encoder.handle_unknown = handle_unknown
        encoder.unknown_val = unknown_val
        encoder.unknown_key = unknown_key
        return encoder


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1), masked=-999):
        self.min = None
        self.max = None
        self.scale_ = None
        self.min_ = None
        self.feature_range = feature_range
        self.masked = masked  # The value to mask

    def fit(self, data):
        # Mask the data by ignoring masked values
        if self.masked is torch.nan:
            mask = ~torch.isnan(data)
        else:
            mask = data != self.masked

        # Only compute min and max on valid (non-masked) data
        valid_data = torch.where(
            mask, data, torch.tensor(float("inf"), dtype=data.dtype)
        )  # Set masked values to +inf for min calculation
        self.min = torch.min(valid_data, dim=0, keepdim=True)[0]

        valid_data = torch.where(
            mask, data, torch.tensor(float("-inf"), dtype=data.dtype)
        )  # Set masked values to -inf for max calculation
        self.max = torch.max(valid_data, dim=0, keepdim=True)[0]

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.max - self.min
        )
        self.min_ = self.feature_range[0] - self.min * self.scale_
        self._fitted = True

    def transform(self, data):
        # Avoid transforming masked values, keep them as is
        if self.masked is torch.nan:
            mask = ~torch.isnan(data)
        else:
            mask = data != self.masked
        scaled_data = data * self.scale_ + self.min_

        # Return scaled data where mask is True, otherwise keep masked values
        return torch.where(
            mask, scaled_data, torch.tensor(self.masked, dtype=data.dtype)
        )

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        # Avoid inverse-transforming masked values
        if self.masked is torch.nan:
            mask = ~torch.isnan(data)
        else:
            mask = data != self.masked
        original_data = (data - self.min_) / self.scale_

        # Return original data where mask is True, otherwise keep masked values
        return torch.where(
            mask, original_data, torch.tensor(self.masked, dtype=data.dtype)
        )

    def to_json(self):
        assert self._fitted, "Scaler needs to be fitted in order to be serialized"
        return {
            "_type": "MinMaxScaler",
            "_fitted": self._fitted,
            "masked": self.masked,
            "min_": self.min_.numpy(),
            "scale_": self.scale_.numpy(),
            "feature_range": self.feature_range,
        }

    @classmethod
    def load(cls, _type, _fitted, masked, min_, scale_, feature_range):
        assert _type == "MinMaxScaler"
        encoder = cls()
        encoder._fitted = _fitted
        encoder.masked = masked
        encoder.min_ = torch.Tensor(min_)
        encoder.scale_ = torch.Tensor(scale_)
        encoder.feature_range = feature_range
        return encoder


class dataset:
    def __init__(
        self,
        data,
        discrete_covariate_names=None,
        continuous_covariate_names=None,
        data_register=dict(),
        device="cpu",
    ):
        self.device = device
        self.matrix = data.X 

        self.discrete_covariate_names = discrete_covariate_names
        self.continuous_covariate_names = continuous_covariate_names
        self.data_size = data.X.shape[0]

        self.covariates = []
        self.covariates_mix = []
        self.continuous_covariates = []

        if isinstance(discrete_covariate_names, list) and discrete_covariate_names:
            if not len(discrete_covariate_names) == len(set(discrete_covariate_names)):
                raise ValueError(
                    f"Duplicate keys were given in: {discrete_covariate_names}"
                )

            for cov in discrete_covariate_names:

                # Covariate encoding
                covariate_names = np.array(data.obs[cov].values)
                encoder_cov = data_register["covar_numeric_encoders"][cov]

                self.covariates.append(
                    torch.Tensor(
                        encoder_cov.transform(covariate_names)  
                    ).long()
                )
                categories = encoder_cov._n_unique  

                indices = torch.randint(0, categories, (self.data_size,))
                self.covariates_mix.append(indices)

        if isinstance(continuous_covariate_names, list) and continuous_covariate_names:
            for cov in continuous_covariate_names:
                self.continuous_covariates.append(
                    data_register["continuous_covariate_scalers"][cov].transform(
                        torch.tensor(
                            np.array(data.obs[cov].values), dtype=torch.float32
                        )
                    )
                )

    def __getitem__(self, i):

        return (
            [self.matrix[i]],
            [*[indx(cov, i) for cov in self.covariates]],
            [*[indx(cov, i) for cov in self.continuous_covariates]],
        )

    def __len__(self):
        return self.data_size


def collate_fn(batch):
    """
    make the accessibility sparse matrix to dense tensor
    """
    x, covariates, continuous_covariates = zip(*batch)
    data_batch = x[0][0].todense()

    return (
        torch.tensor(data_batch, dtype=torch.float32),
        covariates[0],
        continuous_covariates[0],
    )
