import torch
import numpy as np
import inspect
import json

from .data import NumericEncoder, MinMaxScaler


def argmax(tensor, dim=-1, equal_value=-1):
    # Check if all elements along the given dimension are the same
    all_equal = torch.all(tensor == tensor[..., :1], dim=dim)
    argmax_result = torch.argmax(tensor, dim=dim)
    # Replace argmax result with special_value where all elements are the same
    return torch.where(
        all_equal, torch.tensor(equal_value, device=tensor.device), argmax_result
    )


def discrete_map_with_mask(tensor, mask_value=-999, equal_value=-1):
    mask = tensor == mask_value
    # Replace argmax result with special_value where all elements are the same
    return torch.where(mask, torch.tensor(equal_value, device=tensor.device), tensor)

def continuous_map_with_mask(tensor, mask_value=-999, equal_value=0):     
    mask = tensor != mask_value
    return torch.where(
                        mask,
                        tensor,
                        torch.tensor(float(equal_value), dtype=tensor.dtype),
                    ).view(-1, 1)

def _set_dropout(model, drop_rate=0.0):

    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        _set_dropout(child, drop_rate=drop_rate)

def _get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        if isinstance(obj, NumericEncoder):
            return obj.to_json()
        if isinstance(obj, MinMaxScaler):
            return obj.to_json()
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)
    
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]