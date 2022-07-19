import copy

import torch

def index(data, idx):
    """
    Indexes into a dictionary of torch Tensors
    """
    data = copy.copy(data)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value[idx]
    return data

def split(data, split_size_or_sections, dim=0):
    """
    Splits dictionary of torch Tensors using torch.split
    """
    data = copy.copy(data)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.split(split_size_or_sections, dim=dim)
            n_sections = len(data[key])
    return [{key: value[i_split] for key, value in data.items()} for i_split in range(n_sections)]

def cat(data, dim=0):
    """
    Concatenates list of dictionaries of torch Tensors using torch.cat
    """
    return {key: torch.cat([di[key] for di in data], dim=0) for key in data[0]}
    