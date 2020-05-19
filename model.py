

import torch
import torch.nn

import constant


def get_dense_layers(num_features):

    acts = [torch.nn.Tanh() for _ in num_features[1:-1]]
    acts.append(torch.nn.Identity())
    layers = []
    for num_in, num_out, act in zip(num_features, num_features[1:], acts):
        layers.extend((torch.nn.Linear(num_in, num_out), act))
    return torch.nn.Sequential(*layers)


class OffenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = get_dense_layers([11*2+1, 100, 100, 6*2])
        
    def forward(self, X, tss):
        """
        X has shape (-1, 11, 2). tss has shape (-1, 1, 1) representing timestamps.
        """
        X = X.view(-1, 11*2)
        X = torch.cat((X, tss[..., 0]), dim=1)
        X = self.main(X)
        X = X.view(-1, 6, 2)
        X[:, constant.idxs_op, :] = 0.3 * torch.tanh(X[:, constant.idxs_op, :])
        return X


class DefenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = get_dense_layers([11*2+1, 100, 100, 5*2])

    def forward(self, X, tss):
        """
        X has shape (-1, 11, 2). tss has shape (-1, 1, 1) representing timestamps.
        """
        X = X.view(-1, 11*2)
        X = torch.cat((X, tss[..., 0]), dim=1)
        X = self.main(X)
        X = X.view(-1, 5, 2)
        X = 0.3 * torch.tanh(X)
        return X

    
