import torch
from torch import nn
import util

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, final_dim=None, dropout=0.):
        super().__init__()
        if final_dim is None:
            final_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, final_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class ModelNet(nn.Module):
    def __init__(self, game, n_inputs=13, n_outputs=3, embed_dim=60, n_heads=4, mlp_dim=100, n_layers=5, dropout=0.0, residual=True):
        super().__init__()
        # position either xyz or fourier
        # team either one hot concatenated or fourier added/concatenated
        # playerid either one hot concatenated or fourier added/concatenated
        
        self.n_inputs = n_inputs
        self.embed_team = nn.Embedding(len(game.team2onehot), n_inputs)
        self.embed_player = nn.Embedding(len(game.pid2onehot), n_inputs)
        
        self.lin_in = nn.Linear(n_inputs, embed_dim)
        self.residual = residual
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.attns = nn.ModuleList([nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout) for _ in range(n_layers)])
        
        self.mlps = nn.ModuleList([MLP(embed_dim, mlp_dim, dropout=dropout) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        
        self.lin_out = nn.Linear(embed_dim, n_outputs)
       
    def calc_fourier_features(self, x):
        if self.n_inputs%3!=0:
            raise ValueError(f'n_inputs: {self.n_inputs} is not divisible by 3')
        x = torch.cat([util.fourier_pos(0, 94, x[..., 0], d=self.n_inputs//3),
                       util.fourier_pos(0, 50, x[..., 1], d=self.n_inputs//3),
                       util.fourier_pos(0, 20, x[..., 2], d=self.n_inputs//3)], dim=-1)
        return x
        
        
    def forward(self, x, id_team=None, id_player=None):
        x = self.calc_fourier_features(x)
        
        if id_team is not None:
            x = x + self.embed_team(id_team)
        if id_player is not None:
            x = x + self.embed_player(id_player)
        
        x = self.lin_in(x)
        
        # norm attention res, norm mlp res
        for i_layer, (norm1, attn, norm2, mlp) in enumerate(zip(self.norms1, self.attns, self.norms2, self.mlps)):
            a = norm1(x)
            a, _ = attn(a, a, a)
            x = x+a if self.residual else a
            
            a = norm2(x)
            a = mlp(a)
            x = x+a if self.residual else a
            
        x = self.lin_out(x)
        
        # return x.tanh()*4.
        return x