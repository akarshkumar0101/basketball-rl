import torch
from torch import nn

import util

import constants

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
    def __init__(self, mbd, n_outputs=3, n_embed=60, n_heads=4, mlp_dim=100, n_layers=5, dropout=0.0, residual=True):
        super().__init__()
        # position either xyz or fourier
        # team either one hot concatenated or fourier added/concatenated
        # playerid either one hot concatenated or fourier added/concatenated
        self.mbd = mbd
        self.n_embed = n_embed
        
        self.x2embed = self.calc_fourier_features
        self.v2embed = MLP(3, n_embed, n_embed)
        
        self.embed_team = nn.Embedding(len(mbd.team_id2data), n_embed)
        self.embed_player = nn.Embedding(len(mbd.player_id2data), n_embed)
        
        # self.lin_input_v = nn.Linear(3, )
        
        # self.lin_in = nn.Linear(n_inputs, n_embed)
        self.residual = residual
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_layers)])
        self.attns = nn.ModuleList([nn.MultiheadAttention(n_embed, n_heads, batch_first=True, dropout=dropout) for _ in range(n_layers)])
        
        self.mlps = nn.ModuleList([MLP(n_embed, mlp_dim, dropout=dropout) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_layers)])
        
        self.lin_out = nn.Linear(n_embed, n_outputs)
        
    def calc_fourier_features(self, x):
        if self.n_embed%3!=0:
            raise ValueError(f'n_embed: {self.n_embed} is not divisible by 3')
        x = torch.cat([util.fourier_pos(0, 94, x[..., 0], d=self.n_embed//3),
                       util.fourier_pos(0, 50, x[..., 1], d=self.n_embed//3),
                       util.fourier_pos(0, 20, x[..., 2], d=self.n_embed//3)], dim=-1)
        return x
    
    def id2ohid(self, ids, id2data):
        """
        Converts a tensor of ids into a tensor of ohids by using the mapping id->id2data[id]['ohid']
        """
        ohid = ids.clone()
        for key in id2data:
            ohid[ids==key] = id2data[key]['ohid']
        return ohid
    
    def forward(self, x, v=None, id_team=None, id_player=None):
        x = self.x2embed(x) # fourier features or MLP mapping from xyz to embedding 
        
        if v is not None:
            v = v/constants.max_speed_ball
            x = x + self.v2embed(v) # MLP mapping from xyz to embedding 
        if id_team is not None:
            x = x + self.embed_team(self.id2ohid(id_team, self.mbd.team_id2data))
        if id_player is not None:
            x = x + self.embed_player(self.id2ohid(id_player, self.mbd.player_id2data))
        
        # x = self.lin_in(x)
        
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