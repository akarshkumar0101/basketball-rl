import torch
from torch import nn
import numpy as np

from collections import defaultdict

from constants import max_speed_ball, max_speed_human

from util import sliding_window

class BasketballDataset(torch.utils.data.Dataset):
    def __init__(self, game, dg, input_shot_clock=True, input_p_id=True, tqdm=None):
        print('starting')
        self.game = game
        self.dg = torch.from_numpy(dg)
        self.input_shot_clock = input_shot_clock
        self.input_p_id = input_p_id
        print('doing init')
        self.init_ds(tqdm)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [value[idx] for key, value in self.ds]
    
    def init_ds(self, tqdm=None):
        ds = defaultdict(lambda : [])
        print('Creating dataset')
        self.dg = self.dg[:1000]
        idxs = sliding_window(4, len(self.dg))
        
        dg = self.dg[idxs]
        print('Initial dataset size ', dg.shape)
        # dg.shape is all_moments, context moment, features
        # dg.shape is 66000, 4, 48 (game_clock, quarter_clock, shot_clock, quarter, 0, bx, by, br, [p_id, team_id, x, y]*10)
        
        mask_nan = dg.isnan().any(dim=-1).any(dim=-1) # 66000
        print(f'Eliminated {mask_nan.sum().item()} items for nan')
        dg = dg[~mask_nan]
        t = dg[..., 0] # 66000, 4
        dt = t.diff(dim=1) # 66000, 3
        mask_game = ((dt--0.04).abs()>1e-4).any(dim=-1) # 66000
        # do same mask with index 0, 1, 2
        print(f'Eliminated {mask_game.sum().item()} items for bad dt')
        dg = dg[~mask_game]
        
        print('updated dataset size ', dg.shape)
        
        idx_inp_xy = 2
        idx_inp_v = 1
        idx_inp_a = 0
        idx_inp_a_target = 1
        
        """
        . . . . x,y, :3  can be input
         . . . vx, vy :2 can be input
          . . ax, ay, :1 can be input, 1 is target
        """

        shot_clock = dg[..., 2] # 66000, 4
        bx, by, bz = dg[..., 5], dg[..., 6], dg[..., 7] # 66000, 4

        players = dg[..., 8:].reshape(*dg.shape[:-1], 10, 4) # 66000, 4, 10, 4
        id_p = players[..., 0] # 66000, 4, 10
        id_team = players[..., 1] # 66000, 4, 10
        
        # TODO: sort players by pid within each timestep
        
        mask_pid = (id_p.std(axis=1)<1e-3).all(dim=-1) # 66000
        mask_team = (id_team.std(axis=1)<1e-3).all(dim=-1) # 66000
        
        id_p, id_team = id_p[:, idx_inp_xy], id_team[:, idx_inp_xy] # 66000, 10
        id_p = torch.cat([torch.zeros(*id_p.shape[:-1], 1), id_p], dim=-1)
        id_team = torch.cat([torch.zeros(*id_team.shape[:-1], 1), id_team], dim=-1)
        
        id_p = id_p.int().apply_(self.game.pid2onehot.get)
        id_team = id_team.int().apply_(self.game.team2onehot.get)
        
        # add ball and map using tensor.apply_(d.get)
        
        oh_team = nn.functional.one_hot(id_team.long(), len(self.game.team2onehot))
        # oh_id = nn.functional.one_hot(id_p, len(pid2onehot))
        
        x, y = players[..., 2], players[..., 3] # 66000, 4, 10
        z = torch.zeros_like(x) # 66000, 4, 10
        x = torch.cat([bx[..., None], x], dim=-1) # add ball as the first "player"
        y = torch.cat([by[..., None], y], dim=-1)
        z = torch.cat([bz[..., None], z], dim=-1) # 66000, 4, 11
        
        x = torch.stack([x,y,z], axis=-1) # 66000, 4, 11, xyz
        v = torch.diff(x, axis=1) # 66000, 4, 11, xyz
        a = torch.diff(v, axis=1) # 66000, 4, 11, xyz

        dt = dt[idx_inp_v]
        nv = v[idx_inp_a_target] # players, xyz
        x = x[idx_inp_xy] # players, xyz
        v = v[idx_inp_v]/dt # players, xyz
        a = a[idx_inp_a]/dt/dt # players, xyz
        
        ds['shot_clock'] = shot_clock[idx_inp_xy]
        ds['dt'] = dt
        ds['id_team'] = id_team
        ds['x'] = x
        ds['v'] = v
        ds['a'] = a
        ds['nv'] = nv
        
        print(v.shape, dt.shape)
        mask = (ds['v'][:, :1, :].norm(dim=-1)>max_speed_ball).all(dim=-1)
        print(mask.shape)
        
        return
        print(f"removed {(mask).sum()} for ball speed issues")
        print(mask.shape)
        print(ds['v'].shape)
        for key, value in ds.items():
            print(value.shape, mask.shape)
            ds[key] = value[mask]
            
        
        mask = (ds['v'][:, 1:, :].norm(dim=-1)<max_speed_human).all(dim=-1)
        print(f"removed {(mask).sum()} for ball speed issues")
        for key, value in ds.items():
            ds[key] = value[mask]
        # mask = torch.stack([mask1, mask2], dim=0).all(dim=0)
        
        self.ds = ds
            
        # v = (self.x[:, :, [7,8,9]].norm(dim=-1)/.04)
        # self.x = self.x[(v[:, 1:]<max_speed_human).all(dim=-1)]
        
        # self.y = torch.stack([batch[1] for batch in ds]).float()
        # a1 = torch.logical_and(self.y[:, :, 0]>=0., self.y[:, :, 1]>=0.)
        # a2 = torch.logical_and(self.y[:, :, 0]>=0., self.y[:, :, 1]<0.)
        # a3 = torch.logical_and(self.y[:, :, 0]<0., self.y[:, :, 1]>=0.)
        # a4 = torch.logical_and(self.y[:, :, 0]<0., self.y[:, :, 1]<0.)
        # self.y = torch.stack([a1, a2, a3, a4], dim=-1).float().argmax(dim=-1)