import torch
from torch import nn
import numpy as np

from collections import defaultdict

from constants import max_speed_ball, max_speed_human

from util import sliding_window
import copy

def index_data_dict(data, mask):
    data = copy.copy(data)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value[mask]
    return data
def print_data_dict(data):
    for key, value in data.items():
        print(f'{key}: {value.shape}', end=' | ')
    print(); print()
    
def init_ds(game, dg):
    idx_inp_xy = 2
    idx_inp_v = 1
    idx_inp_a = 0

    """
    . . . . x,y, :3  can be input
     . . . vx, vy :2 can be input
      . . ax, ay, :1 can be input, 1 is target
    """
    ds = {}
    print('Creating dataset')

    idxs = sliding_window(4, len(dg))
    ds['dg_m'] = dg[idxs]
    del dg
    print_data_dict(ds)

    # dg.shape is all_moments, context moment, features
    # dg.shape is 66000, 4, 48 (game_clock, quarter_clock, shot_clock, quarter, 0, bx, by, br, [p_id, team_id, x, y]*10)

    mask_nan = ds['dg_m'].isnan().any(dim=-1).any(dim=-1) # 66000
    print(f'Removing {mask_nan.sum().item()} frames for NaN values')
    ds = index_data_dict(ds, ~mask_nan)
    # print_data_dict(ds)

    ds['t_m'] = ds['dg_m'][..., 0] # 66000, 4
    ds['dt_m'] = ds['t_m'].diff(dim=-1) # 66000, 3
    ds['dt'] = ds['dt_m'][..., idx_inp_v]
    mask_game = ((ds['dt_m']--0.04).abs()>1e-4).any(dim=-1) # 66000
    # do same mask with index 0, 1, 2
    print(f'Removing {mask_game.sum().item()} items for bad dt')
    ds = index_data_dict(ds, ~mask_game)
    # print_data_dict(ds)

    shot_clock = ds['dg_m'][..., 2] # 66000, 4
    ds['t_shot'] = shot_clock[..., idx_inp_xy]

    bx, by, bz = ds['dg_m'][..., 5], ds['dg_m'][..., 6], ds['dg_m'][..., 7] # 66000, 4
    players = ds['dg_m'][..., 8:].reshape(*ds['dg_m'].shape[:-1], 10, 4) # 66000, 4, 10, 4
    id_p = players[..., 0] # 66000, 4, 10
    id_team = players[..., 1] # 66000, 4, 10
    # TODO: sort players by pid within each timestep
    mask_pid = (id_p.std(axis=1)<1e-3).all(dim=-1) # 66000
    mask_team = (id_team.std(axis=1)<1e-3).all(dim=-1) # 66000

    id_p, id_team = id_p[:, idx_inp_xy], id_team[:, idx_inp_xy] # 66000, 10
    id_p = torch.cat([torch.zeros(*id_p.shape[:-1], 1), id_p], dim=-1)
    id_team = torch.cat([torch.zeros(*id_team.shape[:-1], 1), id_team], dim=-1)
    id_p = id_p.int().apply_(game.pid2onehot.get)
    id_team = id_team.int().apply_(game.team2onehot.get)
    ds['id_p'], ds['id_team'] = id_p, id_team
    # oh_team = nn.functional.one_hot(id_team.long(), len(game.team2onehot))
    # oh_id = nn.functional.one_hot(id_p.long(), len(pid2onehot))
    # print_data_dict(ds)

    x, y = players[..., 2], players[..., 3] # 66000, 4, 10
    z = torch.zeros_like(x) # 66000, 4, 10
    x = torch.cat([bx[..., None], x], dim=-1) # add ball as the first "player"
    y = torch.cat([by[..., None], y], dim=-1)
    z = torch.cat([bz[..., None], z], dim=-1) # 66000, 4, 11

    ds['x_m'] = torch.stack([x,y,z], axis=-1) # 66000, 4, 11, xyz
    dt = ds['dt'][..., None, None, None]
    ds['v_m'] = torch.diff(ds['x_m'], axis=-3)/dt
    ds['a_m'] = torch.diff(ds['v_m'], axis=-3)/dt/dt

    # print_data_dict(ds)

    ds['x']  = ds['x_m'][..., idx_inp_xy, :, :]
    ds['nx'] = ds['x_m'][..., idx_inp_xy+1, :, :]
    ds['v']  = ds['v_m'][..., idx_inp_v, :, :]
    ds['nv'] = ds['v_m'][..., idx_inp_v+1, :, :]
    ds['a']  = ds['a_m'][..., idx_inp_a, :, :]
    ds['na'] = ds['a_m'][..., idx_inp_a+1, :, :]

    # print_data_dict(ds)

    mask = (ds['v_m'][..., :, :1, :].norm(dim=-1)>max_speed_ball).any(dim=-1).any(dim=-1)
    print(f'Removing {mask.sum().item()} frames for high ball speed')
    ds = index_data_dict(ds, ~mask)
    mask = (ds['v_m'][..., :, 1:, :].norm(dim=-1)>max_speed_human).any(dim=-1).any(dim=-1)
    print(f'Removing {mask.sum().item()} frames for high player speed')
    ds = index_data_dict(ds, ~mask)

    print_data_dict(ds)
    return ds
    
class BasketballDataset(torch.utils.data.Dataset):
    def __init__(self, game, dg, input_shot_clock=True, input_p_id=True, tqdm=None):
        self.game = game
        self.dg = torch.from_numpy(dg)
        # self.input_shot_clock = input_shot_clock
        # self.input_p_id = input_p_id
        # print('doing init')
        self.ds = init_ds(self.game, self.dg)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return index_data_dict(self.ds, idx)