import torch
from torch import nn
import numpy as np

from collections import defaultdict

from constants import max_speed_ball, max_speed_human

class BasketballDataset(torch.utils.data.Dataset):
    def __init__(self, game, dg, input_shot_clock=True, input_p_id=True, tqdm=None):
        self.game = game
        self.dg = dg
        self.input_shot_clock = input_shot_clock
        self.input_p_id = input_p_id
        self.init_ds(tqdm)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [value[idx] for key, value in self.ds]
    
    def init_ds(self, tqdm=None):
        ds = defaultdict(lambda : [])
        ff = 0
        print('going')
        pbar = range(4, len(self.dg))
        if tqdm is not None:
            pbar = tqdm(pbar)
        for i in pbar:
            dgi = self.dg[i-4: i]
            if np.isnan(dgi).any(): #ignore nan values
                ff+=1
                continue
            t = dgi[:, 0]
            dt = np.diff(t)
            # if (np.abs(dt--0.04) > 1e-4).any():
                # continue
            if (np.abs(np.diff(dgi[:, 0])--0.04) > 1e-4).any():
                # print('skipped on game')
                ff+=1
                continue
            if (np.abs(np.diff(dgi[:, 1])--0.04) > 1e-4).any():
                # print('skipped on quarter')
                ff+=1
                continue
            if (np.abs(np.diff(dgi[:, 2])--0.04) > .1).any():
                ff+=1
                continue
            
            """
            . . . . x,y, :3  can be input
             . . . vx, vy :2 can be input
              . . ax, ay, :1 can be input, 1 is target
            """
            idx_inp_xy = 2
            idx_inp_v = 1
            idx_inp_a = 0
            idx_inp_a_target = 1

            shot_clock = dgi[:, 2] 
            bx, by, bz = dgi[:, 5], dgi[:, 6], dgi[:, 7]

            p = dgi[:, 8:].reshape(-1, 10, 4) # t, 10, 4

            id_p = p[:, :, 0]
            id_team = p[:, :, 1]
            if not (id_p.std(axis=0)<1e-3).all(): # players ids change
                continue
            if not (id_team.std(axis=0)<1e-3).all(): # teams ids change
                continue
            id_p, id_team = id_p[idx_inp_xy], id_team[idx_inp_xy]
            id_p, id_team = np.insert(id_p, 0, 0), np.insert(id_team, 0, 0) # add ball id zero
            
            id_team = np.array([self.game.team2onehot[i] for i in id_team.astype(int)])
            id_p = np.array([self.game.pid2onehot[i] for i in id_p.astype(int)])
            
            oh_team = nn.functional.one_hot(torch.from_numpy(id_team), len(self.game.team2onehot))
            # oh_id = nn.functional.one_hot(id_p, len(pid2onehot))
            
            x = p[:, :, 2]
            y = p[:, :, 3]
            z = np.zeros_like(x)
            x = np.concatenate([bx[:, None], x], axis=-1) # add ball as the first "player"
            y = np.concatenate([by[:, None], y], axis=-1)
            z = np.concatenate([bz[:, None], z], axis=-1)
            
            x = np.stack([x,y,z], axis=-1) # t, players, xyz
            v = np.diff(x, axis=0) # t, players, xyz
            a = np.diff(v, axis=0) # t, players, xyz
            
            dt = dt[idx_inp_v]
            nv = v[idx_inp_a_target] # players, xyz
            x = x[idx_inp_xy] # players, xyz
            v = v[idx_inp_v]/dt # players, xyz
            a = a[idx_inp_a]/dt/dt # players, xyz
            
            # xva = np.concatenate([xyz, dxyz, ddxyz], axis=-1) # players, [xyzdxyzddxyz]

            # x_batch = []

            # if self.input_shot_clock:
                # shot_clock = shot_clock[idx_inp_xy]
                # x_batch.append(np.full((len(xva), 1), fill_value=shot_clock))
            # if self.input_p_id:
                # x_batch.append(oh_p_id)
            # x_batch.append(id_team)
            # x_batch.append(oh_team)
            # x_batch.append(xva)
            
            ds['shot_clock'].append(torch.tensor(shot_clock[idx_inp_xy]))
            ds['dt'].append(torch.tensor(dt))
            ds['team'].append(torch.from_numpy(id_team))
            ds['x'].append(torch.from_numpy(x))
            ds['v'].append(torch.from_numpy(v))
            ds['a'].append(torch.from_numpy(a))
            ds['nv'].append(torch.from_numpy(nv))
            # x_batch = torch.from_numpy(np.concatenate(x_batch, axis=-1))
            # ds.append([x_batch, y_batch])
        
        for key, value in ds.items():
            ds[key] = torch.stack(value, dim=0)
            setattr(self, key, ds[key])
        mask1 = (self.v[:, [0], :].norm(dim=-1)<max_speed_ball).all(dim=-1)
        mask2 = (self.v[:,  1:, :].norm(dim=-1)<max_speed_human).all(dim=-1)
        mask = torch.stack([mask1, mask2], dim=0).all(dim=0)
        print(f"removed {(~mask).sum()} for speed issues")
        for key, value in ds.items():
            ds[key] = value[mask]
            setattr(self, key, ds[key])
        self.ds = ds
            
        print('going')
        print('skipped', ff)
        # self.x = torch.stack([batch[0] for batch in ds]).float()
        
        # v = (self.x[:, :, [7,8,9]].norm(dim=-1)/.04)
        # self.x = self.x[(v[:, 1:]<max_speed_human).all(dim=-1)]
        
        # self.y = torch.stack([batch[1] for batch in ds]).float()
        # a1 = torch.logical_and(self.y[:, :, 0]>=0., self.y[:, :, 1]>=0.)
        # a2 = torch.logical_and(self.y[:, :, 0]>=0., self.y[:, :, 1]<0.)
        # a3 = torch.logical_and(self.y[:, :, 0]<0., self.y[:, :, 1]>=0.)
        # a4 = torch.logical_and(self.y[:, :, 0]<0., self.y[:, :, 1]<0.)
        # self.y = torch.stack([a1, a2, a3, a4], dim=-1).float().argmax(dim=-1)