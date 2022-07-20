import os
from collections import defaultdict
import copy

import pandas as pd
import numpy as np
import torch
import util

from constants import max_speed_ball, max_speed_human
import torch_dict

class MetaBasketballData():
    def __init__(self):
        self.game_id2data = {}
        self.team_id2data = {-1: {'id': -1, 'ohid': 0}}
        self.player_id2data = {-1: {'id': -1, 'ohid': 0}}

    def add_game(self, data_game):
        home = data_game['events'][0]['home']
        visitor = data_game['events'][0]['visitor']
        gameid = data_game['gameid'][0]
        self.add_team(home)
        self.add_team(visitor)
        data_game = {'id': gameid, 'ohid': len(self.game_id2data), 'id_home': home['teamid'], 'id_visitor': visitor['teamid']}
        self.game_id2data[gameid] = data_game

    def add_team(self, data_team):
        data_team = copy.copy(data_team)
        data_team['id'] = data_team['teamid']
        data_team['ohid'] = len(self.team_id2data)

        for data_player in data_team['players']:
            self.add_player(data_player, data_team['id'])

        data_team['id_players'] = [player['playerid'] for player in data_team['players']]
        del data_team['players'], data_team['teamid']
        self.team_id2data[data_team['id']] = data_team

    def add_player(self, data_player, id_team):
        data_player = copy.copy(data_player)
        data_player['id'] = data_player['playerid']
        data_player['ohid'] = len(self.player_id2data)
        data_player['id_team'] = id_team
        del data_player['playerid']

        if data_player['id'] in self.player_id2data and data_player['id_team'] != self.player_id2data[data_player['id']]['id_team']:
            raise Exception('Duplicate Player on different team!')

        self.player_id2data[data_player['id']] = data_player
        
# def fill_nan(data, shape=None, dtype=np.float32):
#     ans = np.full(shape, fill_value=np.nan, dtype=dtype)
#     for i, di in enumerate(data):
#         ans[i, :len(di), ...] = di
#     return ans

# def get_data(df):
#     data = defaultdict(lambda : [])
    
#     propers = []
#     for event in tqdm(df['events']):
#         moments = event['moments']
#         if len(moments)==0:
#             continue
        
#         # for moment in moments:
#             # quarter, _, t_game, t_shot, _, player_data = moment
        
#         data['quarter'].append(np.array([ai[0] for ai in moments]))
#         data['t_game'].append(np.array([ai[2] for ai in moments]))
#         data['t_shot'].append(np.array([ai[3] for ai in moments]))
        
#         data['id_team'].append(fill_nan([[aii[0] for aii in ai[5]] for ai in moments],
#                                         shape=(len(moments), 11), dtype=np.int64))
#         data['id_player'].append(fill_nan([[aii[1] for aii in ai[5]] for ai in moments],
#                                           shape=(len(moments), 11), dtype=np.int64))
#         data['x'].append(fill_nan([[aii[2:] for aii in ai[5]] for ai in moments],
#                                   shape=(len(moments), 11, 3)))
#     return data

def game_df2events(df, verbose=False, tqdm=None):
    """
    Convert a dataframe of game data into a list of dictionaries containing the event data.
    """
    data = defaultdict(lambda : [])
    
    n_del_events, n_del_moments = 0, 0
    for event in (df['events'] if tqdm is None else tqdm(df['events'], leave=False)):
        moments = event['moments']
        if len(moments)==0:
            continue
            
        n_players = np.array([len(moment[5]) for moment in moments])
        is_valid_event = (n_players==11).all()
        
        if not is_valid_event:
            n_del_events += 1
            n_del_moments += len(n_players)
            continue
        
        # for moment in moments:
            # quarter, _, t_game, t_shot, _, player_data = moment
        id_game = df['gameid'][0]
        data['id_game'].append(np.full((len(moments), ), fill_value=id_game, dtype=np.int64))
        
        data['quarter'].append(np.array([moment[0] for moment in moments]).astype(np.int32))
        data['t_game'].append(np.array([moment[2] for moment in moments]).astype(np.float32))
        data['t_shot'].append(np.array([moment[3] for moment in moments]).astype(np.float32))
        
        data['id_team'].append(np.array([[player_data[0] for player_data in moment[5]] for moment in moments]).astype(np.int64))
        data['id_player'].append(np.array([[player_data[1] for player_data in moment[5]] for moment in moments]).astype(np.int64))
        data['x'].append(np.array([[player_data[2:] for player_data in moment[5]] for moment in moments]).astype(np.float32))
    
    n_events = len(data['x'])
    n_moments = np.sum([len(event) for event in data['x']])
    if verbose:
        n_total_events, n_total_moments = n_events+n_del_events, n_moments+n_del_moments
        print(f'Deleted {n_del_events:04d} ({n_del_events/n_total_events*100.:0.01f}%) events | '+
              f'{n_del_moments:06d} ({n_del_moments/n_total_moments*100.:0.01f}%) moments')
        print(f'   Kept {n_events:04d} ({n_events/n_total_events*100.:0.01f}%) events | '+
              f'{n_moments:06d} ({n_moments/n_total_moments*100.:0.01f}%) moments')
        print()
    return util.dict_list2list_dict(dict(data))

def event2moments(events, mask=True, verbose=False):
    """
    Convert a dictionary containing the event data to a dictionary containing the moment (x, nx, etc.) data.
    """
    
    data = events
    idx_inp_xy = 2
    idx_inp_v = 1
    idx_inp_a = 0

    """
    . . . . x,y, :3  can be input
     . . . vx, vy :2 can be input
      . . ax, ay, :1 can be input, 1 is target
    """
    ds = {}
    idxs = util.sliding_window(4, len(data['x']))
    
    # id_game, id_team, id_player
    # TODO: sort players by pid within each timestep
    id_game_m = torch.from_numpy(data['id_game'][idxs]) # B, 4
    id_team_m = torch.from_numpy(data['id_team'][idxs]) # B, 4, 11
    id_player_m = torch.from_numpy(data['id_player'][idxs]) # B, 4, 11
    ds['id_game'] = id_game_m[..., -1] # B
    ds['id_team'] = id_team_m[..., -1, :] # B, 11
    ds['id_player'] = id_player_m[..., -1, :] # B, 11
    
    # quarter, t_game, t_shot
    quarter_m = torch.from_numpy(data['quarter'][idxs]) # B, 4
    t_game_m = torch.from_numpy(data['t_game'][idxs]) # B, 4
    t_shot_m = torch.from_numpy(data['t_shot'][idxs]) # B, 4
    ds['quarter'] = quarter_m[..., -1] # B
    ds['t_game'] = t_game_m[..., -1] # B
    ds['t_shot'] = t_shot_m[..., -1] # B
    
    dt_game_m = t_game_m.diff(dim=-1) # B, 3
    dt_shot_m = t_shot_m.diff(dim=-1) # B, 3
    
    x_m = torch.from_numpy(data['x'][idxs]) # B, 4, 11, 3
    dt_v, dt_a = dt_game_m[..., None, None], dt_game_m[..., 1:, None, None]
    v_m = x_m.diff(dim=-3)/dt_v # B, 3, 11, 3
    a_m = v_m.diff(dim=-3)/dt_a/dt_a # B, 2, 11, 3
    ds['x']  = x_m[..., idx_inp_xy, :, :] # B, 11, 3
    ds['nx'] = x_m[..., idx_inp_xy+1, :, :] # B, 11, 3
    ds['v']  = v_m[..., idx_inp_v, :, :] # B, 11, 3
    ds['nv'] = v_m[..., idx_inp_v+1, :, :] # B, 11, 3
    ds['a']  = a_m[..., idx_inp_a, :, :] # B, 11, 3
    ds['na'] = a_m[..., idx_inp_a+1, :, :] # B, 11, 3
    
    masks = {}
    # mask nan values
    masks['mask_nan_x'] = x_m.isnan().any(dim=-1).any(dim=-1).any(dim=-1) # B
    masks['mask_nan_id'] = id_player_m.isnan().any(dim=-1).any(dim=-1) # B
    masks['mask_nan_t_shot'] = t_shot_m.isnan().any(dim=-1) # B
    
    # mask id changes
    masks['mask_id_change'] = (id_player_m.diff(dim=-2).abs()>0).any(dim=-1).any(dim=-1) # B
    
    # mask improper dt's
    masks['mask_dt_game_bad'] = ((dt_game_m--0.04).abs()>2e-2).any(dim=-1) # B
    masks['mask_dt_shot_bad'] = ((dt_shot_m--0.04).abs()>2e-2).any(dim=-1) # B
    
    # mask high ball/player speeds
    masks['mask_speed_ball'] = (v_m[..., :, :1, :].norm(dim=-1)>max_speed_ball).any(dim=-1).any(dim=-1) # B
    masks['mask_speed_human'] = (v_m[..., :, 1:, :].norm(dim=-1)>max_speed_human).any(dim=-1).any(dim=-1) # B
    
    # combined mask
    masks['mask_total'] = torch.stack(list(masks.values()), dim=-1).any(dim=-1)
    
    if mask and verbose:
        for key, value in masks.items():
            print(f"Mask {key:>20} removes {value.sum().item():>10}/{len(ds['x']):>10} frames.")
    if mask:
        ds = torch_dict.index(ds, ~masks['mask_total'])
    return ds

def process_data_dir(dir_input='data_small', dir_output='data_processed', tqdm=None):
    mbd = MetaBasketballData()
    pbar = [dir_input+'/'+file for file in os.listdir(dir_input) if file.endswith('.json')]
    for path in pbar if tqdm is None else tqdm(pbar):
        print(f'Processing Game: {path}')
        df = pd.read_json(path)
        mbd.add_game(df)
        
        torch.save(mbd, f"{dir_output}/mbd")
        
        events = game_df2events(df, verbose=False, tqdm=tqdm)
        gameid = df['gameid'][0]
        
        for i_event, event in enumerate(events):
            moments = event2moments(event, mask=True, verbose=False)
            if moments['x'].shape[0]==0: # this might be zero because mask_t_shot_nan or mask_dt_shot_bad removes everything... sad
                continue
            file = f"{dir_output}/{gameid:010d}_{i_event:05d}.pth"
            torch.save(moments, file)

def data_loader(data_dir='data_processed', batch_size=100, load_factor=10, tqdm=None):
    files = np.array([file for file in os.listdir(data_dir) if file.endswith('.pth')])
    files = files[np.random.permutation(len(files))]
    
    load_size = batch_size*load_factor
    # load files until reaching load size
    load_data = []
    for file in files if tqdm is None else tqdm(files):
        load_data.append(torch.load(data_dir+'/'+file))
        n_moments = np.sum([a['x'].shape[0] for a in load_data])
        if n_moments>load_size: # reached it
            load_data = torch_dict.cat(load_data, dim=0)
            load_data = torch_dict.index(load_data, torch.randperm(n_moments))
            for i_batch, batch in enumerate(torch_dict.split(load_data, batch_size, dim=0)):
                yield batch
            load_data = []