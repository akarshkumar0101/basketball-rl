
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

pos_hoop = torch.tensor([0, 0.85]).to(device, dtype)

idxs_op = torch.arange(5).to(device)
idxs_dp = torch.arange(5, 10).to(device)
idxs_ball = torch.tensor([10]).to(device)

idxs_oentities = torch.cat((idxs_op, idxs_ball)).to(device)




fps = 10
duration = 5
num_game_steps = fps * duration

radius_player = 0.03
radius_ball = 0.02

radius_three_point_line = 0.83
radius_hoop = 0.04

def random_initial_game_state(batch_size, only_game_state=True):
    pos_op = torch.rand(batch_size, len(idxs_op), 2, device=device)*2 - 1
    pos_dp = torch.rand(batch_size, len(idxs_dp), 2, device=device)*2 - 1
    pos_ball = torch.rand(batch_size, len(idxs_ball), 2, device=device)*2 - 1
    
    init_game_state = torch.cat((pos_op, pos_dp, pos_ball), dim=1)
    
    if only_game_state:
        return init_game_state
    else:
        return init_game_state, pos_op, pos_dp, pos_ball


