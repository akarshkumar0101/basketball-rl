
import numpy as np
import torch
from torch import nn

def state_to_nn_input_flat(env, state):
    inp = []
    sc = state['shot_clock']
    scm = env.config['shot_clock_max']
    time_stamp = (sc-scm/2.)/(scm/2.)
    inp.append(np.array(time_stamp)[None])
    print(state.keys())
    posvels = state['posvels']
    print(posvels.shape)
    inp.append(posvels.flatten())
    ball_state = state['ball_state']
    
    print(ball_state)
    dribbling = ball_state['dribbling']
    dribbler = ball_state['dribbler']
    receiver = ball_state['receiver']
    pass_elapsed_time = ball_state['pass_elapsed_time']
    pass_total_time = ball_state['pass_total_time']
    
    print(not ball_state['dribbling'])
    print(int(not ball_state['dribbling']))
    
    v_dribbling = np.array([0, 1] if dribbling else [1, 0]).astype(np.float32)
    v_dribbler = np.array([0, 1] if dribbling else [1, 0]).astype(np.float32)
    
    inp = np.concatenate(inp, axis=0)
    print(inp.shape)

class OffenseNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
    
    def forward(self, x):
        pass
    
    def calc_action(self, state):
        pass
