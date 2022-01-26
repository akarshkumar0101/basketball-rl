from time import time
import numpy as np
import torch
from torch import nn


"""
Process each player (and the dribbler/receiver players) with the same module (weight sharing).
Allows all players to undergo same processing and saves redundant learning.
This modules takes in a one-hot vector for player ID and two 2D vectors for player position+velocity.

Later on, I will incorporate the player statistics as an input.
"""


class ProcessPlayerModule(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        # self.seq = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 32),
        #     nn.Tanh(),
        # )
        # self.seq = nn.Sequential(
        #     nn.Linear(16, 20),
        #     nn.Tanh(),
        #     nn.Linear(20, 20),
        #     nn.Tanh(),
        #     nn.Linear(20, 10),
        # )
        self.seq = nn.Sequential(
            nn.Linear(6+len(self.env.players_all), 32),
            nn.Tanh(),
            nn.Linear(32, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 6),
        )

    def forward(self, x):
        return self.seq(x)

class OffenseNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env

        self.process_player_module = ProcessPlayerModule(self.env)
        n_in = 5+6*(len(self.env.players_all)+2)
        self.seq = nn.Sequential(
            nn.Linear(n_in, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            # nn.Linear(25, 2 * env.config["n_offensive_players"]),
            nn.Linear(25, 2 * len(env.players_all)),
        )

    def forward(self, x):
        pass

    def calc_action(self, state):
        player_info, other_info = self.calc_nn_input(state)
        # process player info through player processing module
        player_info_proc = self.process_player_module(player_info)
        info = torch.cat([player_info_proc.flatten(), other_info], dim=0)
        x = self.seq(info)
        accs = x.reshape(-1, 2).detach().numpy()
        # print(x.shape)

        # calculate action
        action = {
            "shooting": False,
            "accs": accs,
            "passing": False,
            "pass_receiver": None,
        }
        return action

    def calc_nn_input(self, state):
        # time_stamp
        sc = state["shot_clock"]
        scm = self.env.config["shot_clock_max"]
        time_stamp = (sc - scm / 2.0) / (scm / 2.0)
        time_stamp = torch.tensor(time_stamp).float()

        # player_info
        posvels = state["posvels"]
        p_team = torch.zeros(posvels.shape[0], 2)
        p_team[: self.env.config["n_offensive_players"], 0] = 1.0
        p_team[self.env.config["n_defensive_players"] :, 1] = 1.0
        p_idx = torch.eye(len(posvels))
        p_pos = torch.from_numpy(posvels[:, 0, :]/7.5)
        p_vel = torch.from_numpy(posvels[:, 1, :]/2.95)
        player_info = torch.cat([p_team, p_idx, p_pos, p_vel], dim=1)
        empty_player_input = torch.zeros_like(player_info[[0]])

        # pass duration info
        ball_state = state["ball_state"]
        dribbling = ball_state["dribbling"]
        dribbler = ball_state["dribbler"]
        receiver = ball_state["receiver"]
        pass_elapsed_time = ball_state["pass_elapsed_time"]
        pass_total_time = ball_state["pass_total_time"]
        pass_progress = pass_elapsed_time / pass_total_time if not dribbling else 0.0
        pass_progress = torch.tensor(pass_progress * 2.0 - 1.0)
        pass_total_time = torch.tensor(pass_total_time if not dribbling else 0.0)
        v_dribbling = torch.eye(2)[int(dribbling)]
        v_dribbler = player_info[[dribbler]]
        v_receiver = empty_player_input if receiver is None else player_info[[receiver]]

        # all player info (including passer and receiver)
        player_info = torch.cat([player_info, v_dribbler, v_receiver], dim=0)
        # all other infor (time, dribbling, pass, etc.)
        other_info = torch.cat(
            [time_stamp[None], v_dribbling, pass_progress[None], pass_total_time[None]],
            dim=0,
        )
        return player_info, other_info
