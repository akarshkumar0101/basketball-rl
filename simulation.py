
import torch

import constant


def run_simulation(model_o, model_d, init_game_state, with_grad=False, return_all_game_states=True, pbar=None):
    if return_all_game_states:
        all_game_states = [init_game_state]

    current_game_state = init_game_state
    
    batch_size = init_game_state.shape[0]
    
    with torch.set_grad_enabled(with_grad):
        
        if pbar is not None:
            pbar.reset(total=constant.num_game_steps-1)
        for game_step in range(constant.num_game_steps-1):
            tss = (game_step/(constant.num_game_steps-2)) * torch.ones(batch_size, 1, 1)
            
            move_o = model_o(current_game_state, tss)
            move_d = model_d(current_game_state, tss)

            current_game_state = current_game_state.clone()
            current_game_state[:, constant.idxs_oentities, :] += move_o
            current_game_state[:, constant.idxs_dp, :] += move_d
            current_game_state = current_game_state.clamp(-1, 1)
            
            if return_all_game_states:
                all_game_states.append(current_game_state)

            if pbar is not None:
                pbar.update()
                
    if return_all_game_states:
        all_game_states = torch.stack(all_game_states, dim=1)
        return all_game_states
    else:
        return current_game_state


