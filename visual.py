

import matplotlib.pyplot as plt
import matplotlib.animation

import constant


def show_game_state(game_state, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8)) # note we must use plt.subplots, not plt.subplot
    # (or if you have an existing figure)
#     fig = plt.gcf()
#     ax = fig.gca()
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    
    for idx_op in constant.idxs_op.numpy():
        x, y = game_state[idx_op]
        circle = plt.Circle((x, y), radius=constant.radius_player, color='b')
        ax.add_artist(circle)
    for idx_dp in constant.idxs_dp.numpy():
        x, y = game_state[idx_dp]
        circle = plt.Circle((x, y), radius=constant.radius_player, color='r')
        ax.add_artist(circle)
    for idx_ball in constant.idxs_ball.numpy():
        x, y = game_state[idx_ball]
        circle = plt.Circle((x, y), radius=constant.radius_ball, color='orange')
        ax.add_artist(circle)
    
    ax.add_artist(plt.Circle(constant.pos_hoop.numpy(), radius=constant.radius_hoop, color='orange', fill=False))
    ax.add_artist(plt.Circle(constant.pos_hoop.numpy(), radius=constant.radius_three_point_line, color='orange', fill=False))
    
    ax.set_title('Game State')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    return ax


def show_game_animation(all_game_states, fps=10):
    num_games = len(all_game_states)
    fig, axs = plt.subplots(1, num_games, figsize=(2*num_games, 2))
    
    all_entities = []
    for ax in axs:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.add_artist(plt.Circle(constant.pos_hoop.numpy(), radius=constant.radius_hoop, color='orange', fill=False))
        ax.add_artist(plt.Circle(constant.pos_hoop.numpy(), radius=constant.radius_three_point_line, color='orange', fill=False))
        
        entities = []
        for _ in constant.idxs_op.numpy():
            circle = plt.Circle((0, 0), radius=constant.radius_player, color='b')
            entities.append(circle)
        for _ in constant.idxs_dp.numpy():
            circle = plt.Circle((0, 0), radius=constant.radius_player, color='r')
            entities.append(circle)
        for _ in constant.idxs_ball.numpy():
            circle = plt.Circle((0, 0), radius=constant.radius_ball, color='orange')
            entities.append(circle)
            
        all_entities.append(entities)
    
    time_text = axs[num_games//2].text(0., -0.9, '')

    def init():
        for idx_ax, ax in enumerate(axs):
            entities = all_entities[idx_ax]
            for idx_entity, entity in enumerate(entities):
                pos = all_game_states[idx_ax, 0, idx_entity, :]
                entity.center = (pos[0].item(), pos[1].item())
                ax.add_artist(entity)
                
        time_text.set_text(f'{0.0} sec')
        return entities + [time_text]

    def animate(game_step):
        for idx_ax, ax in enumerate(axs):
            entities = all_entities[idx_ax]
            for idx_entity, entity in enumerate(entities):
                pos = all_game_states[idx_ax, game_step, idx_entity, :]
                entity.center = (pos[0].item(), pos[1].item())
                
        time_text.set_text(f'{game_step/fps} sec')
        return entities + [time_text]
    
    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=constant.num_game_steps, blit=True,
                                              interval=1000/fps)
    return anim

