import gym

import json

# from gym import error, spaces, utils
# from gym.utils import seeding

class PlayerStat:
    def __init__(self):
        pass
    
    
class BasketballEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, config=None):
        with open('config/default_env.json', 'r') as f:
            self.config = json.load(f)
        if config is not None:
            self.config.update(config)



        
        self.space = pymunk.Space()
        self.boundary = 1.1 # simulation boundary
        self.oob = 1. # out of bounds
        
        wall = pymunk.Body(0, 0, pymunk.Body.STATIC)
        bthick = .2
        b = self.boundary+bthick
        walls = [
            pymunk.Segment(wall, (-b, -b), (-b, b), bthick),
            pymunk.Segment(wall, (-b, b), (b, b), bthick),
            pymunk.Segment(wall, (b, b), (b, -b), bthick),
            pymunk.Segment(wall, (b, -b), (-b, -b), bthick),
        ]
        self.space.add(wall, *walls)
        
        self.shot_clock = 0.
        self.shot_clock_max = 24.
        
        self.done = False
        
        self.fps = 4.2
        
        def add_body_to_space(mass, radius):
            xy = tuple(np.random.uniform(-1, 1, size=2).astype(np.float32))
            moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, moment)
            body.position = xy
            body.start_position = pymunk.Vec2d(*body.position)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.9999999
            self.space.add(body, shape)
            return body
        
        self.stats_offense = [PlayerStat() for _ in range (5)]
        self.stats_defense = [PlayerStat() for _ in range (5)]
        
        self.bodies_offense = [add_body_to_space(1.0, 0.03) for _ in range(5)]
        self.bodies_defense = [add_body_to_space(1.0, 0.03) for _ in range(5)]
        self.bodies_players = [*self.bodies_offense, *self.bodies_defense]
        
        # self.body_ball = add_body_to_space(1.0, 0.02)
        self.body_ball = 0
        self.ball_passing = False
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)

    def reset(self):
        if self.config['init_pos_players']=='random':
            pass
        return State(self)

    def close(self):
        pass

            
    def step(self, action):
        dt = 1./self.fps
        
        self.space.step(dt)
        self.shot_clock += dt
        
        self.done = self.shot_clock >= self.shot_clock_max or action.shooting
        
        if type(self.body_ball) is int and type(action.passdata) is int and self.body_ball!=action.passdata:
            self.body_ball = (self.body_ball, action.passdata, 0.)
            self.ball_passing = True
        elif self.ball_passing:
            a, b, t = self.body_ball
            t += 0.1
            self.body_ball = (a, b, t)
            if t>=1.:
                self.body_ball = b
                self.ball_passing = False
        
        reward = 0.
        info = None
        
        if action.shooting:
            reward = 1.
        
        return State(self), reward, self.done, info
        # return obs, reward, done, info
    

    def render(self, mode="human"):
        pass

    def render_episode(self):
        pass
    
def get_body_position_numpy(body):
    return np.array([body.position.x, body.position.y])
    
class State:
    def __init__(self, env):
        self.shot_clock = env.shot_clock
        
        self.locs_offense = np.stack([get_body_position_numpy(b) for b in env.bodies_offense]).astype(np.float32)
        self.locs_defense = np.stack([get_body_position_numpy(b) for b in env.bodies_defense]).astype(np.float32)
        
        if type(env.body_ball) is int:
            self.loc_ball = get_body_position_numpy(env.bodies_offense[env.body_ball]).astype(np.float32)
        else:
            a, b, t = env.body_ball
            a = get_body_position_numpy(env.bodies_offense[a]).astype(np.float32)
            b = get_body_position_numpy(env.bodies_offense[b]).astype(np.float32)
            self.loc_ball = (1-t)*a+t*b

class Action:
    def __init__(self):
        self.shooting = False
        self.accs = None
        self.passdata = np.random.randint(low=0, high=5, size=None)
        


def show_game_state(state, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    # (or if you have an existing figure)
#     fig = plt.gcf()
#     ax = fig.gca()
    ax.set_aspect('equal')
    
    ax.set_xlim(-env.boundary, env.boundary)
    ax.set_ylim(-env.boundary, env.boundary)
    ax.axhline(env.oob, c='gray'); ax.axvline(env.oob, c='gray')
    ax.axhline(-env.oob, c='gray'); ax.axvline(-env.oob, c='gray', label='out of bounds')
    
    for x, y in state.locs_offense:
        circle = plt.Circle((x, y), radius=constant.radius_player, color='g')
        ax.add_artist(circle)
    for x, y in state.locs_defense:
        circle = plt.Circle((x, y), radius=constant.radius_player, color='r')
        ax.add_artist(circle)
    for x, y in [state.loc_ball]:
        circle = plt.Circle((x, y), radius=constant.radius_ball, color='orange')
        ax.add_artist(circle)
    
    ax.add_artist(plt.Circle(constant.pos_hoop.numpy(), radius=constant.radius_hoop, color='orange', fill=False))
    ax.add_artist(plt.Circle(constant.pos_hoop.numpy(), radius=constant.radius_three_point_line, color='k', fill=False))
    
    ax.set_title('Game State')
    ax.set_xlabel('X-axis'); ax.set_ylabel('Y-axis')
    
    ax.barh(y=-1.03, width=2*state.shot_clock/env.shot_clock_max, left=-1., height=0.05, color='purple')
    # plt.grid(
    
    # plt.legend()
    
    return ax

import celluloid
from IPython.display import HTML # to show the animation in Jupyter

def animate_game_states(states, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    camera = celluloid.Camera(fig)# the camera gets the fig we'll plot
    for state in states:
        show_game_state(state, ax)
        camera.snap() # the camera takes a snapshot of the plot
    animation = camera.animate() # animation ready
    plt.close()
    vid = HTML(animation.to_html5_video()) # displaying the animation
    return vid