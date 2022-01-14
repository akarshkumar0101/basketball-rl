import numpy as np
import gym

import json

import celluloid
from IPython.display import HTML  # to show the animation in Jupyter

import pymunk

import matplotlib.pyplot as plt

# from gym import error, spaces, utils
# from gym.utils import seeding


class Player:
    def __init__(self, name):
        self.name = name
        self.stats = {
            "height": 1.88,  # m
            "mass": 83,  # kg
            "speed": 6.7,  # m/s
            "acceleration": 2.95,  # m/s^2, https://www.wired.com/2012/08/maximum-acceleration-in-the-100-m-dash/
            "speed_ball": 14.6,  # m/s
            "radius": 0.27,  # m
        }


# 50 ft -> 2 -> 15.24 m
# 7.62


class BasketballEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        # with open("config/default_env.json", "r") as f:
        # self.config = json.load(f)
        self.config = {
            "fps": 10,
            "shot_clock": 24,
            "n_offensive_players": 5,
            "n_densive_players": 5,
            "init_pos_players": "random",
            "pos_hoop": [0, 6.477],
            "out_of_bounds": [-7.62, -7.62, 7.62, 7.62],  # m
            "boundary": [-9.144, -9.144, 9.144, 9.144],  # m
            "distance_three_point": 6.33,  # m
            "radius_ball": 0.12,  # m
            "radius_hoop": 0.31,  # m
        }
        if config is not None:
            self.config.update(config)

        self.space = pymunk.Space()

        xmin, ymin, xmax, ymax = self.config["boundary"]
        self.boundary = xmax  # simulation boundary
        xmin, ymin, xmax, ymax = self.config["out_of_bounds"]
        self.oob = xmax  # out of bounds
        wall = pymunk.Body(0, 0, pymunk.Body.STATIC)
        bthick = 0.2
        b = self.boundary + bthick
        walls = [
            pymunk.Segment(wall, (-b, -b), (-b, b), bthick),
            pymunk.Segment(wall, (-b, b), (b, b), bthick),
            pymunk.Segment(wall, (b, b), (b, -b), bthick),
            pymunk.Segment(wall, (b, -b), (-b, -b), bthick),
        ]
        self.space.add(wall, *walls)

        self.shot_clock = 0.0
        self.shot_clock_max = 24.0

        self.done = False

        def add_body_to_space(mass, radius):
            xy = tuple(np.random.uniform(-7.5, 7.5, size=2).astype(np.float32))
            moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, moment)
            body.position = xy
            body.start_position = pymunk.Vec2d(*body.position)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.9999999
            self.space.add(body, shape)
            return body

        self.stats_offense = [Player(f"i") for i in range(5)]
        self.stats_defense = [Player(f"i") for i in range(5)]

        self.bodies_offense = [add_body_to_space(1.0, 0.03) for _ in range(5)]
        self.bodies_defense = [add_body_to_space(1.0, 0.03) for _ in range(5)]
        self.bodies_players = [*self.bodies_offense, *self.bodies_defense]

        # self.body_ball = add_body_to_space(1.0, 0.02)
        self.body_ball = 0
        self.ball_passing = False

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)

    def reset(self):
        if self.config["init_pos_players"] == "random":
            pass

        self.state = State(self)
        self.states = [self.state]

        return self.state

    def close(self):
        pass

    def step(self, action):
        dt = 1.0 / self.config["fps"]

        self.space.step(dt)
        self.shot_clock += dt

        self.done = self.shot_clock >= self.shot_clock_max or action.shooting

        if (
            type(self.body_ball) is int
            and type(action.passdata) is int
            and self.body_ball != action.passdata
        ):
            self.body_ball = (self.body_ball, action.passdata, 0.0)
            self.ball_passing = True
        elif self.ball_passing:
            a, b, t = self.body_ball
            t += 0.1
            self.body_ball = (a, b, t)
            if t >= 1.0:
                self.body_ball = b
                self.ball_passing = False

        reward = 0.0
        info = None

        if action.shooting:
            reward = 1.0

        self.state = State(self)
        self.states.append(self.state)
        return self.state, reward, self.done, info
        # return obs, reward, done, info

    def render(self, state=None, mode="human", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if state is None:
            state = self.state

        # (or if you have an existing figure)
        #     fig, ax = plt.gcf(), fig.gca()
        ax.set_aspect("equal")

        xmin, ymin, xmax, ymax = self.config["boundary"]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        xmin, ymin, xmax, ymax = self.config["out_of_bounds"]
        ax.axvline(xmin, c="gray")
        ax.axvline(xmax, c="gray")
        ax.axhline(ymin, c="gray")
        ax.axhline(ymax, c="gray", label="out of bounds")

        # for x, y in state.locs_offense:
        for player, (x, y) in zip(self.stats_offense, state.locs_offense):
            circle = plt.Circle((x, y), radius=player.stats["radius"], color="g")
            ax.add_artist(circle)
        for player, (x, y) in zip(self.stats_defense, state.locs_defense):
            circle = plt.Circle((x, y), radius=player.stats["radius"], color="r")
            ax.add_artist(circle)
        for x, y in [state.loc_ball]:
            circle = plt.Circle(
                (x, y), radius=self.config["radius_ball"], color="orange"
            )
            ax.add_artist(circle)

        ax.add_artist(
            plt.Circle(
                np.array(self.config["pos_hoop"]),
                radius=self.config["radius_hoop"],
                color="orange",
                fill=False,
            )
        )
        ax.add_artist(
            plt.Circle(
                np.array(self.config["pos_hoop"]),
                radius=self.config["distance_three_point"],
                color="k",
                fill=False,
            )
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        xmin, ymin, xmax, ymax = self.config["boundary"]
        ax.barh(
            y=ymin,
            left=xmin,
            width=(xmax - xmin) * state.shot_clock / self.config["shot_clock"],
            height=(ymax - ymin) * 0.03,
            color="purple",
            align="center",
        )
        # plt.grid()
        # plt.legend()

        return ax

    def render_episode(
        self, states=None, mode="vid", fig=None, ax=None, tqdm=lambda x: x, **kwargs
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if states is None:
            states = self.states
        if mode in ["vid", "animation", "file"]:
            camera = celluloid.Camera(fig)  # the camera gets the fig we'll plot
            for state in tqdm(states):
                self.render(state=state, ax=ax)
                camera.snap()  # the camera takes a snapshot of the plot
                # ax.clear()
            animation = camera.animate()  # animation ready
            # plt.cla()
            # plt.clf()
            plt.close()

            if mode == "animation":
                return animation
            elif mode == "vid":
                vid = HTML(animation.to_html5_video())  # displaying the animation
                return animation, vid
            elif mode == "file":
                animation.save(kwargs["filename"])
                return animation

        elif mode == "grid":
            nrows, ncols = kwargs["nrows"], kwargs["ncols"]
            states = np.array(states)[:: (len(states) // (nrows * ncols))]
            for r in range(nrows):
                for c in range(ncols):
                    cax = fig.add_subplot(nrows, ncols, r * ncols + c + 1)
                    self.render(state=states[r * ncols + c], ax=cax)
            plt.tight_layout()
            return fig


def get_body_position_numpy(body):
    return np.array([body.position.x, body.position.y])


class State:
    def __init__(self, env):
        self.shot_clock = env.shot_clock

        self.locs_offense = np.stack(
            [get_body_position_numpy(b) for b in env.bodies_offense]
        ).astype(np.float32)
        self.locs_defense = np.stack(
            [get_body_position_numpy(b) for b in env.bodies_defense]
        ).astype(np.float32)

        if type(env.body_ball) is int:
            self.loc_ball = get_body_position_numpy(
                env.bodies_offense[env.body_ball]
            ).astype(np.float32)
        else:
            a, b, t = env.body_ball
            a = get_body_position_numpy(env.bodies_offense[a]).astype(np.float32)
            b = get_body_position_numpy(env.bodies_offense[b]).astype(np.float32)
            self.loc_ball = (1 - t) * a + t * b


class Action:
    def __init__(self):
        self.shooting = False
        self.accs = None
        self.passdata = np.random.randint(low=0, high=5, size=None)
