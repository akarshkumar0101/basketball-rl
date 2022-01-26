import numpy as np
import gym

import json
import copy

import celluloid
from IPython.display import HTML  # to show the animation in Jupyter

import pymunk

import matplotlib.pyplot as plt

from ppp.ppp_hc import PPPHC

# from gym import error, spaces, utils
# from gym.utils import seeding


class Player:
    def __init__(self, name):
        self.name = name
        self.stats = {
            "height": 1.88,  # m
            "mass": 83,  # kg
            "max_speed": 6.7,  # m/s
            "max_acceleration": 2.95,  # m/s^2, https://www.wired.com/2012/08/maximum-acceleration-in-the-100-m-dash/
            "speed_pass": 14.6,  # m/s
            "radius": 0.25,  # m # smaller that actual width of .3 bc it is average in all directions
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
            "shot_clock_max": 24.0,
            "n_offensive_players": 5,
            "n_defensive_players": 5,
            "init_pos_players": "random",
            "pos_hoop": [0, 6.477],
            "out_of_bounds": [-7.62, -7.62, 7.62, 7.62],  # m
            "boundary": [-9.144, -9.144, 9.144, 9.144],  # m
            "distance_3_point": 6.33,  # m
            "thickness_3_point": 1.0,  # m
            "radius_ball": 0.12,  # m
            "radius_hoop": 0.31,  # m
            "use_segment_walls": False,
        }
        if config is not None:
            self.config.update(config)

        self.space = pymunk.Space()

        xmin, ymin, xmax, ymax = self.config["boundary"]
        self.boundary = xmax  # simulation boundary
        xmin, ymin, xmax, ymax = self.config["out_of_bounds"]
        self.oob = xmax  # out of bounds
        wall = pymunk.Body(0, 0, pymunk.Body.STATIC)
        b = self.boundary
        bthick = 2
        if self.config["use_segment_walls"]:
            # the thickness that is fed into the segment is (.5 times) the *display* thickness for the matplotlib line
            # not sure, but don't think that this is a simulation thickness
            walls = [
                pymunk.Segment(wall, (-b, -b), (-b, b), bthick * 10),
                pymunk.Segment(wall, (-b, b), (b, b), bthick * 10),
                pymunk.Segment(wall, (b, b), (b, -b), bthick * 10),
                pymunk.Segment(wall, (b, -b), (-b, -b), bthick * 10),
            ]
        else:
            overlap = 1.3
            walls = [
                pymunk.Poly(
                    wall,
                    [
                        (-b * overlap, -b),
                        (b * overlap, -b),
                        (-b * overlap, -b - bthick),
                        (b * overlap, -b - bthick),
                    ],
                ),  # bottom
                pymunk.Poly(
                    wall,
                    [
                        (-b * overlap, b),
                        (b * overlap, b),
                        (-b * overlap, b + bthick),
                        (b * overlap, b + bthick),
                    ],
                ),  # top
                pymunk.Poly(
                    wall,
                    [
                        (-b, -b * overlap),
                        (-b, b * overlap),
                        (-b - bthick, -b * overlap),
                        (-b - bthick, b * overlap),
                    ],
                ),  # left
                pymunk.Poly(
                    wall,
                    [
                        (b, -b * overlap),
                        (b, b * overlap),
                        (b + bthick, -b * overlap),
                        (b + bthick, b * overlap),
                    ],
                ),  # right
            ]
        self.space.add(wall, *walls)

        self.shot_clock = 0.0

        self.done = False

        def add_body_to_space(mass, radius, offense=True):
            xy = tuple(np.random.uniform(-7.5, 7.5, size=2).astype(np.float32))
            moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, moment)
            body.position = xy
            body.start_position = pymunk.Vec2d(*body.position)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.9999999
            self.space.add(body, shape)
            shape.color = (0, 255, 0, 100) if offense else (255, 0, 0, 100)
            return body

        self.players_offense = []
        for i in range(self.config["n_offensive_players"]):
            player = Player(f"O{i}")
            player.body = add_body_to_space(
                player.stats["mass"],
                player.stats["radius"],
                offense=True,
            )
            self.players_offense.append(player)

        self.players_defense = []
        for i in range(self.config["n_defensive_players"]):
            player = Player(f"D{i}")
            player.body = add_body_to_space(
                player.stats["mass"],
                player.stats["radius"],
                offense=False,
            )
            self.players_defense.append(player)

        self.players_all = [*self.players_offense, *self.players_defense]

        # not keeping ball in the pymunk simulation
        self.ball_state = {
            "dribbling": True,  # dribbling or passing
            "dribbler": 0,  # dribbler/passer
            "receiver": None,
            "pass_elapsed_time": None,
            "pass_total_time": None,
        }

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)

        self.ppp_calc = PPPHC(self)

    def evaluate_ppp(self):
        ppp = self.ppp_calc.evaluate_ppp()
        if True or self.ball_state["dribbling"]:  # only if dribbling
            return ppp
        else:  # zero if passing
            return 0.0

    def get_state(self):
        state = {}
        state["shot_clock"] = self.shot_clock

        # player, pos/vel, x/y
        posvels = np.zeros((len(self.players_all), 2, 2), dtype=np.float32)
        for i, player in enumerate(self.players_all):
            posvels[i] = body2pos_vel(player.body)
        state["posvels"] = posvels

        state["ball_state"] = copy.deepcopy(self.ball_state)
        return state

    def set_state(self, state):
        self.shot_clock = state["shot_clock"]

        for i, player in enumerate(self.players_all):
            player.body.position = tuple(state["posvels"][i][0])
            player.body.velocity = tuple(state["posvels"][i][1])

        self.ball_state = copy.deepcopy(state["ball_state"])

    def get_pos_ball(self, state=None):
        if state is None:
            state = self.state
        ball_state = state["ball_state"]

        if ball_state["dribbling"]:
            # body2pos_vel(self.players_offense[ball_state["dribbler"]].body)[0]
            return state["posvels"][ball_state["dribbler"], 0]
        else:
            # a = body2pos_vel(self.players_offense[ball_state["dribbler"]].body)[0]
            # b = body2pos_vel(self.players_offense[ball_state["receiver"]].body)[0]
            a = state["posvels"][ball_state["dribbler"], 0]
            b = state["posvels"][ball_state["receiver"], 0]
            t = ball_state["pass_elapsed_time"] / ball_state["pass_total_time"]
            return a + (b - a) * t

    def reset(self):
        if self.config["init_pos_players"] == "random":
            for player in self.players_all:
                player.body.position = tuple(np.random.uniform(-7.5, 7.5, size=2).astype(np.float32))
                player.body.velocity = (0, 0)
            self.ball_state = {
                "dribbling": True,  # dribbling or passing
                "dribbler": 0,  # dribbler/passer
                "receiver": None,
                "pass_elapsed_time": None,
                "pass_total_time": None,
            }
        

        self.state = self.get_state()
        self.states = [self.state]

        return self.state

    def close(self):
        pass

    def step(self, action):
        dt = 1.0 / self.config["fps"]

        # inputting the acceleration actions
        for player, acc in zip(self.players_all, action['accs']):
            max_acc = player.stats["max_acceleration"]
            player.body.force = tuple(acc * max_acc * player.body.mass)

        # enforcing max velocity+acceleration
        for player in self.players_all:
            vel, force = player.body.velocity, player.body.force
            mass = player.body.mass
            max_speed = player.stats["max_speed"]
            max_force = player.stats["max_acceleration"] * mass
            if abs(vel) > max_speed:
                player.body.velocity = vel.scale_to_length(max_speed)
            if abs(force) > max_force:
                player.body.force = force.scale_to_length(max_force)

        self.space.step(dt)
        self.shot_clock += dt

        self.done = (
            self.shot_clock >= self.config["shot_clock_max"] or action["shooting"]
        )

        if self.ball_state["dribbling"]:
            if (
                action["passing"]
                and action["pass_receiver"] is not self.ball_state["dribbler"]
            ):
                pass_total_time = 1.0  # TODO: set time based on distance of pass
                self.ball_state = {
                    "dribbling": False,
                    "dribbler": self.ball_state["dribbler"],
                    "receiver": action["pass_receiver"],
                    "pass_elapsed_time": 0.0,
                    "pass_total_time": pass_total_time,
                }
        else:  # passing
            self.ball_state["pass_elapsed_time"] += dt
            if (
                self.ball_state["pass_elapsed_time"]
                > self.ball_state["pass_total_time"]
            ):
                self.ball_state = {
                    "dribbling": True,
                    "dribbler": self.ball_state["receiver"],
                    "receiver": None,
                    "pass_elapsed_time": None,
                    "pass_total_time": None,
                }

        reward = 0.0
        info = None

        if action["shooting"]:
            reward = 1.0

        self.state = self.get_state()
        self.states.append(self.state)
        return self.state, reward, self.done, info
        # return obs, reward, done, info

    def render(
        self, state=None, mode="human", backend="manual", ax=None, player_names=False
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if state is None:
            state = self.state

        plt.sca(ax)

        # (or if you have an existing figure)
        #     fig, ax = plt.gcf(), fig.gca()

        n_offense, n_defense = len(self.players_offense), len(self.players_defense)
        xy_offense = state["posvels"][:n_offense, 0, :]
        xy_defense = state["posvels"][n_offense:, 0, :]
        if backend == "manual":
            for i, (player, (x, y)) in enumerate(zip(self.players_offense, xy_offense)):
                circle = plt.Circle(
                    (x, y),
                    radius=player.stats["radius"],
                    color="g",
                    label="Offense" if i == 0 else None,
                )
                ax.add_artist(circle)
                if player_names:
                    ax.text(x, y, player.name)
            for i, (player, (x, y)) in enumerate(zip(self.players_defense, xy_defense)):
                circle = plt.Circle(
                    (x, y),
                    radius=player.stats["radius"],
                    color="r",
                    label="Defense" if i == 0 else None,
                )
                ax.add_artist(circle)
                if player_names:
                    ax.text(x, y, player.name)
            for x, y in [self.get_pos_ball(state)]:
                circle = plt.Circle(
                    (x, y), radius=self.config["radius_ball"], color="orange"
                )
                ax.add_artist(circle)

        elif backend == "pymunk":
            # TODO: currently only supports the current state of env
            o = pymunk.matplotlib_util.DrawOptions(ax)
            self.space.debug_draw(o)

        ax.add_artist(
            plt.Circle(
                np.array(self.config["pos_hoop"]),
                radius=self.config["radius_hoop"],
                color="orange",
                fill=False,
                label="hoop",
            )
        )
        ax.add_artist(
            plt.Circle(
                np.array(self.config["pos_hoop"]),
                radius=self.config["distance_3_point"],
                color="k",
                fill=False,
                label="3-point line",
            )
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        xmin, ymin, xmax, ymax = self.config["out_of_bounds"]
        ax.axvline(xmin, c="gray")
        ax.axvline(xmax, c="gray")
        ax.axhline(ymin, c="gray")
        ax.axhline(ymax, c="gray", label="out of bounds")

        xmin, ymin, xmax, ymax = self.config["boundary"]
        ax.axvline(xmin, c="red")
        ax.axvline(xmax, c="red")
        ax.axhline(ymin, c="red")
        ax.axhline(ymax, c="red", label="boundary")
        ax.barh(
            y=ymin,
            left=xmin,
            width=(xmax - xmin) * state["shot_clock"] / self.config["shot_clock_max"],
            height=(ymax - ymin) * 0.03,
            color="purple",
            align="center",
        )
        plt.xlim(xmin * 1.02, xmax * 1.02)
        plt.ylim(ymin * 1.02, ymax * 1.02)
        # plt.xlim(-12, 12)
        # plt.ylim(-12, 12)
        ax.set_aspect("equal")
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
        states = np.array(states)
        if mode in ["vid", "animation", "file"]:
            camera = celluloid.Camera(fig)  # the camera gets the fig we'll plot
            for state in tqdm(states):
                self.render(state=state, ax=ax)
                camera.snap()  # the camera takes a snapshot of the plot
                # ax.clear()
            animation = camera.animate(
                interval=(1000 * 1.0 / self.config["fps"])
            )  # animation ready
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
        elif mode == "overlay":
            for state in tqdm(states):
                self.render(state=state, ax=ax)
            return fig
        elif mode == "grid":
            nrows, ncols = kwargs["nrows"], kwargs["ncols"]
            states = states[:: (len(states) // (nrows * ncols))]
            for r in range(nrows):
                for c in range(ncols):
                    cax = fig.add_subplot(nrows, ncols, r * ncols + c + 1)
                    self.render(state=states[r * ncols + c], ax=cax)
            plt.tight_layout()
            return fig


def body2pos_vel(body):
    return np.stack([np.array(body.position), np.array(body.velocity)])


def get_random_action(env, state):
    pass_receiver = np.random.randint(low=0, high=len(env.players_offense), size=None)
    action = {
        "shooting": False,
        "accs": np.random.randn(len(env.players_all), 2),
        "passing": True,
        "pass_receiver": pass_receiver,
    }
    return action
