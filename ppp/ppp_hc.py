"""
maps from 
array of player ID, height, weight, vertical, age, O/D, years experience
to
PPP

using a neural network trained on the real life data or NBA-2k data.
"""


import numpy as np
import torch


class PPPHC:
    def __init__(self, env):
        self.env = env
        self.pos_hoop = torch.tensor(self.env.config["pos_hoop"])

    def evaluate_ppp(self, env=None):
        if env is None:
            env = self.env
        posvels = env.state["posvels"]
        pos_op = posvels[: len(env.players_offense), 0, :]  # only look at position
        pos_dp = posvels[len(env.players_offense) :, 0, :]  # only look at position

        pos_op = torch.from_numpy(pos_op)
        pos_dp = torch.from_numpy(pos_dp)

        # TODO: encorporate the velocity when considering raw ppp
        # velocity towards hoop is fine for layop floater, but is bad for three pointer
        # and velocity sideways to hoop is almost always bad, etc.

        ppp = self.points_per_possession(pos_op, pos_dp)
        return ppp

    # general functions
    def distance(self, a, b=None):
        """Computes the distance between pos and pos_base.
        pos should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop
        return (a - b).norm(dim=-1)

    def direction(self, a, b=None):
        """Computes the direction of pos from pos_base.
        pos should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop
        locations = a - b
        return torch.atan2(
            locations[..., 1], locations[..., 0] + 0.0001
        )  # TODO: change this to be more stable

    def polar_coordinates(self, a, b=None):
        """Computes the distance and angle of pos from pos_base.
        pos should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop
        return self.distance(a, b), self.direction(a, b)

    # basketball functions
    def raw_accuracy(self, a, b=None):
        """The accuracy of a player at pos_op, given the hoop is at pos_base.
        pos_op should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos_op and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop
        r = self.distance(a, b)
        accuracy = torch.exp(
            -((r / 7.62) ** 2)
        )  # guassian is be better for closer shots
        #     accuracy = 1/torch.exp(r)
        return accuracy

    def raw_points(self, a, b=None):
        """The amount of points a player at pos_op will score, given the basket is at pos_base.
        pos_op should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos_op and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop
        r = self.distance(a, b)
        t = -np.log(1.0 / 0.954 - 1.0) / self.env.config["thickness_3_point"]
        d3 = self.env.config["distance_3_point"]
        point_scale = 2 + 1 / (1 + torch.exp(-t * (r - d3)))
        return point_scale

    def raw_contest_distance(self, dist):
        """The contest coefficient a player will feel if someone is dist away from them.
        dist can be of any shape."""
        return 1 - torch.exp(-50 * dist / 7.62)

    def raw_contest_angle(self, theta):
        """The contest coefficient a player will feel if someone is theta degrees away from the direction of the.
        theta can be of any shape."""
        high = 0.8
        low = 0.4
        return (torch.cos(theta - np.pi) + 1) / 2.0 * (high - low) + low

    def contest(self, x, y, b=None):
        """The contest a player at pos_op will feel with defenders at pos_dp, given the basket is at pos_base.
        pos_op should be of shape (..., 2)
        pos_dp should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos_op, pos_dp, and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop

        theta_hoop = self.direction(b, x)  # theta of hoop from op
        theta_defender = self.direction(y, x)  # theta of dp from op

        # directional contest on op from dp
        contest_directional = self.raw_contest_angle(theta_hoop - theta_defender)

        r = self.distance(y, x)  # distance of dp from op
        # distance contest on op from dp
        contest_distance = self.raw_contest_distance(r)
        # weight two combination of terms based on distance of dp from op
        weight = torch.tanh(3 * r / 7.62)
        return weight * contest_distance + (1 - weight) * contest_directional
        # return contest_distance

    def points_per_possession(self, x, y, b=None):
        """The points per possession a player at pos_op has with defenders at pos_dp and hoop at pos_base.
        pos_op should be of shape (..., 2)
        pos_dp should be of shape (..., 2)
        pos_base should be of shape (..., 2)
        pos_op, pos_dp, and pos_base should be broadcastable to each other."""
        if b is None:
            b = self.pos_hoop

        # points per possession
        no_contest_ppp = self.raw_accuracy(x, b) * self.raw_points(x, b)

        if y is None:
            contest_coeff = 1.0
        else:
            contest_coeff = self.contest(x, y, b)
            # all dps contest all ops
            contest_coeff = contest_coeff.prod(dim=-1, keepdim=True)
        return contest_coeff * no_contest_ppp
