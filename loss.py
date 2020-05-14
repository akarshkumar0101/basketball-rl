
import numpy as np
import torch

import constant


# general functions
def distance(pos, pos_base=constant.pos_hoop):
    """Computes the distance between pos and pos_base.
    pos should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos and pos_base should be broadcastable to each other. """
    return (pos-pos_base).norm(dim=-1)


def direction(pos, pos_base=constant.pos_hoop):
    """Computes the direction of pos from pos_base.
    pos should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos and pos_base should be broadcastable to each other. """
    locations = pos - pos_base
    return torch.atan2(locations[..., 1], locations[..., 0] + 0.0001)  # TODO: change this to be more stable


def polar_coordinates(pos, pos_base=constant.pos_hoop):
    """Computes the distance and angle of pos from pos_base.
    pos should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos and pos_base should be broadcastable to each other. """
    return distance(pos, pos_base), direction(pos, pos_base)


# basketball functions
def raw_accuracy(pos_op, pos_base=constant.pos_hoop):
    """The accuracy of a player at pos_op, given the hoop is at pos_base.
    pos_op should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos_op and pos_base should be broadcastable to each other. """
    r = distance(pos_op, pos_base)
    accuracy = 1/torch.exp(r)
    return accuracy


def raw_points(pos_op, pos_base=constant.pos_hoop):
    """The amount of points a player at pos_op will score, given the basket is at pos_base.
    pos_op should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos_op and pos_base should be broadcastable to each other. """
    r = distance(pos_op, pos_base)
    point_scale = 2 + 1 / (1 + torch.exp(-50 * (r-0.9)))
    return point_scale


def raw_contest_distance(dist):
    """The contest coefficient a player will feel if someone is dist away from them.
    dist can be of any shape. """
    return 1 - torch.exp(-50*dist)


def raw_contest_angle(theta):
    """The contest coefficient a player will feel if someone is theta degrees away from the direction of the.
    theta can be of any shape. """
    
    high = .8
    low = .4
    return (torch.cos(theta-np.pi)+1)/2.*(high-low)+low


def contest(pos_op, pos_dp, pos_base=constant.pos_hoop):
    """The contest a player at pos_op will feel with defenders at pos_dp, given the basket is at pos_base.
    pos_op should be of shape (..., 2)
    pos_dp should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos_op, pos_dp, and pos_base should be broadcastable to each other. """
    
    theta_hoop = direction(pos_base, pos_op) # theta of hoop from op
    theta_defender = direction(pos_dp, pos_op) # theta of dp from op
    
    contest_directional = raw_contest_angle(theta_hoop - theta_defender) # directional contest on op from dp
    
    r = distance(pos_dp, pos_op)  # distance of dp from op
    contest_distance = raw_contest_distance(r)  # distance contest on op from dp
    
    weight = torch.tanh(3*r)  # weight two combination of terms based on distance of dp from op
    return weight*contest_distance + (1-weight)*contest_directional


def points_per_possession(pos_op, pos_dp, pos_base=constant.pos_hoop):
    """The points per possession a player at pos_op has with defenders at pos_dp and hoop at pos_base.
    pos_op should be of shape (..., 2)
    pos_dp should be of shape (..., 2)
    pos_base should be of shape (..., 2)
    pos_op, pos_dp, and pos_base should be broadcastable to each other. """
    
    # points per possession
    no_contest_ppp = raw_accuracy(pos_op, pos_base) * raw_points(pos_op, pos_base)
    contest_coeff = contest(pos_op, pos_dp, pos_base)
    contest_coeff = contest_coeff.prod(dim=-1)[..., None] # all dps contest all ops
    return contest_coeff * no_contest_ppp


