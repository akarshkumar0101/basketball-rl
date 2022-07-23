#%load_ext autoreload
#%autoreload 2

import os
import copy
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.signal
import xarray as xr

import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb

import util
import torch_dict
import data
import agent
import animation
import constants
import constants_ui
import Constant

import discrete_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

