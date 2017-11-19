from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from collections import deque
from contextlib import contextmanager

from baselines import logger
from baselines.common.cg import cg
from baselines.common import colorize
from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common import explained_variance, zipsame, dataset


def learn(env, policy_func,
          timesteps_per_batch,
          max_kl,
          cg_iters,
          gamma,
          lam,
          entcoeff,
          cg_damping,
          vf_stepsize,
          vf_iters,
          max_timesteps,
          max_episodes,
          max_iters,
          callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # setip losses and stuff
    ob_space = env.observation_space
    ac_space = env.action_space


def main():
    np.set_printoptions(precision=3)