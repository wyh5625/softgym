import os.path as osp
import argparse
import numpy as np
import os
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt

import plotly.express as px
import pandas as pd

import time

import ast

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
import math
from data_structure import *
import cv2
from softgym.utils.pyflex_utils import center_object
from scipy.spatial.transform import Rotation


def create_env():
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    # parser.add_argument('--env_name', type=str, default='ClothPushPPP')
    # parser.add_argument('--headless', type=int, default=0,
    #                     help='Whether to run the environment with headless rendering')
    # parser.add_argument('--num_variations', type=int, default=1,
    #                     help='Number of environment variations to be generated')
    # parser.add_argument('--save_video_dir', type=str,
    #                     default='./data/', help='Path to the saved video')
    # parser.add_argument('--img_size', type=int, default=720,
    #                     help='Size of the recorded videos')
    # parser.add_argument('--test_depth', type=int, default=0,
    #                     help='If to test the depth rendering by showing it')
    # parser.add_argument('--tweak_panel', type=int, default=0,
    #                     help='Whether to run the environment with tweak panel')

    # args = parser.parse_args()

    env_name = 'PantsPushPPP'
    env_kwargs = env_arg_dict[env_name]

    # # Generate and save the initial states for running this environment for the first time
    # env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = 1
    env_kwargs['render'] = True
    env_kwargs['headless'] = 0
    
    env_kwargs['action_mode'] = 'pusher'
    env_kwargs['num_picker'] = 7
    env_kwargs['picker_radius'] = 0.01
    env_kwargs['pusher_length'] = 0.125
    env_kwargs['tweak_panel'] = 0

    env_kwargs['constraints'] = True

    # if not env_kwargs['use_cached_states']:
    #     print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS[env_name](**env_kwargs)

    print("particle radius: ", env.cloth_particle_radius)

    # --- Currently have to reset the environment which call pyflex.init() --- #
    # env.reset()
    # env.init_pusher([0,-0.01,0,0])

    # frames = [env.get_image(720, 720)]

    # flatten_area = env._set_to_flatten()
    pyflex.step()
    
    return env



def main():
    # --------- 1. create environment --------#
    env = create_env()

    # push_pose = generate_pusher_poses(env)[0]

    # push_poses = env.generate_pusher_poses()
    # pp = push_poses[7]

    # particle_pos = env._get_flat_pos()
    # # pusher_pos = np.array([*particle_pos[pp[0]], pp[1]])
    # pusher_pos = [-0.18749998229387188, 0.001, 0.27063556833560654, 0]

    
    # # print(pusher_pos)
    # env.init_pusher(pusher_pos)

    # pyflex.draw_rect(*[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0])
    

    while True:
        pyflex.step()


if __name__ == '__main__':
    main()