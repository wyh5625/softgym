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
from data_structure import MultiLayerTransitionGraph
import cv2




def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothPushPPP')
    parser.add_argument('--headless', type=int, default=0,
                        help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1,
                        help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str,
                        default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720,
                        help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0,
                        help='If to test the depth rendering by showing it')
    parser.add_argument('--tweak_panel', type=int, default=0,
                        help='Whether to run the environment with tweak panel')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['action_mode'] = 'pusher'
    env_kwargs['num_picker'] = 7
    env_kwargs['picker_radius'] = 0.01
    env_kwargs['pusher_length'] = 0.125
    env_kwargs['tweak_panel'] = args.tweak_panel

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    env.reset()
    # env.init_pusher([0,0.01,0,0])

    frames = [env.get_image(args.img_size, args.img_size)]

    flatten_area = env._set_to_flatten()

    # env.transform_cloth(np.array([0, 0, -0.5]), np.deg2rad(-29), np.array([0,0,0]))
    env.transform_cloth(np.array([-0.3, 0, 0.4]), np.deg2rad(-112), np.array([0,0,0]))

    pyflex.step()
    

    # selectBestAction(env)
    return env





if __name__ == '__main__':

    env = main()

    table = [2.0, 2.0, (0,0)] # table width, table height, table center
    env.set_working_area(*table)

    # flat_state_dict = env.get_state()

    # env.set_state(flat_state_dict)

    frame = env.render(mode='rgb_array')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # save frame as image
    cv2.imwrite("frame.jpg", frame)