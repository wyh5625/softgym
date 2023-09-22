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
import Cloth_control_planner as cp

# deform, trans, rot

# cost_weight = [[10, 0.5, 0.3],
#                 [10, 0.1, 0.8]]

cost_weight = [[10, 0.5, 0.3],
                [1, 1, 0.8]]


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

    env_name = 'ClothPushPPP'
    env_kwargs = env_arg_dict[env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = 1
    env_kwargs['render'] = True
    env_kwargs['headless'] = 0
    env_kwargs['action_mode'] = 'pusher'
    env_kwargs['num_picker'] = 1
    env_kwargs['picker_radius'] = 0.01
    env_kwargs['pusher_length'] = 0.125
    env_kwargs['tweak_panel'] = 0

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS[env_name](**env_kwargs)
    env.reset()
    env.init_pusher([0,-0.01,0,0])

    # frames = [env.get_image(720, 720)]

    # flatten_area = env._set_to_flatten()
    pyflex.step()
    
    return env


# push_pos = [push_x, push_y, push_ori], action = [trans_x, trans_y, rot]
def construct_push_action(push_pos, action, closed=True):
    init_pusher_pos = np.array([push_pos[0], 2*0.01, push_pos[1], push_pos[2]])
    end_pos = init_pusher_pos + np.array([action[0], 0, action[1], action[2]])
    push_action = [*end_pos, closed]
    push_action = np.array(push_action)
    return push_action

def execute_path(env, path):
    prev_layer = 0
        
    # env.set_state(flat_state_dict)
    env.init_pos()

    # Recording
    env.start_record()
    for (next_node, action) in path[1:]:
            
        pos = env.get_pusher_pos()
        push_x, push_y, push_ori = pos[0], pos[2], pos[3]

        # if not touched:
        push_x, push_y, push_ori = env.get_push_pos(action['push_x'], action['push_y'], action['push_ori'])
        env.set_pusher(np.array([push_x, 0.01, push_y, push_ori]))
        # touched = True

        # push
        env.get_current_corner_pos()
        ori_shift = env.get_current_ori_change2(env.cornerPos_init, env.cornerPos)
        shift_action = [*env.rotate_vector([action['trans_x'], action['trans_y']], np.deg2rad(prev_layer)), action['rot']]
        push_action = construct_push_action(np.array([push_x, push_y, push_ori]), shift_action, closed=True)
        env.push(push_action)

        # keep track of layer of last node, which is used to rotate the action
        prev_layer = next_node['layer']

    save_name = osp.join('./data/', str(target_config) + "_{}.gif".format("rrt"))
    env.end_record(save_name)


    pos = env.get_pusher_pos()
    push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    env.push(np.array([push_x, -0.01, push_y, push_ori, 0]))

    pyflex.step()

def push_sampling(env):
    # choose a contact point

    # init the point pusher on the contact point

    # sample a push direction

    # execute the push

    # record the result


if __name__ == '__main__':
    # --------- 1. create environment --------#
    env = create_env()
    # table = [2.0, 2.0, (0,0)] # table width, table height, table center


    # --------- 2. define task --------#
    start_config = (0, 0, 0)
    target_config = (-0.0, 0.0, -180)
    env.set_target_corner([target_config[0], target_config[1]], np.deg2rad(target_config[2]))
    # env.set_working_area(*table)
    # target_config = (0.2, 0.8, 170)
    # target_config = (0.6, 0.8, 80)
    # target_config = (0.6, -0.2, 180)
    # target_config = (0.5, -0.7, 90)
    # target_config = (-0.5, 0.8, 30)
    # target_config = (-0.5, 0.5, 120)


    # --------- 3. create planner --------#
    # mltg = MultiLayerTransitionGraph()
    # mltg.init_sampled_actions('deform_data_small_step.csv')


    # --------- 4. plan path for the task --------#
    # path = mltg.plan(start_config, target_config)
    # if path is None:
    #     print("no path found")
    #     exit(0)
    # mltg.visualize_path(path, mltg.width, mltg.height)

    path = cp.plan(start_config, target_config)
    print(path)
    cp.visualize_path(path)
 

    # --------- 5. execute the path --------#
    # path: [(node, action)...]
    execute_path(env, path)
