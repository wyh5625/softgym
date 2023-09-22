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
from softgym.utils.pyflex_utils import center_object
from scipy.spatial.transform import Rotation
from RRT_Star_MG import coordinate_to_matrix, matrix_to_coordinate


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
    env_kwargs['num_picker'] = 7
    env_kwargs['picker_radius'] = 0.01
    env_kwargs['pusher_length'] = 0.125
    env_kwargs['tweak_panel'] = 0

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS[env_name](**env_kwargs)
    # env.reset()
    env.init_pusher([0,-0.01,0,0])

    # frames = [env.get_image(720, 720)]

    # flatten_area = env._set_to_flatten()
    pyflex.step()
    
    return env



def sample(env, pusher_poses):
    # dir sampling
    dir_step = 15
    rot_step = 15
    move_len = 0.1
    # relative to the frame of the pusher
    dir_angles = np.deg2rad(np.arange(0, 181, dir_step))
    vectors = move_len * \
        np.stack((np.cos(dir_angles), np.sin(dir_angles)), axis=-1)
    rots = np.deg2rad(np.arange(-90, 91, rot_step))


    start_time = time.time()

    # vectors += init_pusher_pos[:3]
    # rots += init_pusher_pos[3]

    # create a multi-array list
    
    dimx, dimy = env.get_current_config()['ClothSize']
    
    particle_pos = env._get_flat_pos()
    # particle_pos = particle_pos.reshape((dimy, dimx,3))
    # pusher half length idx
    phi = int(0.5*env.action_tool.pusher_length/env.cloth_particle_radius)

    # 9 primary sample points
    Xs = [phi, int(dimx/2), dimx-phi]
    Ys = [phi, int(dimy/2), dimy-phi]


    XX, YY = np.meshgrid(Xs, Ys)
    XX = XX.flatten()
    YY = YY.flatten()
    deform_data = []


    # get the state of the flat configuration for discovering if the particle pos/vel is in nan state
    flat_state_dict = env.get_state()

    np.set_printoptions(precision=3, suppress=True, formatter={'separator': ','})


    for push_pose in pusher_poses:
        # print("particle pos: ", particle_pos[push_pose[0]])
        init_pusher_pos = np.array([*particle_pos[push_pose[0]], push_pose[1]])
        # init_pusher_pos = np.array([push_pose[0][0], 0.001, push_pose[0][1], push_pose[1]])
        print("pusher pose: ", init_pusher_pos)
        for i in range(len(vectors)):
            # find the proper pusher pose that has the normal direction 
            # with cross angle between moving direction which is smaller than 90 degree
            # center_pos = particle_pos[push_pose[0]]
            # center_pos[1] = 0.01
            # # print(center_pos)
            # pos_2d = center_pos[[0, 2]]
            # norm_v = env.get_normal_of_push_pose(pos_2d[0], pos_2d[1], push_pose[1])
            # move_v = np.array([vectors[i][0], vectors[i][1]])
            # cross_angle = np.arccos(np.dot(norm_v, move_v)/(np.linalg.norm(norm_v)*np.linalg.norm(move_v)))
            # if cross_angle < np.pi/2:
            print("push direction: ", vectors[i])
                # get a list of angles from -pi/2 to pi/2, step size is 15 degree
                # rots = np.arange(-np.pi/2, np.pi/2, np.pi/12)

                # find the rot with minimum unfollowed error
                # min_error = 999
                # min_rot = 0
            actions_all_rots = []
            for rot in rots:
                print("rot: ", rot)
                env.set_state(flat_state_dict)
                    # do test sample
                    # end_pos = init_pusher_pos + \
                    #     np.array([vectors[i][0], 0, vectors[i][1], rot])
                    # action = [*end_pos, 1.0]

                    # do test sample
                p = [init_pusher_pos[0], init_pusher_pos[2], init_pusher_pos[3]]
                u = [vectors[i][0], vectors[i][1], rot]
                M_p = coordinate_to_matrix(*p)
                M_u = coordinate_to_matrix(*u)

                M = np.matmul(M_p, M_u)
                p_end = matrix_to_coordinate(M)

                end_pos = np.array([p_end[0], 0.01, p_end[1], p_end[2]])
                action = [*end_pos, 1.0]

                    # push_pose_init = np.array([*center_pos, push_pose[1]])
                error = env.test_sample(init_pusher_pos, action, record=False,
                                            img_size=720, save_video_dir='./data/')
                    
                    # env.get_current_corner_pos()

                    # print(env.cornerPos)

                data_item = (p[0], p[1], p[2], u[0], u[1], u[2], error)
                    # deform_data.append(data_item)
                actions_all_rots.append(data_item)
                    # print(result_points_str)
                    
                    
                    # if error < min_error:
                    #     min_error = error
                    #     min_rot = rot
                # sort the actions_all_rots by error
            actions_all_rots.sort(key=lambda x: x[6])
                # append first two actions to deform_data
            deform_data.append(actions_all_rots[0])
            deform_data.append(actions_all_rots[1])
            deform_data.append(actions_all_rots[2])

    columns = ['push_x', 'push_y', 'push_ori', 'trans_x', 'trans_y', 'rot', 'deformation']
    df = pd.DataFrame(deform_data, columns=columns)
    df.to_csv('control_cloth_10cm.csv', index=False)

    # deform_data = np.array(deform_data)

    # np.save("deform_3x3_1.npy", deform_data)

    print("--- %s seconds ---" % (time.time() - start_time))




if __name__ == '__main__':


    env = create_env()

    pusher_poses = env.generate_pusher_poses()

    print("pusher poses: ", pusher_poses)

    sample(env, pusher_poses)