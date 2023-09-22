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
import RRT_Star3 as rrt_star

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
    env_kwargs['num_picker'] = 7
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
        # next_node = next_state.node
            
        if action is None:
            continue

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

    save_name = osp.join('./data/', "ML_{}.gif".format("rrt_star"))
    env.end_record(save_name)


    pos = env.get_pusher_pos()
    push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    env.push(np.array([push_x, -0.01, push_y, push_ori, 0]))

    pyflex.step()

def test_sample_dataset(env):
    

    engine = create_engine('sqlite:///sample.db', echo=True)

    # load the Pandas DataFrame file
    df = pd.read_csv('deform_data_small_step.csv')

    # write the data to the SQL database
    df.to_sql('deform_data_small_step', engine, if_exists='replace', index=False)



    # query = """
    #     SELECT t1.*
    #     FROM action_sample t1
    #     INNER JOIN (
    #     SELECT push_x, push_y, push_ori, trans_x, trans_y, MIN(deformation) AS min_deformation
    #     FROM action_sample
    #     GROUP BY push_x, push_y, push_ori, trans_x, trans_y
    #     ) t2
    #     ON t1.push_x = t2.push_x
    #     AND t1.push_y = t2.push_y
    #     AND t1.push_ori = t2.push_ori
    #     AND t1.trans_x = t2.trans_x
    #     AND t1.trans_y = t2.trans_y
    #     AND t1.deformation = t2.min_deformation;
    # """

    query = """
        SELECT *
        FROM deform_data_small_step
    """

    query = text(query)
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
        # print(result)
    result_df = pd.DataFrame(result, columns=['push_x', 'push_y', 'push_ori', 'rot', 'trans_x', 'trans_y', 'deformation', 'result_points'])
    # print(result_df)

    # get the state of the flat configuration for discovering if the particle pos/vel is in nan state
    flat_state_dict = env.get_state()
    env.set_state(flat_state_dict)

    # get initial flat position of corners
    env.get_current_corner_pos()
    print("initial corner pos: ", env.cornerPos)
    init_corners = env.cornerPos[:]

    for _, row in result_df.iterrows():
        env.set_state(flat_state_dict)
        # do test sample
        push_pose_init = np.array([row['push_x'], 0.01, row['push_y'], row['push_ori']])
        end_pos = np.array([2*row['push_x'] + row['trans_x'], 0.01, 2*row['push_y']+row['trans_y'], row['push_ori'] + row['rot']])
        action = [*end_pos, 1.0]
        error = env.test_sample(push_pose_init, action, record=False,
                                            img_size=720, save_video_dir='./data/')
        
        parsed_points = ast.literal_eval(row['result_points'])

        env.get_current_corner_pos()
        ori_change = env.get_current_ori_change2(init_corners, env.cornerPos)
        # print("ori_change: ", ori_change)
        # print("center pos(sim): ", np.mean(env.cornerPos[:,[0,2]], axis=0))
        # print("center pos(db): ", np.mean(parsed_points, axis=0))
        print("db points", parsed_points)
        print("sim points", env.cornerPos[:,[0,2]])

        # calculate_center_orientation3
        # print("error: ", error)

def push_pose_test():
    # --------- 1. create environment --------#
    env = create_env()
    
    table = [0.5, 0.5, (0,0)] # table width, table height, table center


    # --------- 2. define task --------#
    start_config = (0, 0, 0)
    target_config = (-0.5, -0.5, 60)
    env.set_target_corner([target_config[0], target_config[1]], np.deg2rad(target_config[2]))

    pyflex.step()

    # 1. define a pushing action
    # 1.1 get the pushing pose
    
    # get the poses of corners
    env.get_current_corner_pos()
    contact_pose_start = env.get_corner_side_contact_pose(env.cornerPos)

    contact_pose_target = env.get_corner_side_contact_pose(env.target_cornersPos)


    cnr_id = 0
    side_id = 0

    
    pusher_pose = [*contact_pose_start[cnr_id][2*side_id], env.dir2ori(contact_pose_start[cnr_id][2*side_id+1])]
    env.set_pusher(np.array([pusher_pose[0], 0.01, pusher_pose[1], pusher_pose[2]]))

    pusher_pose = [*contact_pose_target[cnr_id][2*side_id], env.dir2ori(contact_pose_target[cnr_id][2*side_id+1])]
    end_pos = [pusher_pose[0], 0.01, pusher_pose[1], pusher_pose[2]]
    push_action = [*end_pos, 1]

    env.push(push_action)
    pyflex.step()



    # 2. execute the pushing action

    # execute_path(env, path)

def p2p_repositioning(env, epsilon = 0.01, no_penetration = True, middle_state = None):

    # get corner pos
    env.get_current_corner_pos()
    c_start = env.cornerPos

    c_target = env.target_cornersPos
    if middle_state is not None:
        # print("use middle state: ", middle_state)
        c_target = middle_state

    contact_pose_target = env.get_corner_side_contact_pose(c_target)

    while D(c_start, c_target) > epsilon:
        print("new_state: ", c_start)
        contact_pose_start = env.get_corner_side_contact_pose(c_start)
        # optimal contact pose id
        cnr_id = -1
        side_id = -1

        d_max = 0

        # loop corners
        for i in range(len(contact_pose_start)):
            # loop sides
            for j in [0,1]:
                x_dir = contact_pose_start[i][2*j+1]
                y_dir = env.rotate_vector(x_dir, np.deg2rad(90))


                delta_x, delta_y, delta_theta = *(contact_pose_target[i][2*j] - contact_pose_start[i][2*j]), contact_pose_target[i][2*j+1] - contact_pose_start[i][2*j+1]

                move_dir = [delta_x, delta_y]
                cos_theta = np.dot(move_dir, y_dir) / (np.linalg.norm(move_dir) * np.linalg.norm(y_dir))
            
                if no_penetration and cos_theta < 0:
                    continue

                if np.linalg.norm([delta_x, delta_y]) > d_max:
                    cnr_id = i
                    side_id = j
                    d_max = np.linalg.norm([delta_x, delta_y])

        if cnr_id == -1:
            print("no valid contact pose found")
            break

        # construct push action
        pusher_pose = [*contact_pose_start[cnr_id][2*side_id], env.dir2ori(contact_pose_start[cnr_id][2*side_id+1])]
        env.set_pusher(np.array([pusher_pose[0], 0.02, pusher_pose[1], pusher_pose[2]]))

        pusher_pose = [*contact_pose_target[cnr_id][2*side_id], env.dir2ori(contact_pose_target[cnr_id][2*side_id+1])]
        end_pos = [pusher_pose[0], 0.02, pusher_pose[1], pusher_pose[2]]
        push_action = [*end_pos, 1]

        env.push(push_action)

        # execute push action
        for i in range(5):
            pyflex.step()


        # update state
        env.get_current_corner_pos()
        c_start = env.cornerPos


# average distance between every pair of corresponding corners
def D(c_start, c_target):
    c_start_2d = c_start[:, [0,2]]
    c_target_2d = c_target[:, [0,2]]
    dist = 0
    for i in range(c_start.shape[0]):
        dist += np.linalg.norm(c_start_2d[i] - c_target_2d[i])
    return dist / c_start.shape[0]

def p2p_repositioning_test():
    start_config = (0, 0, 0)
    target_config = (0.5, -0.5, 60)

    env = create_env()
    env.set_target_corner([target_config[0], target_config[1]], np.deg2rad(target_config[2]))

    c_target = env.target_cornersPos

    pyflex.step()

    p2p_repositioning(env)


def main(use_ompl=False):
    # --------- 1. create environment --------#
    env = create_env()
    
    table = [0.5, 0.5, (0,0)] # table width, table height, table center


    # --------- 2. define task --------#
    start_config = (0, 0, 0)
    target_config = (-0.0, -0.0, 180)
    env.set_target_corner([target_config[0], target_config[1]], np.deg2rad(target_config[2]))
    # env.set_working_area(*table)
    # target_config = (0.2, 0.8, 170)
    # target_config = (0.6, 0.8, 80)
    # target_config = (0.6, -0.2, 180)
    # target_config = (0.5, -0.7, 90)
    # target_config = (-0.5, 0.8, 30)
    # target_config = (-0.5, 0.5, 120)
    pyflex.step()

    

    if not use_ompl:

        mltg = MultiLayerTransitionGraph()
        # --------- 3. create planner --------#
    
        mltg.init_sampled_actions('deform_data_small_step.csv')

        # --------- 4. plan path for the task --------#
        # path = mltg.plan(start_config, target_config)
        # if path is None:
        #     print("no path found")
        #     exit(0)
        # mltg.visualize_path(path, mltg.width, mltg.height)
        samples_db = pd.read_csv('deform_data_small_step.csv')
        samples_db, _ = process_dataframe(samples_db, push_idx=0)


        start_time = time.monotonic()

        path = mltg.plan(start_config, target_config)

        end_time = time.monotonic()
        print(f"Plan runtime: {end_time - start_time:.6f} seconds")

        mltg.visualize_path_on_graph(path, mltg)
    else:
        path = rrt_star.plan(start_config, target_config)
        # path = cp.plan(start_config, target_config)
        # print(path)
        rrt_star.visualize_path(path, 20, 20)
        # cp.visualize_path(path)

    execute_path(env, path)

    # --------- 5. execute the path --------#
    # path: [(node, action)...]
    # execute_path(env, path)

def generate_graph():
    # --------- 1. create environment --------#
    env = create_env()
    # table = [2.0, 2.0, (0,0)] # table width, table height, table center


    # --------- 2. define task --------#
    start_config = (0, 0, 0)
    target_config = (-0.0, 0.0, -180)
    # env.set_target_corner([target_config[0], target_config[1]], np.deg2rad(target_config[2]))


    # --------- 3. create planner --------#
    mltg = MultiLayerTransitionGraph()
    # mltg.init_sampled_actions('deform_data_small_step.csv')

    # --------- 4. plot reachable graph --------#
    samples_db = pd.read_csv('deform_data_small_step.csv')
    samples_db, push_pose = process_dataframe(samples_db, push_idx=0)

    # visualize push_pose
    # env.set_state(flat_state_dict)
    # env.set_pusher(np.array([push_pose[0], 0.01, push_pose[1], push_pose[2]]))
    # env.push(np.array([push_pose[0], 0.01, push_pose[1], push_pose[2], 0]))

    # frame = env.render(mode='rgb_array')
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Push pose", frame)
    # # save frame as image
    # cv2.imwrite("frame.jpg", frame)


    # for i in range(12):
    #     onecp_samples_db, push_pose = process_dataframe(samples_db, push_idx=i)
    #     mltg.generate_graph(start_config, onecp_samples_db, one_step=False)
    mltg.generate_graph(start_config, samples_db, one_step=False)


    # mltg.load_graph('multi_layer_transition_graph2.pkl')
    mltg.visualize_graph(mltg)
    


    # --------- 5. plan path for the task --------#
    # path = mltg.plan(start_config, target_config)
    # if path is None:
    #     print("no path found")
    #     exit(0)
    # mltg.visualize_path(path, mltg.width, mltg.height)


if __name__ == '__main__':
    # main(use_ompl=False)
    # push_pose_test()
    p2p_repositioning_test()
    # generate_graph()