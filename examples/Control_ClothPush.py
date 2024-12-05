import os.path as osp
import argparse
import numpy as np
import os
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif, save_frame_as_image, make_grid
import pyflex
from matplotlib import pyplot as plt

import plotly.express as px
import pandas as pd

import time
import datetime
from tabulate import tabulate
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
import RRT_Star_MG as rrt_star
# from Control_ClothPush2 import p2p_repositioning

# deform, trans, rot

# cost_weight = [[10, 0.5, 0.3],
#                 [10, 0.1, 0.8]]

record_deform = True

position_tolerance = 0.01
push_height = 0.02
hang_height = 0.05

cost_weight = [[10, 0.5, 0.3],
                [1, 1, 0.8]]

times = []
time_eplapsed = datetime.timedelta(seconds=0)
time_prev = None

deformation_errors = []
distance_errors = []
D_errors = []

step = 0
steps = []
critical_points = []

state_frames = []

def clear():
    global times, time_eplapsed, time_prev, deformation_errors, distance_errors, D_errors, step, steps, critical_points, state_frames
    times = []
    time_eplapsed = datetime.timedelta(seconds=0)
    time_prev = None

    deformation_errors = []
    distance_errors = []
    D_errors = []

    step = 0
    steps = []
    critical_points = []

    state_frames = []


def create_env(env_name='PantsPushPPP'):

    # env_name = 'PantsPushPPP'
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

    env_kwargs['constraints'] = False

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS[env_name](**env_kwargs)
    # env.reset()
    
    env.init_pusher([0,0.1,0,0])

    # frames = [env.get_image(720, 720)]

    # flatten_area = env._set_to_flatten()
    pyflex.step()
    
    return env


# push_pos = [push_x, push_y, push_ori], action = [trans_x, trans_y, rot]
def construct_push_action(push_pos, action, closed=True):
    P = rrt_star.coordinate_to_matrix(push_pos[0], push_pos[1], push_pos[2])
    U = rrt_star.coordinate_to_matrix(action[0], action[1], action[2])
    P_ = np.dot(P, U)

    push_pose_ = rrt_star.matrix_to_coordinate(P_)

    # init_pusher_pos = np.array([push_pos[0], 2*0.01, push_pos[1], push_pos[2]])
    end_pos = np.array([push_pose_[0], push_height, push_pose_[1], push_pose_[2]])
    push_action = [*end_pos, closed]
    push_action = np.array(push_action)
    return push_action

def generate_push_actions(start_pos, target_pos, circular_rate=0.0, center_side=1, path_side=1, waypoint_size=1):
        # circular_rate is [0, 1], radius of path R = R_min / circular_rate(or cos(theta)), larger circular_rate, smaller R
        # center_side = -1/1, which means the center of circle is on the LHS/RHS of straight path
        # path_side = -1/1, which means the left/right part of circular path is chosen.
        waypoints = []
        actions = []
        # straight line
        if circular_rate == 0:
            diff = target_pos - start_pos
            if diff[3] > np.pi:
                diff[3] -= 2*np.pi
            elif diff[3] < -np.pi:
                diff[3] += 2*np.pi
            step = diff/waypoint_size

            for i in range(waypoint_size):
                waypoint = start_pos + (i+1)*step
                waypoints.append(waypoint)
                action = np.array([*waypoint, 1.0])
                actions.append(action)
        else:
            r_min = np.linalg.norm(target_pos - start_pos)/2
            r_path = r_min / circular_rate
            s = np.sqrt(1 - circular_rate**2)
            mid2center = s*r_path

            # find the rotation center
            mid_x = (start_pos + target_pos)/2
            move_l = target_pos - start_pos
            move_l_dir = move_l / np.linalg.norm(move_l)
            mid2center_dir = np.array(
                [move_l_dir[2], move_l_dir[1], -move_l_dir[0]])
            center = mid_x + mid2center * center_side * mid2center_dir

            # find the circular path
            angle = 2*(np.pi/2 - np.arccos(circular_rate))
            if center_side*path_side > 0:
                angle = 2*np.pi - angle
            if path_side == -1:
                angle = -angle
            step = angle/waypoint_size
            start_pos -= center
            print("Rotate angle:" + str(angle))
            # print(angle)
            for i in range(waypoint_size):
                new_pos = start_pos.copy()
                da = 0 + (i+1)*step
                new_pos[0] = (np.cos(da) * start_pos[0] -
                              np.sin(da) * start_pos[2])
                new_pos[2] = (np.sin(da) * start_pos[0] +
                              np.cos(da) * start_pos[2])
                waypoint = new_pos+center
                waypoints.append(waypoint)
                action = np.array([*waypoint, 1.0])
                actions.append(action)

        return actions

def execute_a_push_action(env, push_end, push_start, contact_pose, primary=False, with_standby=False):
    push_start += np.array([0, env.surface_height, 0, 0])
    push_end += np.array([0, env.surface_height, 0, 0])
    
    global step, time_eplapsed

    # define a standby pusher pose
    standby_pusher_pos = np.array([0, 0.5, 0, 0])

    # assume pusher is in standby pose
    # env.set_pusher(standby_pusher_pos)

    # append 0 to the end of contact pose
    
    # set speed
    env.action_tool.delta_move = 0.007

    # print("move to contact pose: ", push_start)
    # move to contact pose
    # push_start[1] -= 0.01
    if with_standby:
        push_start = np.append(push_start, 0)
        time_prev = datetime.datetime.now()
        env.push(push_start)
        time_eplapsed += datetime.datetime.now() - time_prev
    else:
        hang_over = push_start + np.array([0, hang_height, 0, 0])
        hang_over = np.append(hang_over, 0)

        time_prev = datetime.datetime.now()
        env.push(hang_over)
        push_start = np.append(push_start, 0)
        env.push(push_start)
        time_eplapsed += datetime.datetime.now() - time_prev
    
    # g_dt = 0.01s, the time interval of each step in the simulation, so 0.002/0.01 = 0.2 m/s
    env.action_tool.delta_move = 0.002
    

    # execute push action
    # divide push action into 5 steps
    actions = generate_push_actions(push_start[:4], push_end[:4], waypoint_size=10)


    # print("Start pushing")
    first_push = True
    # delta = (push_action[:4] - push_start) / 5
    for ac in actions:
        # print("ac: ", ac)
        if first_push:
            record_frame(env)
            first_push = False

        if record_deform:
            record_deformation(env)
        # pa_small = push_start + delta * i
        # pa_small = np.append(pa_small, 1)
        time_prev = datetime.datetime.now()
        env.push(ac)
        time_eplapsed += datetime.datetime.now() - time_prev

        

        if primary:
            inter_pose = env.align_inter_with_pusher(ac[[0,2,3]], contact_pose)
            env.set_inter_corner([inter_pose[0], inter_pose[1]], inter_pose[2], draw=False)

        if first_push:
            record_frame(env)
            first_push = False

        # record deformation w.r.t contact pose after each push action
        # record_deformation(env, ac[[0, 2, 3]], contact_pose)
        

            # calculate deformation for every mass point

            # calculate
        
    critical_points.append(step)

    # env.push(push_action)


    env.action_tool.delta_move = 0.007
    # move back to standby pose
    if with_standby:
        standby_pusher_pos = np.append(standby_pusher_pos, 0)
        
        time_prev = datetime.datetime.now()
        env.push(standby_pusher_pos)
        time_eplapsed += datetime.datetime.now() - time_prev
    else:
        hang_over = push_end + np.array([0, hang_height, 0, 0])
        # hang_over = np.append(hang_over, 0)
        hang_over = np.append(hang_over, 0)

        time_prev = datetime.datetime.now()
        env.push(hang_over)
        time_eplapsed += datetime.datetime.now() - time_prev


# deformation is relative to the rigid state defined by contact pose
def record_deformation(env):
    global step, time_eplapsed

    # s_rigid_from_cp = env.align_inter_with_pusher(pusher_pose, cp)

    # s_rigid_cp = env.set_inter_corner([next_node[0], next_node[1]], next_node[2], draw=draw_inter)

    # env.get_current_corner_pos()
    # ds = D(env.cornerPos, middle_state)

    # translation = np.array([s_rigid_from_cp[0], 0, s_rigid_from_cp[1]])

    # s_rigid_particle = env.transform_particles(env.init_pos, 
    #                         translation=translation, 
    #                         angle=s_rigid_from_cp[2], 
    #                         center=np.mean(env.cornerPos_init, axis=0)[:3], 
    #                         set_position=False)
    
    # env.set_inter_corner(translation[[0,2]], s_rigid_from_cp[2], draw=True)

    
        

    curr_pos = pyflex.get_positions().reshape(-1, 4)

    # if time_prev is None:
    #     time_prev = time.time()
    #     times.append(0)
    # else:
    #     times.append(time.time() - time_prev + times[-1])

    times.append(time_eplapsed.total_seconds())
    
    # CD(deformation) from s_rigid_particle to init_pos
    # EMD(distance) from s_rigid_particle to target_pos
    cd = env.CD(env.init_pos[:,[0,2]], curr_pos[:,[0,2]])
    emd = env.EMD(curr_pos[:,:3], env.t_pos[:,:3])
    avg_d = D(curr_pos[:,:3], env.t_pos[:,:3])

    # error = env.distance(s_rigid_particle, env.init_pos)

    # record time 
    # time_prev = time.time()
    
    # record deformation error
    deformation_errors.append(cd)
    distance_errors.append(emd)
    D_errors.append(avg_d)
        
    # record steps
    steps.append(step)
    step += 1

    
    # record distance
    # env.get_current_corner_pos()
    # err = D(env.cornerPos, env.target_cornersPos)
    # distance_errors.append(err)

def record_frame(env):
    # record frame
    # hide pusher
    old_pose = env.hide_pusher()

    env.set_pusher([0, -0.1, 0, 0])
    env.shoot_frame()
    # show pusher
    env.set_pusher(old_pose)

    state_frames.append(env.video_frames[-1])


def write_path(path, file_path):
    """Write a path to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(path, file)
        print(f"Path saved to '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while writing the path: {e}")

def load_path(file_path):
    """Load a path from a file."""
    try:
        with open(file_path, 'rb') as file:
            path = pickle.load(file)
            return path
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the path: {e}")
        return None

def angle_between_orientations(a, b):
    # Calculate the difference between angles
    diff = b - a

    # Adjust for the periodic nature of angles within [-π, π]
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    # Take the absolute value of the result to get the angle between orientations
    angle = np.abs(diff)
    return angle

def p2p_repositioning(env, epsilon = 0.01, no_penetration = True, middle_state = None, deformed_first = False):

    

    # get corner pos
    env.get_current_corner_pos()
    c_start_cnr = env.cornerPos

    
    
    if middle_state is None:
        # print("use middle state: ", middle_state)
        c_target_pos = env.t_pos
        c_target_cnr = env.target_cornersPos
    else:
        c_target_pos = env.transform_particles(env.init_pos, 
                                translation=np.array([middle_state[0], 0, middle_state[1]]), 
                                angle=middle_state[2], 
                                center=np.mean(env.cornerPos_init, axis=0)[:3], 
                                set_position=False)
        c_target_cnr = env.set_inter_corner([middle_state[0], middle_state[1]], middle_state[2], draw=False)

    contact_pose_target = env.get_corner_side_contact_pose(c_target_cnr)

    refine_count = 0

    # get current particles
    curr_pos = pyflex.get_positions().reshape(-1, 4)

    # print center of target particles
    print("center of target particles: ", np.mean(c_target_pos, axis=0)[:3])

    # print center of current particles
    print("center of current particles: ", np.mean(curr_pos, axis=0)[:3])



    while D(curr_pos, c_target_pos) > epsilon and refine_count < 10:
        # print center of target particles
        print("center of target particles: ", np.mean(c_target_pos, axis=0)[:3])

        # print center of current particles
        print("center of current particles: ", np.mean(curr_pos, axis=0)[:3])

        print("D distance: ", D(curr_pos, c_target_pos))
        refine_count += 1
        # print("new_state: ", c_start)
        contact_pose_start = env.get_corner_side_contact_pose(c_start_cnr)
        # optimal contact pose id
        cnr_id = -1
        side_id = -1

        d_max = 0

        # loop corners
        for i in range(len(contact_pose_start)):
            # loop sides
            for j in [0,1]:
                x_ori = contact_pose_start[i][2*j+1]
                # y_ori should be within [-pi, pi]
                y_ori = x_ori + np.deg2rad(90)

                # print("x_ori: ", x_ori, " y_ori: ", y_ori)
                if y_ori > np.pi:
                    y_ori -= 2*np.pi
                elif y_ori < -np.pi:
                    y_ori += 2*np.pi


                delta_x = contact_pose_target[i][2*j][0] - contact_pose_start[i][2*j][0]
                delta_y = contact_pose_target[i][2*j][1] - contact_pose_start[i][2*j][1]

                move_dir = [delta_x, delta_y]
                move_ori = env.dir2ori(move_dir)

                # cos_theta = np.dot(move_dir, y_dir) / (np.linalg.norm(move_dir) * np.linalg.norm(y_dir))
                
                # get angle between two orientations
                delta_ori = angle_between_orientations(y_ori, move_ori)
                # delta_ori = abs(y_ori - move_ori)%(2*np.pi)
                
                # print("corner: ", i, " side: ", j, " delta_ori: ", delta_ori, " delta_x: ", delta_x, " delta_y: ", delta_y, "y_ori: ", y_ori, " move_ori: ", move_ori)
                if no_penetration and delta_ori > np.pi/2:
                    continue

                if np.linalg.norm([delta_x, delta_y]) > d_max:
                    cnr_id = i
                    side_id = j
                    d_max = np.linalg.norm([delta_x, delta_y])

        if cnr_id == -1:
            print("no valid contact pose found")
            break

        # construct push action
        pusher_pose = [*contact_pose_start[cnr_id][2*side_id], contact_pose_start[cnr_id][2*side_id+1]]
        pusher_pose_start = np.array([pusher_pose[0], push_height, pusher_pose[1], pusher_pose[2]])
        
        # env.set_pusher(np.array([pusher_pose[0], 0.02, pusher_pose[1], pusher_pose[2]]))
        pusher_pose = [*contact_pose_target[cnr_id][2*side_id], contact_pose_target[cnr_id][2*side_id+1]]
        end_pos = [pusher_pose[0], push_height, pusher_pose[1], pusher_pose[2]]
        # push_action = [*end_pos, 1]

        print("pusher pose: ", pusher_pose_start)
        print("end_pos: ", end_pos)

        contact_pose = [*env.cornerCP_init[cnr_id][2*side_id], env.cornerCP_init[cnr_id][2*side_id+1]]
        execute_a_push_action(env, end_pos, pusher_pose_start, contact_pose)

        # record_frame(env)
        

        # env.push(push_action)

        # execute push action
        for i in range(5):
            pyflex.step()


        # update state
        env.get_current_corner_pos()
        c_start_cnr = env.cornerPos

        curr_pos = pyflex.get_positions().reshape(-1, 4)

    if refine_count >= 10:
        print("refinement failed")
        return False
    
    return True


# average distance between every pair of corresponding corners
def D(c_start, c_target):
    c_start_2d = c_start[:, [0,2]]
    c_target_2d = c_target[:, [0,2]]
    dist = 0
    for i in range(c_start.shape[0]):
        dist += np.linalg.norm(c_start_2d[i] - c_target_2d[i])
    return dist / c_start.shape[0]
    # return np.max(np.linalg.norm(c_start_2d - c_target_2d, axis=1))

def execute_path2(env, path, refinement=False, draw_inter=False):
    # prev_layer = 0
        
    # env.set_state(flat_state_dict)
    # env.init_pos()

    # wait until the cloth is stable
    # for i in range(100):
    #     pyflex.step()

    previous_node = path[0][0]
    u_path = path[1:]
    num_u = len(u_path)
    for i in range(num_u):
        (next_node, action) = u_path[i]
        # next_node = next_state.node
            
        if action is None:
            previous_node = next_node
            continue

        pos = env.get_pusher_pos()
        # push_x, push_y, push_ori = pos[0], pos[2], pos[3]

        # if not touched:
        
        push_x, push_y, push_ori = env.get_push_pos(action['push_x'], action['push_y'], action['push_ori'])
        push_pose_start = np.array([push_x, push_height, push_y, push_ori])
        # print("previous_node: ", previous_node)
        # push_pose_rigid
        push_pose_rigid = env.get_push_pos_rigid((previous_node[0], previous_node[1]), previous_node[2], (action['push_x'], action['push_y'], action['push_ori']))

        # push
        env.get_current_corner_pos()
        action_ = [action['trans_x'], action['trans_y'], action['rot']]
        next_push_pose_rigid = env.get_push_pos_rigid((next_node[0], next_node[1]), next_node[2], (action['push_x'], action['push_y'], action['push_ori']))
        # push_action = construct_push_action(push_pose_rigid, action_, closed=True)
        # print("push_action: ", push_action)

        # draw intermediate state

        inter_corners = env.set_inter_corner([next_node[0], next_node[1]], next_node[2], draw=draw_inter)
        # record_frame(env)

        
        # inter_corners = env.set_inter_corner([next_node[0], next_node[1]], next_node[2])
        # push_action = np.array([1,10,1,3.14,1])
        # primary pushing action
        # env.push(push_action)
        # env.push(push_action)
        push_pose_rigid = np.array([push_pose_rigid[0], push_height, push_pose_rigid[1], push_pose_rigid[2]])
        end_pos = np.array([next_push_pose_rigid[0], push_height, next_push_pose_rigid[1], next_push_pose_rigid[2]])
        # primary action
        execute_a_push_action(env, end_pos, push_pose_start, np.array([action['push_x'], action['push_y'], action['push_ori']]), primary=False)

        # record deformation error
        

        # record time
        # times.append(env.time_step)

        # if the next action's cp is different from the current one, do repositioning
        # check if cp change
        cp_change = False
        if i < num_u - 1:
            push_x_, push_y_, push_ori_ = u_path[i+1][1]['push_x'], u_path[i+1][1]['push_y'], u_path[i+1][1]['push_ori']
            if np.any(np.array([action['push_x'], action['push_y'], action['push_ori']]) != np.array([push_x_, push_y_, push_ori_])):
                print("cp change")
                cp_change = True
        elif i == num_u - 1:
            cp_change = True

        # refinement
        if refinement and cp_change:
            p2p_repositioning(env, epsilon=0.02, no_penetration=True, middle_state=next_node)
            

        # keep track of layer of last node, which is used to rotate the action
        # prev_layer = next_node[2]

        previous_node = next_node
    


    save_name = osp.join('./data/', "ML_{}.gif".format("rrt_star"))
    


    pos = env.get_pusher_pos()
    push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    # env.push(np.array([push_x, -0.01, push_y, push_ori, 0]))

    pyflex.step()

control_file = [
    'control_cloth_50cm.csv',
    'control_pants_50cm_1.csv'
]

demos = [
    'ClothPushPPP',
    'PantsPushPPP'
]

methods = ['rrt_star', 'rrt_star_refinement', 'p2p']


target_configs = [
    (-0.5, 0.0, np.deg2rad(0)),
    (0.0, -0.5, np.deg2rad(0)),
    (0.5, -0.5, np.deg2rad(-90)),
    (0.5, 0.5, np.deg2rad(135)),
    (0.0, 0.0, np.deg2rad(180)),
    (0.0, 0.0, np.deg2rad(70)),
    (0.2, 0.0, np.deg2rad(130)),
    (0.2, -0.5, np.deg2rad(48)),
    (-0.7, 0.1, np.deg2rad(137)),
    (-0.0, 0.5, np.deg2rad(78))

]

def read_data(file_path):
    data = {'time': [], 'error': []}
    with open(file_path, 'r') as file:
        for line in file:
            time, error = line.strip().split()
            data['time'].append(float(time))
            data['error'].append(float(error))
    return data

def compare(demo_id=0, task_id=0):

    target_config = target_configs[task_id]

    # create file path
    data_path = './data'
    demo_path = os.path.join(data_path, demos[demo_id])
    target_config_degree = (target_config[0], target_config[1], np.rad2deg(target_config[2]))
    task_name = "_".join(str(x) for x in target_config_degree)
    task_path = os.path.join(demo_path, task_name)

    # plt.figure(figsize=(10, 6))

    error_data = {}

    # read data from deformation_error.txt, target_error.txt, target_d_error.txt
    for method_id in range(len(methods)): 
        method = methods[method_id]
        error_data[methods[method_id]] = {}  # Initialize a sub-dictionary for each method

        if method_id == 2:
            method_path = os.path.join(task_path, method)
            deformation_error_path = os.path.join(method_path, 'deformation_error.png')
            target_error_path = os.path.join(method_path, 'target_error.png')
            target_error_d_path = os.path.join(method_path, 'target_d_error.png')
            gif_path = os.path.join(method_path, 'demo.gif')
            rrt_plot_path = os.path.join(method_path, 'rrt_path.png')
            deformation_error_txt = os.path.join(method_path, 'deformation_error.txt')
            target_error_txt = os.path.join(method_path, 'target_error.txt')
            target_error_d_txt = os.path.join(method_path, 'target_d_error.txt')
            frames_path = os.path.join(method_path, 'frames')
            time_path = os.path.join(method_path, 'time.txt')
            
            error_data[methods[method_id]]['deformation'] = read_data(deformation_error_txt)
            error_data[methods[method_id]]['target'] = read_data(target_error_txt)
            error_data[methods[method_id]]['target_d'] = read_data(target_error_d_txt)


        else:
            parent_method_path = os.path.join(task_path, "rrt")
            rrt_plot_path = os.path.join(parent_method_path, 'rrt_path.png')
            
            method_path = os.path.join(parent_method_path, method)
            deformation_error_path = os.path.join(method_path, 'deformation_error.png')
            target_error_path = os.path.join(method_path, 'target_error.png')
            target_error_d_path = os.path.join(method_path, 'target_d_error.png')
            gif_path = os.path.join(method_path, 'demo.gif')
            deformation_error_txt = os.path.join(method_path, 'deformation_error.txt')
            target_error_txt = os.path.join(method_path, 'target_error.txt')
            target_error_d_txt = os.path.join(method_path, 'target_d_error.txt')
            frames_path = os.path.join(method_path, 'frames')
            time_path = os.path.join(method_path, 'time.txt')

            error_data[methods[method_id]]['deformation'] = read_data(deformation_error_txt)
            error_data[methods[method_id]]['target'] = read_data(target_error_txt)
            error_data[methods[method_id]]['target_d'] = read_data(target_error_d_txt)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Loop through error types and plot data for each method
    for ax, error_type in zip(axes, ['deformation', 'target', 'target_d']):
        ax.set_title(f'{error_type.capitalize()} Error Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error')

        for method_name, method_data in error_data.items():
            data = method_data[error_type]
            ax.plot(data['time'], data['error'], label=method_name)

        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(task_path, 'error_comparison.png'))
    plt.show()


def generate_performance_table(data_path, demo, target_configs, methods):
    table = []

    for task_id, target_config in enumerate(target_configs):
        target_config_degree = (target_config[0], target_config[1], np.rad2deg(target_config[2]))
        task_name = f"Transform({target_config_degree[0]:.2f}, {target_config_degree[1]:.2f}, {target_config_degree[2]:.2f})"
        task_rows = []

        
        task_name_ = "_".join(str(x) for x in target_config_degree)

        for method_id, method in enumerate(methods):
            if method_id == 2:
                method_path = os.path.join(data_path, demo, task_name_, method)
            else:
                method_path = os.path.join(data_path, demo, task_name_, 'rrt', method)
            time_path = os.path.join(method_path, 'time.txt')
            success = 'No'
            time = 'None'
            max_deformation = None

            if os.path.exists(time_path):
                with open(time_path, 'r') as file:
                    time = float(file.readline().strip())
                    if time != -1:
                        success = 'Yes'
                    else:
                        time = 'None'

            deformation_path = os.path.join(method_path, 'deformation_error.txt')
            if os.path.exists(deformation_path):
                deformation_data = read_data(deformation_path)
                max_deformation = max(deformation_data['error'])

            task_rows.append((method, time, success, max_deformation))  # Convert the list to a tuple

        table.extend([(task_name if i == 0 else '',) + row for i, row in enumerate(task_rows)])

    headers = ['Task', 'Method', 'Time(s)', 'Success', 'Max Deformation']
    table_str = tabulate(table, headers=headers, tablefmt='grid')

    performance_table_path = os.path.join(data_path, demo, 'performance_table.txt')
    with open(performance_table_path, 'w') as file:
        file.write(table_str)

def generate_performance_table_xlsx(data_path, demo, target_configs, methods):
    table = []

    for task_id, target_config in enumerate(target_configs):
        target_config_degree = (target_config[0], target_config[1], np.rad2deg(target_config[2]))
        task_name = f"Transform({target_config_degree[0]:.2f}, {target_config_degree[1]:.2f}, {target_config_degree[2]:.2f})"
        task_rows = []

        task_name_ = "_".join(str(x) for x in target_config_degree)

        for method_id, method in enumerate(methods):
            if method_id == 2:
                method_path = os.path.join(data_path, demo, task_name_, method)
            else:
                method_path = os.path.join(data_path, demo, task_name_, 'rrt', method)
            time_path = os.path.join(method_path, 'time.txt')
            success = 'No'
            time = 'None'
            max_deformation = None

            if os.path.exists(time_path):
                with open(time_path, 'r') as file:
                    time = float(file.readline().strip())
                    if time != -1:
                        success = 'Yes'
                    else:
                        time = 'None'

            deformation_path = os.path.join(method_path, 'deformation_error.txt')
            if os.path.exists(deformation_path):
                deformation_data = read_data(deformation_path)
                max_deformation = max(deformation_data['error'])

            task_rows.append([method, time, success, max_deformation])

        table.append([task_name, task_rows])

    # Create a DataFrame from the table data
    data = []
    for task_name, task_rows in table:
        for row in task_rows:
            method, time, success, max_deformation = row
            data.append([task_name, method, time, success, max_deformation])
    
    columns = ['Task', 'Method', 'Time(s)', 'Success', 'Max Deformation']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to an Excel file
    excel_output_path = os.path.join(data_path, demo, 'performance_table.xlsx')
    df.to_excel(excel_output_path, index=False)
        

def sample_target_config(level='easy', num_samples=30, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), angle_range=(-np.pi, np.pi), angle_range_=None):    
    # sample target config and save them to a file in folder data/{level}
    # also set the target config in simulation and save the image
    # create file path
    data_path = './data'
    config_path = os.path.join(data_path, level)
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    
    target_configs = []

    if angle_range_ is not None:
        num_samples = num_samples // 2
        for i in range(num_samples):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            angle = np.random.uniform(angle_range[0], angle_range[1])
            target_configs.append((x, y, angle))
        for i in range(num_samples):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            angle = np.random.uniform(angle_range_[0], angle_range_[1])
            target_configs.append((x, y, angle))
    else:
        for i in range(num_samples):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            angle = np.random.uniform(angle_range[0], angle_range[1])
            target_configs.append((x, y, angle))
        

    # save target configs to file
    with open(os.path.join(config_path, 'target_configs.txt'), 'w') as file:
        for target_config in target_configs:
            file.write(f"{target_config[0]} {target_config[1]} {target_config[2]}\n")
    

# write a function to read target configs from a file and plot them in a grid
def plot_target_configs(level='easy', demo_id=0):
    env = create_env(demos[demo_id])
    env.set_pusher(np.array([0, -0.01, 0, 0]))
    env.start_record()
    
    # create file path
    data_path = './data'
    config_path = os.path.join(data_path, level)
    demo_path = os.path.join(config_path, demos[demo_id])
    frame_path = os.path.join(demo_path, 'frames')

    if not os.path.exists(demo_path):
        os.mkdir(demo_path)

    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    target_configs = []
    with open(os.path.join(config_path, 'target_configs.txt'), 'r') as file:
        for line in file:
            x, y, angle = line.strip().split()
            target_configs.append((float(x), float(y), float(angle)))
    
    # plot all target configs in grid, 10 per row，the number of row is determined by the number of target configs
    # num_rows = len(target_configs) // 10 + 1
    # fig, axes = plt.subplots(nrows=num_rows, ncols=10, figsize=(18, 6))

    frames = []
    for i, target_config in enumerate(target_configs):
        # ax = axes[i // 10, i % 10]

        start_config = (0, 0, 0)
        env.set_target_corner([target_config[0], target_config[1]], target_config[2])

        middle = [(start_config[0] + target_config[0])/2, (start_config[1] + target_config[1])/2]
        camera_pos = [middle[0], 2.3, middle[1]]
        env.set_camera_pos(camera_pos)
        env.shoot_frame()

        save_frame_as_image(env.video_frames[-1], os.path.join(frame_path, f"target_config_{i}.png"))
        frames.append(env.video_frames[-1])
        # ax.imshow(env.video_frames[-1])
        # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(demo_path, 'target_configs.png'))
    # plt.show()

    grid_img = make_grid(np.array(frames), ncol=10, padding=5, image_size=(256, 256))
    grid_path = os.path.join(demo_path, 'target_grid.png')
    save_frame_as_image(grid_img, grid_path)

# write a function to compare the performance of different methods
def compare_performance(level='easy', demo_id=0, tolerance=0.02, max_steps=10):
    # create file path
    data_path = './data'
    config_path = os.path.join(data_path, level)
    demo_path = os.path.join(config_path, demos[demo_id])

    # summarize the performance of each method in all tasks, the performance are avg time, success rate, avg deformation
    # read target configs from file
    target_configs = []
    with open(os.path.join(config_path, 'target_configs.txt'), 'r') as file:
        for line in file:
            x, y, angle = line.strip().split()
            target_configs.append((float(x), float(y), float(angle)))
    
    # for each target config, each method, read the performance data from file
    performance_data = {}
    for target_config in target_configs:
        approximated_target = (round(target_config[0], 2), round(target_config[1], 2), int(np.rad2deg(target_config[2])))
        # target_config_degree = (target_config[0], target_config[1], np.rad2deg(target_config[2]))
        task_name = "_".join(str(x) for x in approximated_target)

        for method_id, method in enumerate(methods):
            if method_id == 2:
                method_path = os.path.join(demo_path, task_name, method)
            else:
                method_path = os.path.join(demo_path, task_name, 'rrt', method)
            time_path = os.path.join(method_path, 'time.txt')
            success = 0
            time = 'None'
            actions = 0
            max_deformation = None

            # read push file, which has 3 columns: push idx, time after the push, error_after_push
            push_path = os.path.join(method_path, 'push.txt')
            push_data = []
            if os.path.exists(push_path):
                with open(push_path, 'r') as file:
                    for line in file:
                        push_data.append(line.strip().split())
            # find the push idx < 11 that has error_after_push < tolerance, and get the time after the push, this is a success task; if no such push idx, this is a failed task, and get the time after the max_steps
            if len(push_data) > 0:
                for i in range(len(push_data)):
                    if float(push_data[i][2]) < tolerance:
                        success = 1
                        time = float(push_data[i][1])
                        actions = i + 1
                        break
                
                if success == 0 or actions > 10:
                    time = float(push_data[9][1])
                    actions = 10
                    success == 0
            
            deformation_path = os.path.join(method_path, 'deformation_error.txt')

            # average the deformation from start to the actions-th push
            # retrive the time of the actions-th push, and get the deformation error of the actions-th push
            if os.path.exists(deformation_path):
                deformation_data = read_data(deformation_path)
                # get the index of deformation_data that has time == time
                idx = 0
                for i in range(len(deformation_data['time'])):
                    if deformation_data['time'][i] == time:
                        idx = i
                        break
                # average the deformation from 0 to the idx-th 'error' of deformation_data
                avg_deformation = np.mean(deformation_data['error'][:idx+1])

            else:
                print("deformation file not found: ", deformation_path)

            # deformation_path = os.path.join(method_path, 'deformation_error.txt')
            # if os.path.exists(deformation_path):
            #     deformation_data = read_data(deformation_path)
            #     max_deformation = max(deformation_data['error'])

            if task_name not in performance_data:
                performance_data[task_name] = {}

            # if method not in performance_data[task_name]:
            performance_data[task_name][method] = {'time': time, 'success': success, 'deformation': avg_deformation, 'actions': actions}
            

    # calculate the average performance of each method in all tasks: avg time, success rate, avg deformation and avg actions
    avg_performance = {}
    for task_name, task_data in performance_data.items():
        for method, method_data in task_data.items():
            if method not in avg_performance:
                avg_performance[method] = {'time': [], 'success': [], 'deformation': [], 'actions': []}
            avg_performance[method]['time'].append(method_data['time'])
            avg_performance[method]['success'].append(method_data['success'])
            avg_performance[method]['deformation'].append(method_data['deformation'])
            avg_performance[method]['actions'].append(method_data['actions'])
    for method, method_data in avg_performance.items():
        method_data['time'] = np.mean(method_data['time'])
        method_data['success'] = np.mean(method_data['success'])
        method_data['deformation'] = np.mean(method_data['deformation'])
        method_data['actions'] = np.mean(method_data['actions'])
    
    # print the average performance of each method
    # print("Average performance of each method in all tasks:")
    # for method, method_data in avg_performance.items():
    #     print(f"Method: {method}, Avg time: {method_data['time']}, Success rate: {method_data['success']}, Avg deformation: {method_data['deformation']}, Avg actions: {method_data['actions']}")
    return avg_performance



    


    

def experiment(env, level="easy", demo_id=0):
    data_path = os.path.join('./data', level)
    # for each demo_id, each target config, each method, run the algorithm and save the result
    # read target config from file
    target_configs = []
    with open(os.path.join(data_path, 'target_configs.txt'), 'r') as file:
        for line in file:
            x, y, angle = line.strip().split()
            target_configs.append((float(x), float(y), float(angle)))
    
    for target_config in target_configs[:13]:
        for method_id in range(len(methods)):
            print("method: ", methods[method_id])
            clear()
            env.set_init_pos()
            reuse_path = False
            if method_id == 1:
                reuse_path = True
            main(env, data_path, demo_id, target_config, method_id, reuse_path=reuse_path)

# def do_task(env, level="easy", demo_id=0, task_id=0, method_id=0, reuse_path=True):

def main(env, data_path="./data", demo_id=0, target_config=(0.5,0,0), method_id=0, reuse_path=True):

    method = methods[method_id]

    # --------- 2. define task --------#
    start_config = (0, -0.0, 0)
    # target_config = target_configs[task_id]
    env.set_start_state([0, -0.0], 0)
    env.set_target_corner([target_config[0], target_config[1]], target_config[2])

    middle = [(start_config[0] + target_config[0])/2, (start_config[1] + target_config[1])/2]
    camera_pos = [middle[0], 2.3, middle[1]]
    env.set_camera_pos(camera_pos)
    
    # env.set_working_area(*table)
    # target_config = (0.2, 0.8, 170)
    # target_config = (0.6, 0.8, 80)
    # target_config = (0.6, -0.2, 180)
    # target_config = (0.5, -0.7, 90)
    # target_config = (-0.5, 0.8, 30)
    # target_config = (-0.5, 0.5, 120)
    pyflex.step()

    env.set_pusher(np.array([0, -0.01, 0, 0]))
    for i in range(20):
        pyflex.step()


    # create file path
    demo_path = os.path.join(data_path, demos[demo_id])
    # in target_config_degree has target_config[0] and target_config[1] keep 2 decimals and target_config[2] convert to degree and keep integer
    approximated_target = (round(target_config[0], 2), round(target_config[1], 2), int(np.rad2deg(target_config[2])))
    # target_config_degree = (target_config[0], target_config[1], np.rad2deg(target_config[2]))
    task_name = "_".join(str(x) for x in approximated_target)
    task_path = os.path.join(demo_path, task_name)
    


    if method_id == 2:
        method_path = os.path.join(task_path, method)
        deformation_error_path = os.path.join(method_path, 'deformation_error.png')
        target_error_path = os.path.join(method_path, 'target_error.png')
        target_error_d_path = os.path.join(method_path, 'target_d_error.png')
        gif_path = os.path.join(method_path, 'demo.gif')
        rrt_plot_path = os.path.join(method_path, 'rrt_path.png')
        deformation_error_txt = os.path.join(method_path, 'deformation_error.txt')
        target_error_txt = os.path.join(method_path, 'target_error.txt')
        target_error_d_txt = os.path.join(method_path, 'target_d_error.txt')
        frames_path = os.path.join(method_path, 'frames')
        time_path = os.path.join(method_path, 'time.txt')
        push_path = os.path.join(method_path, 'push.txt')
    else:
        parent_method_path = os.path.join(task_path, "rrt")
        rrt_plot_path = os.path.join(parent_method_path, 'rrt_path.png')
        
        method_path = os.path.join(parent_method_path, method)
        deformation_error_path = os.path.join(method_path, 'deformation_error.png')
        target_error_path = os.path.join(method_path, 'target_error.png')
        target_error_d_path = os.path.join(method_path, 'target_d_error.png')
        gif_path = os.path.join(method_path, 'demo.gif')
        deformation_error_txt = os.path.join(method_path, 'deformation_error.txt')
        target_error_txt = os.path.join(method_path, 'target_error.txt')
        target_error_d_txt = os.path.join(method_path, 'target_d_error.txt')
        frames_path = os.path.join(method_path, 'frames')
        time_path = os.path.join(method_path, 'time.txt')
        push_path = os.path.join(method_path, 'push.txt')

    # Create directories if they don't exist
    os.makedirs(frames_path, exist_ok=True)


    env.start_record()
    # the initial frame
    

    # record frame
    # record_frame(env)

    # record initial data
    if record_deform:
        record_deformation(env)

    # env.adjust_controls()

    standby_pusher_pos = np.array([0, 0.5, 0, 0])
    env.set_pusher(standby_pusher_pos)

    task_success = False

    if method_id == 0 or method_id == 1:
        rrt_path_file = 'path_data.pkl'

        path = []

        if reuse_path and os.path.exists(os.path.join(parent_method_path, rrt_path_file)):
            print("Path file exists!")
            path = load_path(os.path.join(parent_method_path, rrt_path_file))
        else:
            path = rrt_star.plan(start_config, target_config, control_file[demo_id], save_plot=rrt_plot_path, repeat=10)
            # save the path to file
            if not path:
                print("No MG-RRT* path found!")
                return
            write_path(path, os.path.join(parent_method_path, rrt_path_file))
        # print("Path: ")
        # rrt_star.set_env(env_name)
        # rrt star init controls
        
        

        print("Path: ")
        print(path)


        if method_id == 0:
            execute_path2(env, path, refinement=False, draw_inter=True)
        elif method_id == 1:
            execute_path2(env, path, refinement=True, draw_inter=True)
            
        # path = cp.plan(start_config, target_config)
        # print(path)
        # rrt_star.visualize_path(path, 20, 20)
        # cp.visualize_path(path)

        # remove inter box
        env.remove_inter()

        task_success = p2p_repositioning(env, epsilon=position_tolerance, no_penetration=True, middle_state=None)
    
    elif method_id == 2:
        task_success = p2p_repositioning(env, epsilon=position_tolerance, no_penetration=False, middle_state=None)
    

    # draw target state again
    # env.set_inter_corner([target_config[0], target_config[1]], target_config[2])
    
    if record_deform:
        record_deformation(env)

    # release pusher
    env.set_pusher(np.array([0, -0.01, 0, 0]))

    # the final frame
    # env.shoot_frame()

    # record the final frame
    record_frame(env)

    # wait for cloth to settle
    for i in range(100):
        pyflex.step()

    # if env_name == 'TshirtPushPPP':
    #     env_name = 'PantsPushPPP'

    # ----------------- Plot deformation error -----------------
    if record_deform:
        # plot deformation_error and distance_error with steps in the same plot
        plt.figure()
        plt.plot(times, deformation_errors, label='deformation_error(Procrustes)')
        # Add markers and labels for critical time positions
        print("len of times: ", len(times))
        print("len of deformation_errors: ", len(deformation_errors))
        for i, pos in enumerate(critical_points):
            # find the index of the critical time in the times array
            # index = np.where(times == pos)[0][0]
            # how to get the deformation_error at the critical time
            plt.annotate(f'Push {i+1}', xy=(times[pos], deformation_errors[pos]), xytext=(10, 30),
                        textcoords='offset points', arrowprops=dict(arrowstyle="->"))
        # plt.plot(times, distance_errors, label='distance_error')
        plt.xlabel('times')
        plt.ylabel('deformation error')
        plt.legend()

        # Draw the 0 horizontal line for deformation error
        plt.axhline(y=0, color='r', linestyle='--', label='Position Tolerance')
        # plt.xticks(times)
        # Save the plot
        plt.savefig(deformation_error_path)
        

        # write error to txt file
        with open(deformation_error_txt, 'w') as f:
            for i in range(len(steps)):
                f.write('{} {}\n'.format(times[steps[i]], deformation_errors[i]))

        # ----------------- Plot distance error -----------------

        # Plot the second graph
        plt.figure()
        plt.plot(times, distance_errors, label='target_error(EMD)')
        # Add markers and labels for critical time positions
        for i, pos in enumerate(critical_points):
            plt.annotate(f'Push {i+1}', xy=(times[pos], distance_errors[pos]), xytext=(10, 30),
                        textcoords='offset points', arrowprops=dict(arrowstyle="->"))
        plt.xlabel('times')
        plt.ylabel('target error')
        plt.legend()

        # Draw the 0 horizontal line for distance error
        plt.axhline(y=position_tolerance, color='r', linestyle='--', label='Target Tolerance')

        # Save the second plot
        plt.savefig(target_error_path)

        # Show both plots
        # plt.show()
        
        # plt.savefig('./data/{}/{}/{}/error.png'.format(demos[demo_id], task_name, method))   
        plt.savefig(target_error_path)
        with open(target_error_txt, 'w') as f:
            for i in range(len(steps)):
                f.write('{} {}\n'.format(times[steps[i]], distance_errors[i]))

        # ----------------- Plot D error -----------------

        plt.figure()
        plt.plot(times, D_errors, label='target_error(D)')
        # Add markers and labels for critical time positions
        for i, pos in enumerate(critical_points):
            plt.annotate(f'Push {i+1}', xy=(times[pos], D_errors[pos]), xytext=(10, 30),
                        textcoords='offset points', arrowprops=dict(arrowstyle="->"))
        plt.xlabel('times')
        plt.ylabel('target D error')
        plt.legend()

        # Draw the 0 horizontal line for distance error
        plt.axhline(y=position_tolerance, color='r', linestyle='--', label='Target Tolerance')



        plt.savefig(target_error_d_path)
        with open(target_error_d_txt, 'w') as f:
            for i in range(len(steps)):
                f.write('{} {}\n'.format(times[steps[i]], D_errors[i]))

        # ----------------- End Plot -----------------

        # write time_elapsed to txt file
        with open(time_path, 'w') as f:
            # if task_success:
            f.write('{}'.format(time_eplapsed.total_seconds()))
            # else:
            #     f.write('{}'.format(-1))

        # write push to txt file, first column is the number of pushes, second column is the time, and the third is the error to target pose after push, start from 0(without push), the initial error and time is 0
        with open(push_path, 'w') as f:
            # f.write('{} {} {}\n'.format(0, 0, 0))
            for i, pos in enumerate(critical_points):
                f.write('{} {} {}\n'.format(i+1, times[pos], D_errors[pos]))

    # show the plot
    # plt.show()
    


    # save_name = osp.join('./data/{}/{}/{}/', "demo.gif".format(demos[demo_id], task_name, method))
    # save_name = osp.join('./data/', "MG_{}.gif".format("p2p"))
    env.end_record(gif_path)

    global state_frames

    print("frames: ", len(state_frames))

    # convert the state frames to a grid image, where state_frames is a list of images in the format of N*H*W
    state_frames = np.array(state_frames)
    
    # check if frames_path contains any files, delete them
    if os.path.exists(frames_path):
        for file in os.listdir(frames_path):
            file_path = os.path.join(frames_path, file)
            os.remove(file_path)
            # print("delete file: ", file_path)

    # save the state frames as png files
    for i, frame in enumerate(state_frames):
        file_path = os.path.join(frames_path, 'frame_{}.png'.format(i))
        save_frame_as_image(frame, file_path)
        print("save frame: ", file_path)


    grid_img = make_grid(state_frames, ncol=10, padding=5, image_size=(256, 256))
    # cv2.imshow('name', grid_img)
    # cv2.waitKey()

    grid_path = os.path.join(frames_path, 'frame_grid.png')
    save_frame_as_image(grid_img, grid_path)
    
    
        
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
    #     onecp_samples_db, push_pose = process_dataframe(samples_db, push_EMD=i)
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

def save_to_excel(table, filename='performance_data.xlsx'):
    # Create a Pandas DataFrame
    columns = ['Task', 'Shape', 'Method', 'Time(s)', 'Success', 'Avg Def', 'Actions']
    data_list = []

    # Iterate through your dictionary structure
    for method, shapes in table.items():
        for shape, levels in shapes.items():
            for level, data in levels.items():
                time = data.get('time', '--')
                success = data.get('success', '--')
                avg_def = data.get('deformation', '--')  # I'm assuming this is the key based on your code
                actions = data.get('actions', '--')
                row = [level.capitalize(), shape, method, time, success, avg_def, actions]
                data_list.append(row)

    df = pd.DataFrame(data_list, columns=columns)
    
    # Save to Excel
    df.to_excel(filename, index=False, engine='openpyxl')




if __name__ == '__main__':
    # ----------------- Sample target config ----------------- #
    # easy setting
    # level = 'easy'
    # num_samples = 30
    # x_range = (-0.5, 0.5)
    # y_range = (-0.5, 0.5)
    # angle_range = (-np.pi/2, np.pi/2)

    # median setting
    # level = 'median'
    # num_samples = 30
    # x_range = (-0.5, 0.5)
    # y_range = (-0.5, 0.5)
    # angle_range = (-np.pi, -np.pi/2)
    # angle_range_ = (np.pi/2, np.pi)

    level = 'hard'
    num_samples = 30
    x_range = (0, 0)
    y_range = (0, 0)
    angle_range = (-np.pi, -np.pi/2)
    angle_range_ = (np.pi/2, np.pi)

    
    
    # sample target config
    # sample_target_config('easy', num_samples, x_range, y_range, angle_range)
    # sample_target_config('median', num_samples, x_range, y_range, angle_range, angle_range_)
    # sample_target_config('hard', num_samples, x_range, y_range, angle_range, angle_range_)

    # plot target configs
    # plot_target_configs(level='easy', demo_id=1)
    # plot_target_configs(level='median', demo_id=0)
    # plot_target_configs(level='hard', demo_id=0)

    # ----------------- End sample target config ----------------- #

    # tasks = 4
    # demos = 2
    # methods = ['rrt_star', 'rrt_star_refine', 'p2p']
    # for di in range(demos):
    #     for ti in range(tasks):
    #         for mi in methods:
    #             if mi == 'rrt_star':
    #                 main(demo_id=di, task_id=ti, method_id=0, reuse_path=True)
    #             elif mi == 'rrt_star_refine':
    #                 main(demo_id=di, task_id=ti, method_id=1, reuse_path=True)
    #             elif mi == 'p2p':
    #                 main(demo_id=di, task_id=ti, method_id=2, reuse_path=True)
    #             pyflex.clean()

    # cloth
    # main(demo_id=0, task_id=0, method_id=0, reuse_path=True)
    # main(demo_id=0, task_id=0, method_id=1, reuse_path=True)
    # main(demo_id=0, task_id=0, method_id=2, reuse_path=True)
    # compare(demo_id=0, task_id=0)

    # main(demo_id=0, task_id=1, method_id=0, reuse_path=True)
    # main(demo_id=0, task_id=1, method_id=1, reuse_path=True)
    # main(demo_id=0, task_id=1, method_id=2, reuse_path=True)
    # compare(demo_id=0, task_id=1)

    # main(demo_id=0, task_id=2, method_id=0, reuse_path=False)
    # main(demo_id=0, task_id=2, method_id=1, reuse_path=True)
    # main(demo_id=0, task_id=2, method_id=2, reuse_path=True)
    # compare(demo_id=0, task_id=2)

    # main(demo_id=0, task_id=3, method_id=0, reuse_path=False)
    # main(demo_id=0, task_id=3, method_id=1, reuse_path=True)
    # main(demo_id=0, task_id=3, method_id=2, reuse_path=True)
    # compare(demo_id=0, task_id=3)

    # main(demo_id=0, task_id=4, method_id=0, reuse_path=True)
    # main(demo_id=0, task_id=4, method_id=1, reuse_path=True)

    # target_configs[4]
    demo_id = 1
    env = create_env(demos[demo_id])
    main(env, demo_id=1, target_config=(0, -0.0, np.deg2rad(180)), method_id=0, reuse_path=True)
    # compare(demo_id=0, task_id=4)

    # main(demo_id=0, task_id=5, method_id=0, reuse_path=True)
    # main(demo_id=0, task_id=6, method_id=0, reuse_path=False)
    # main(demo_id=0, task_id=7, method_id=0, reuse_path=True)
    # main(demo_id=0, task_id=8, method_id=0, reuse_path=True)
    # demo_id = 0
    # env = create_env(demos[demo_id])
    # for i in range(5):
    # clear()
    # env.set_init_pos()
    # main(env, demo_id=demo_id, task_id=9, method_id=0, reuse_path=True)

    # clear()
    # env.set_init_pos()
    # main(env, demo_id=0, task_id=8, method_id=2, reuse_path=True)

    # main(demo_id=0, task_id=9, method_id=2, reuse_path=True)
    # main(demo_id=0, task_id=6, method_id=1, reuse_path=True)

    # generate_performance_table('./data', demos[0], target_configs, methods)
    # generate_performance_table_xlsx('./data', demos[0], target_configs, methods)

    # pants
    # main(demo_id=1, task_id=0, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=0, method_id=1, reuse_path=True)
    # main(demo_id=1, task_id=0, method_id=2, reuse_path=True)
    # compare(demo_id=1, task_id=0)

    # main(demo_id=1, task_id=1, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=1, method_id=1, reuse_path=True)
    # main(demo_id=1, task_id=1, method_id=2, reuse_path=True)
    # compare(demo_id=1, task_id=1)

    # main(demo_id=1, task_id=2, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=2, method_id=1, reuse_path=True)
    # main(demo_id=1, task_id=2, method_id=2, reuse_path=True)
    # compare(demo_id=1, task_id=2)

    # main(demo_id=1, task_id=3, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=3, method_id=1, reuse_path=True)
    # main(demo_id=1, task_id=3, method_id=2, reuse_path=True)
    # compare(demo_id=1, task_id=3)

    # main(demo_id=1, task_id=4, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=4, method_id=1, reuse_path=True)
    # main(demo_id=1, task_id=4, method_id=2, reuse_path=True)
    # compare(demo_id=1, task_id=4)

    # main(demo_id=1, task_id=5, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=6, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=7, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=8, method_id=0, reuse_path=True)
    # main(demo_id=1, task_id=9, method_id=0, reuse_path=True)


    # Example usage
    # generate_performance_table('./data', demos[1], target_configs, methods)
    # generate_performance_table_xlsx('./data', demos[1], target_configs, methods)

    # demo_id = 1
    # env = create_env(demos[demo_id])
    # experiment(env, level="easy", demo_id=demo_id)
    # experiment(env, level="easy", demo_id=1)

    # experiment(env, level="median", demo_id=0)
    # experiment(env, level="median", demo_id=1)

    # experiment(env, level="hard", demo_id=0)
    # experiment(env, level="hard", demo_id=1)
    
    # compare_performance(level='easy', demo_id=0, tolerance=0.02, max_steps=10)
    # compare_performance(level='easy', demo_id=1, tolerance=0.02, max_steps=10)

    # compare_performance(level='median', demo_id=0, tolerance=0.02, max_steps=10)
    # compare_performance(level='median', demo_id=1, tolerance=0.02, max_steps=10)

    # compare_performance(level='hard', demo_id=0, tolerance=0.02, max_steps=10)
    # compare_performance(level='hard', demo_id=1, tolerance=0.02, max_steps=10)

    # summarize performance of each method on all tasks in all demos as a table
    # table = {}
    # for demo_id in range(len(demos)):
    #     for level in ['easy', 'median', 'hard']:
    #         performance = compare_performance(level=level, demo_id=demo_id, tolerance=0.02, max_steps=10)
    #         for method, method_data in performance.items():
    #             if method not in table:
    #                 table[method] = {}
    #             if demos[demo_id] not in table[method]:
    #                 table[method][demos[demo_id]] = {}
    #             table[method][demos[demo_id]][level] = method_data
    
    # # print the table, each row is a method, each column is a demo, and each child column of demo column is a metric
    # print("Performance table:")

    
    # Execute the function
    # save_to_excel(table)
    # print(table)

    