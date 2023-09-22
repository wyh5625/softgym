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

# deform, trans, rot

# cost_weight = [[10, 0.5, 0.3],
#                 [10, 0.1, 0.8]]

cost_weight = [[10, 0.5, 0.3],
                [1, 1, 0.8]]

# cost_weight = [[1, 1, 0],
#                 [1, 1, 0]]

# def round_half_up(n, decimals=0):
#     multiplier = 10 ** decimals
#     return math.floor(n*multiplier + 0.5) / multiplier

# def show_depth():
#     # render rgb and depth
#     img, depth = pyflex.render()
#     img = img.reshape((720, 720, 4))[::-1, :, :3]
#     depth = depth.reshape((720, 720))[::-1]
#     # get foreground mask
#     rgb, depth = pyflex.render_cloth()
#     depth = depth.reshape(720, 720)[::-1]
#     # mask = mask[:, :, 3]
#     # depth[mask == 0] = 0
#     # show rgb and depth(masked)
#     depth[depth > 5] = 0
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].imshow(img)
#     axes[1].imshow(depth)
#     plt.show()


def get_action_samples(filename):
    # engine = create_engine('sqlite:///sample.db', echo=True)

    # load the Pandas DataFrame file
    result_df = pd.read_csv(filename)

    return result_df

# def modify_action_samples(df):
#     df['trans_x'] += df['push_x']
#     df['trans_y'] += df['push_y']
#     return df

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


# def createActionDatabase(df, table_name, engine):
#     # import pandas as pd

#     # current_dir = osp.dirname(osp.abspath(__file__))
#     # # Load a file called "example.txt" in the current directory
#     # filename = osp.join(current_dir, "deform_3x3.npy")

#     # # Load the .npy file into a NumPy array
#     # arr = np.load(filename, allow_pickle=True)

#     # # Convert the NumPy array into a pandas DataFrame
#     # df = pd.DataFrame(list(arr))

#     # Create a SQLite database engine
#     df.to_sql(table_name, con=engine)
    

# push_pos = [push_x, push_y, push_ori], action = [trans_x, trans_y, rot]
def construct_push_action(push_pos, action, closed=True):
    init_pusher_pos = np.array([push_pos[0], 2*0.01, push_pos[1], push_pos[2]])
    end_pos = init_pusher_pos + np.array([action[0], 0, action[1], action[2]])
    push_action = [*end_pos, closed]
    push_action = np.array(push_action)
    return push_action

# def selectStartAction(env, engine):
#     # createActionDatabase()
#     # print("query start action ---")
#     # (ChatGPT) This query will partition the table by the push_x and push_y columns, 
#     # and for each partition, it will assign a row number to each row based 
#     # on the deformation column, ordered in ascending order. Then, the outer 
#     # query will select only the rows where the row number is less than or 
#     # equal to 10, giving you the 10 smallest values of deformation in each group.
#     query = """
#         SELECT *
#         FROM (
#         SELECT *, ROW_NUMBER() OVER (PARTITION BY push_x, push_y ORDER BY deformation) AS rn
#         FROM deform_3x3
#         ) sub
#         WHERE rn <= 100
#     """
#     query = text(query)
#     with engine.connect() as conn:
#         result = conn.execute(query).fetchall()
#         # print(result)

#     # set the display.max_columns option to None

#     result_df = pd.DataFrame(result, columns=['index', 'push_x', 'push_y', 'push_ori', 'rot', 'trans_x', 'trans_y', 'deformation', 'rn'])

  

#     # get the action with the minimum cost
#     min_cost = 999
#     min_action = None
#     min_pos = None
#     deformation = 0
#     for index, row in result_df.iterrows():
#         # print(row['push_x'], row['push_y'], row['deformation'])
#         push_pos = [row['push_x'], row['push_y'], row['push_ori']]
#         action = [row['trans_x'], row['trans_y'], row['rot']]
#         cost, _ ,_, _ = env.cost(action, push_pos, row['deformation'], *cost_weight[0])
#         if cost < min_cost and np.round(row['push_x'] + row['push_y'],2) != 0:
#             min_cost = cost
#             min_action = action
#             min_pos = push_pos
#             deformation = row['deformation']

#     min_pos = np.array(min_pos)


#     return min_pos, min_action, deformation, result_df

# def selectActionOnPos(push_pos, cur_pos, engine, result_df=None):
#     # print("push_pose for query: ")
#     # print(push_pos)
#     # print("push_pos -----------------", push_pos)
#     # print(round_half_up(push_pos[0],2), round_half_up(push_pos[1],2), round_half_up(push_pos[2],2))
#     if result_df is None:
#         query = """
#             SELECT trans_x, trans_y, rot, deformation
#             FROM deform_3x3
#             WHERE ROUND(push_x,2)={} and ROUND(push_y,2)={} and ROUND(push_ori,2)={}
#         """.format(round_half_up(push_pos[0],2), round_half_up(push_pos[1],2), round_half_up(push_pos[2],2))
#         query = text(query)
#         with engine.connect() as conn:
#             result = conn.execute(query).fetchall()

#         result_df = pd.DataFrame(result, columns=['trans_x', 'trans_y', 'rot', 'deformation'])
#     # print("result of query")
#     # print(result_df)

#     # get the action with the minimum cost
#     min_cost = 999
#     min_action = None
#     min_index = 0
#     deformation = 0
#     dist_min = 0

#     env.get_current_corner_pos()
#     ori_shift = env.get_current_ori_change2(env.cornerPos_init, env.cornerPos)
#     min_center = None

#     for index, row in result_df.iterrows():
#         # print(row['push_x'], row['push_y'], row['deformation'])
#         # push_pos = [row['push_x'], row['push_y'], row['push_ori']]
#         action = [row['trans_x'], row['trans_y'], row['rot']]
#         shift_action = [*env.rotate_vector(action[:2], ori_shift), action[2]]
#         # print(action)
#         cost, deform, dist, new_center = env.cost(shift_action, cur_pos, row['deformation'], *cost_weight[1])

#         # print idx, move_dir, appro(center), cost
#         # print(index, action[:2], approach, cost)

#         # print("dist_cost: ", dist_cost)
#         if cost < min_cost:
#             min_cost = cost
#             min_action = shift_action
#             min_index = index
#             dist_min = dist
#             deformation = deform
#             min_center = new_center
#             # dist_cost = dist
#             # deformation_cost = cost - dist_cost
    
#     # print("best action ----------------")
#     # print(min_index, min_action[:2])
#     # print(min_pos, min_action)
#     print("Best deform: ", deformation)
#     print("Best dist: ", dist_min)
#     print("Best center: ", min_center)
#     # print("deformation: ", deformation_cost)
#     # print("dist cost: ", dist_cost)

#     # print("action")
#     # print(action)

#     # print("new pos:")
#     # print(new_pos)

#     return min_action, result_df, deformation


# def selectActionFromDB(cur_pos, engine, result_df=None):
#     # print("push_pose for query: ")
#     # print(push_pos)
#     # print("push_pos -----------------", push_pos)
#     # print(round_half_up(push_pos[0],2), round_half_up(push_pos[1],2), round_half_up(push_pos[2],2))
#     if result_df is None:
#         query = """
#         SELECT *
#         FROM (
#         SELECT *, ROW_NUMBER() OVER (PARTITION BY push_x, push_y ORDER BY deformation) AS rn
#         FROM deform_3x3
#         ) sub
#         WHERE rn <= 100
#         """
#         query = text(query)
#         with engine.connect() as conn:
#             result = conn.execute(query).fetchall()

#         result_df = pd.DataFrame(result, columns=['index', 'push_x', 'push_y', 'push_ori', 'rot', 'trans_x', 'trans_y', 'deformation', 'rn'])
#     # print("result of query")
#     # print(result_df)

#     # get the action with the minimum cost
#     min_cost = 999
#     min_action = None
#     min_index = 0
#     deformation = 0
#     dist_min = 0

#     env.get_current_corner_pos()
#     ori_shift = env.get_current_ori_change2(env.cornerPos_init, env.cornerPos)
#     min_center = None

#     for index, row in result_df.iterrows():
#         # print(row['push_x'], row['push_y'], row['deformation'])
#         # push_pos = [row['push_x'], row['push_y'], row['push_ori']]
#         action = [row['trans_x'], row['trans_y'], row['rot']]
#         push_pos = [row['push_x'], row['push_y'], row['push_ori']]

#         # only rotate the translation vector in action
#         shift_action = [*env.rotate_vector(action[:2], ori_shift), action[2]]
#         # print(action)
#         cost, deform, dist, new_center = env.cost(shift_action, cur_pos, row['deformation'], *cost_weight[1])

#         # print idx, move_dir, appro(center), cost
#         # print(index, action[:2], approach, cost)

#         # print("dist_cost: ", dist_cost)
#         if cost < min_cost:
#             min_cost = cost

#             # merge push_pos and shift_action into min_action
#             min_action = [*push_pos, *shift_action]

#             # min_action = shift_action
#             min_index = index
#             dist_min = dist
#             deformation = deform
#             min_center = new_center
#             # dist_cost = dist
#             # deformation_cost = cost - dist_cost
    
#     # print("best action ----------------")
#     # print(min_index, min_action[:2])
#     # print(min_pos, min_action)
#     print("Best deform: ", deformation)
#     print("Best dist: ", dist_min)
#     print("Best center: ", min_center)
#     # print("deformation: ", deformation_cost)
#     # print("dist cost: ", dist_cost)

#     # print("action")
#     # print(action)

#     # print("new pos:")
#     # print(new_pos)

#     return min_action, result_df, deformation




def find_path_and_execute(env):
    pass

# def draw_target_rectangle(t_x, t_y, t_ori):
#     height = length_of_cloth
#     rot = 90
#     axis = np.array([0, -1, 0])
#     rotation = Rotation.from_rotvec(rot * axis)
#     # Convert the Rotation object to a quaternion
#     quaternion = rotation.as_quat()

#     pyflex.add_box(np.array([width/2, 0.002, height/2]), [0,0,0], quaternion, 1)

#     # four middle points of four sides for the target rectangle
#     # t_x = 0.0
#     # t_y = 0.0
#     # t_ori = 0.0
#     t_corners = np.array([[t_x - width/2, t_y - height/2],
#     pass




if __name__ == '__main__':
    print("create engine!")
    engine = create_engine('sqlite:///sample.db', echo=True)


    env = create_env()
    table = [2.0, 2.0, (0,0)] # table width, table height, table center
    # env.set_working_area(*table)

    # target_config = (0.2, 0.8, 170)
    # target_config = (0.6, 0.8, 80)
    # target_config = (0.6, -0.2, 180)
    # target_config = (0.5, -0.7, 90)
    # target_config = (-0.5, 0.8, 30)
    # target_config = (-0.5, 0.5, 120)
    

    # pusher_poses = generate_pusher_poses(env)


    # sample(env, pusher_poses)

    # state_transition_sampling(env, pusher_poses, 2.0, 2.0, sample_step=0.2)

    # test_sample_dataset(env)

    flat_state_dict = env.get_state()
    # env._set_to_flatten()
    # env.reset()

    # pos = env.get_pusher_pos()
    # push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    # env.push(np.array([push_x, -0.11, push_y, push_ori, 0]))

    # pyflex.step()
    
    # center_object()
    # # camera_param = np.array([*[0.0, 2.0, 0], *[0.0, -90 / 180. * np.pi, 0.0], env.camera_width, env.camera_height])
    # camera_param = dict()
    # camera_param['pos'] = [0.0, 1.0, 0]
    # camera_param['angle'] = [0.0, -90 / 180. * np.pi, 0.0]
    # camera_param['width'] = env.camera_width
    # camera_param['height'] = env.camera_height
    # env.update_camera("zoom_in", camera_param)

    # frame = env.render(mode='rgb_array')
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # # save frame as image
    # cv2.imwrite("last_frame.jpg", frame)

    # print(samples_db)


    # create MultiLayer
    mltg = MultiLayerTransitionGraph()

    # init the graph by setting the start config
    

    # filename = "multi_layer_transition_graph2.pkl"
    # check_file = False
    # if os.path.exists(filename) and check_file:
    #     print("load graph from file")
    #     mltg.load_graph(filename)
    # else:
    #     print("generate graph")
    #     samples_db = get_action_samples('deform_data_small_step.csv')
    #     samples_db, push_pose = process_dataframe(samples_db, push_idx=1)

    #     # visualize push_pose
    #     # env.set_state(flat_state_dict)
    #     # env.set_pusher(np.array([push_pose[0], 0.01, push_pose[1], push_pose[2]]))
    #     # env.push(np.array([push_pose[0], 0.01, push_pose[1], push_pose[2], 0]))

    #     # frame = env.render(mode='rgb_array')
    #     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     # cv2.imshow("Push pose", frame)
    #     # # save frame as image
    #     # cv2.imwrite("frame.jpg", frame)

    #     # show frame
        
    #     start_config = (0, 0, 0)
    #     print("generate graph of push pose: ", push_pose)
    #     mltg.generate_graph(start_config, samples_db, one_step=False)
    #     mltg.save_graph("multi_layer_transition_graph2.pkl")

    # print("graph initialized!")

    print("visualize graph")

    # show the graph in 3D view
    # mltg.visualize_graph(mltg)

    mltg.init_sampled_actions('deform_data_small_step.csv')
    # mltg.init_sampled_actions2(samples_db)

    # samples_db = modify_action_samples(samples_db)

    # samples_db, push_pose = process_dataframe(samples_db, push_idx=0)
    # samples_db = process_dataframe(samples_db)
    

    # print('test database')
    # test_database(env, samples_db, flat_state_dict)
    start_config = (0, 0, 0)
    target_config = (-0.0, 0.0, -180)
    # target_config = (-0.5, 0.8, -150)
    # target_config = (-0.5, -0.5, -150)

    env.set_target_corner([target_config[0], target_config[1]], np.deg2rad(target_config[2]))
    
    

    # define target config
    

    # target_config = (-0.3, -0.5, 50)

    # find the closest node to the target config

    # target_node = mltg.nearest_expanded_node(target_node)

    # path2target = mltg.transition_dict[target_node['layer']][target_node['index_x']][target_node['index_y']]['incoming']
    # print("how many paths to target: ", len(path2target))


    # find path with A* algorithm
    # path = mltg.find_path_from_target(start_node, target_node)

    # path = mltg.a_star_search(start_node, target_node)

    # time_start = time.time()
    # path = mltg.a_star_search_in_expanding(start_node, target_node, samples_db)
    # time_end = time.time()
    # print("time for single grasp a star search: ", time_end - time_start)

    # -------------- plan path ---------------- #
    start_time = time.monotonic()

    path = mltg.plan(start_config, target_config)

    end_time = time.monotonic()
    print(f"Plan runtime: {end_time - start_time:.6f} seconds")

    
    if len(path) == 0:
        print("no path found")

    print("length of path: ", len(path) - 1)
    # -------------- visualize path ---------------- #
    mltg.visualize_path(path, mltg.width, mltg.height)
 

    # -------------- execute path ---------------- #
    if path is None:
        print("no path found")
        exit(0)
    
    path_list = [path]

    touched = False


    for i, path in enumerate(path_list):
        prev_layer = 0
        env.set_state(flat_state_dict)
        env.init_pos()

        # Recording
        env.start_record()
        for (next_node, action) in path[1:]:
            
            
            # find current push pose of action
            # find current trans_x and trans_y of action
            # rot is the same

            # set pusher
            
            
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

        save_name = osp.join('./data/', str(target_config) + "_{}_{}.gif".format("a_star", i))
        env.end_record(save_name)
        # break

    # env.set_pusher([0, 10, 0, 0])

    pos = env.get_pusher_pos()
    push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    env.push(np.array([push_x, -0.01, push_y, push_ori, 0]))

    pyflex.step()


    # --------- Capture the last frame ---------------- #
    # center_object()
    # # camera_param = np.array([*[0.0, 2.0, 0], *[0.0, -90 / 180. * np.pi, 0.0], env.camera_width, env.camera_height])
    # camera_param = dict()
    # camera_param['pos'] = [0.0, 1.0, 0]
    # camera_param['angle'] = [0.0, -90 / 180. * np.pi, 0.0]
    # camera_param['width'] = env.camera_width
    # camera_param['height'] = env.camera_height
    # env.update_camera("zoom_in", camera_param)
    
    # env.get_current_corner_pos()
    # ori_shift = env.get_current_ori_change2(env.cornerPos_init, env.cornerPos)

    # print("ori change: ", ori_shift)

    # frame = env.render(mode='rgb_array')
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # # save frame as image
    # cv2.imwrite("last_frame.jpg", frame)
    # focus camera and save image
    


    # # print("approach vector", env.get_approach_vec())

    # # pusher placement and first action
    # # push_pos = [push_x, push_y, push_ori], action = [trans_x, trans_y, rot]
    # start_pos, action, deformation, df_table = selectStartAction(env, engine)
    # deforms.append(deformation)
    # push_action = construct_push_action(start_pos, action, closed=True)

    # env.init_pusher(np.array([start_pos[0], 2*0.01, start_pos[1], start_pos[2]]))
    # # env.set_pusher(np.array([start_pos[0] + 0.05, 2*0.01, start_pos[1] + 0.2, start_pos[2]]))
    # env.push(push_action)

    # env.get_current_ori_change()

    # action_db = None
    # new_pos = np.array([push_action[0], push_action[2], push_action[3]])
    
    # # subsequent action
    # for i in range(100):
    #     # in each new state, find the best action considering all push_pos
    #     action, action_db, deformation = selectActionFromDB(new_pos, engine, df_table)
    #     # the sample action should be projected to the real position on the fabric
    #     push_x, push_y, push_ori = env.get_push_pos(action[0], action[1], action[2])

        
    #     deforms.append(deformation)
    #     push_action = construct_push_action(new_pos, action[3:], closed=False)
        
    #     # if deformation > 0.135:
    #     #     print("large deformation, not a good action, done!")
    #     #     break

    #     # move pusher to the push_pos before pushing
    #     env.set_pusher(np.array([push_x, 0.01, push_y, push_ori]))

    #     env.push(push_action)
    #     # env.get_current_ori_change()
    #     new_pos = np.array([push_action[0], push_action[2], push_action[3]])
    #     env.get_current_corner_pos()
    #     cur_dist = env.distance(env.target_cornersPos, env.cornerPos)
    #     print("Current distance to the target position: ", cur_dist)

    #     rot_dist = abs(env.get_current_ori_change2(env.target_cornersPos, env.cornerPos))
    #     print("Current rot to the target ori: ", np.degrees(rot_dist))

    #     # if abs(np.degrees(rot_dist)) < 10:
    #     #     break

    #     config = env.get_default_config()
    #     cloth_dimx, cloth_dimy = config['ClothSize']
    #     # if(cur_dist < 0.5*min(cloth_dimx, cloth_dimy)*env.cloth_particle_radius):
    #     #     break
            
    #     # print("target:")
    #     # print(env.target_cornersPos)
    #     # print(np.mean(env.target_cornersPos, axis=0))
    #     # print("center: ", env.get_center())
    #     # print("target center: ", np.mean(env.target_cornersPos, axis=0))

    # save_name = osp.join('./data/', "%s.gif" % "best_action")
    # # env.end_record(save_name)

    # import matplotlib.pyplot as plt

    # # Sample data
    # x = range(len(deforms))
    # y = deforms
    # print(y)

    # # Create a plot
    # plt.plot(x, y)

    # # Add labels and a title
    # plt.xlabel('step')
    # plt.ylabel('deformation')
    # plt.title('Deformation Graph')

    # # Display the plot
    # plt.show()



    


    
