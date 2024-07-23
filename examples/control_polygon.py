from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
import pyflex
import numpy as np
import sys
import pickle

Tolerance = 0.015

def create_env(env_name='PolygonPushPPP', model_name='Pants', contstrained=False):

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
    env_kwargs['model_name'] = model_name

    env_kwargs['constraints'] = contstrained

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS['PolygonPushPPP'](**env_kwargs)
    # env.reset()
    
    
    env.init_pusher([0,0.1,0,0])

    # frames = [env.get_image(720, 720)]

    # flatten_area = env._set_to_flatten()
    pyflex.step()
    
    return env

def test_function(env):
    env.snap_to_center()

    ### Test basic functions ###
    # env.set_vertices()
    # env.show_contact_poses()
    
    # For a new model, we have to run this to find the corner and segments
    # env.set_init_pos()
    # env.find_corner_and_segments()
    # env.plot_corner_and_segments()

    # If the new model is the rectangle cloth without obj file, we have to use another function to find the corner and segments
    
    # for i in range(21):
    #     env.place_contact_pose(i)
    # start_pose = env.place_contact_pose(0)
    # cp = env.place_contact_pose(2)
    # print("cp: ", cp)

    env.place_contact_poses()


    ### Test push action ###
    # push cloth, x, y, z, theta, closed
    # start_pose = env.place_contact_pose(2)
    # # -0.43301270189221935, 0.25, 1.3089969389957472
    # target_pose = np.array(start_pose) + np.array([0.25, 0.43301270189221935, 0.5, 1.3089969389957472])
    # # append 1 to target_pose
    # action = np.append(target_pose, 1)
    # env.push(action)

    while True:
        pyflex.step()

def test_sample_action(env):
    ### Sample action ###
    contact_poses = env.get_contact_poses()
    env.sample_action(contact_poses)

def take_pictures(env):
    env.start_record()

    env.snap_to_center(center=[env.table_center[0], env.table_size[1], env.table_center[1]])
    

    start_pose = [0, 0, 0]
    target_pose = [0.3, 0.1, np.deg2rad(68)] # relative to start_pose
    env.set_start_state([start_pose[0] + env.table_center[0], env.table_size[1], start_pose[1] + env.table_center[1]], start_pose[2])
    
    env.camera_on_cloth()
    for i in range(10):
        env.video_frames.append(env.render(mode='rgb_array'))

    env.set_target_corner([target_pose[0], target_pose[1]], target_pose[2], draw_target=True)
    env.video_frames.append(env.render(mode='rgb_array'))

    # Hide cloth from camera
    env.set_start_state([10, 0, 10], 0)
    pyflex.step()
    for i in range(10):
        env.video_frames.append(env.render(mode='rgb_array'))
    

    gif_path = './data/pictures.gif'
    env.end_record(gif_path)

def method_1(env):
    return env.p2p_repositioning(epsilon=Tolerance)

def method_2(env, node_s, node_t, initial_contact_poses):
    # p2p + action sample -> contact pose
    constraints = {
        "table_center": (env.table_center[0], env.table_center[1]),
        "table_size": (env.table_size[0], env.table_size[2]),
        "object_size": (0.48, 0.4),
        "object_center_offset": (0, 0)
    }

    task_config = {
        "node_s": node_s,
        "node_t": node_t,
        "cps": initial_contact_poses,
        "control_file": env.action_file,
        "constraints": constraints
    }
    return env.p2p_repositioning(epsilon=Tolerance, cp_optimal=task_config)

def method_3(env, node_s, node_t, initial_contact_poses):
    # p2p + subgoal
    constraints = {
        "table_center": (env.table_center[0], env.table_center[1]),
        "table_size": (env.table_size[0], env.table_size[2]),
        "object_size": (0.48, 0.4),
        "object_center_offset": (0, 0)
    }

    target_pose = [node_t[0] - env.table_center[0], -(node_t[1] - env.table_center[1]), -int(np.rad2deg(node_t[2]))]

    rrt_path_file = './data/RRT_Star/{}.pkl'.format(str(target_pose).replace(' ', ''))
    path = load_path(rrt_path_file)

    # path = env.rrt_planning(node_s, node_t, initial_contact_poses, constraints=constraints)
    env.path_repositioning(path, refinement=False, cp_star=False)

    return env.p2p_repositioning(epsilon=Tolerance)

def method_4(env, node_s, node_t, initial_contact_poses, reuse_path=False):
    constraints = {
        "table_center": (env.table_center[0], env.table_center[1]),
        "table_size": (env.table_size[0], env.table_size[2]),
        "object_size": (0.4, 0.4),
        "object_center_offset": (0, 0)
    }
    # print(env.action_file)
    

    target_pose = [node_t[0] - env.table_center[0], -(node_t[1] - env.table_center[1]), -int(np.rad2deg(node_t[2]))]

    rrt_path_file = './data/RRT_Star/{}.pkl'.format(str(target_pose).replace(' ', ''))
    import os
    if reuse_path and os.path.exists(rrt_path_file):
        print("Path file exists!")
        
        path = load_path(rrt_path_file)
    else:
        path = env.rrt_planning(node_s, node_t, initial_contact_poses, constraints=constraints)
        # save the path to file
        if not path:
            print("No MG-RRT* path found!")
            return
        write_path(path, rrt_path_file)

    env.path_repositioning(path, refinement=False)

    return env.p2p_repositioning(epsilon=Tolerance)

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
    
def write_path(path, file_path):
    """Write a path to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(path, file)
        print(f"Path saved to '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while writing the path: {e}")


def test_planning(env, target, method=0):

    # Without subgoal
    # Method 1 = P2P（angle）
    # Method 2 = P2P (action sample -> contact pose）

    # With subgoal
    # Method 3 = P2P(angle + subgoal)
    # Method 4 = RRT* = P2P(action sample -> contact pose  + subgoal)

    # real_path = [([0.0, 0.0, 0.0], None), ([-0.19657929908312763, 0.05705603813970472, 1.3089969389957472], 18), ([0.17061203617598364, -0.028495357201568483, 2.6179938779914944], 5), ([0.0, 0.0, 3.141592566167013], 2)]
    env.set_start_state([0, 0, 0], 0)
    # get default contact poses relative to the fabric frame(should call it when the fabric frame is the world frame)
    init_contact_poses = env.get_contact_poses()

    env.remove_inter()

    env.remove_target()

    # env.snap_to_center()
    env.snap_to_center(center=[env.table_center[0], env.table_size[1], env.table_center[1]])

    

    start_pose = [0, 0, 0]
    target_pose = target # relative to start_pose

    env.camera_on_cloth()

    env.start_record()

    # start_pose = np.array(real_path[0][0])
    env.set_start_state([start_pose[0] + env.table_center[0], env.table_size[1], start_pose[1] + env.table_center[1]], start_pose[2])
    
    for i in range(50):
        # capture the initial state
        env.shoot_frame()

    env.set_pusher([env.table_center[0],0.1,env.table_center[1],0])

    # set_target_corner is a 2D transformation task
    env.set_target_corner([target_pose[0], target_pose[1]], target_pose[2], draw_target=(method==0 or method==1))

    # RRT planning
    node_s = [start_pose[0] + env.table_center[0], start_pose[1] + env.table_center[1], start_pose[2]]
    node_t = [target_pose[0] + env.table_center[0], target_pose[1] + env.table_center[1], target_pose[2]]

    
    env.reset_planner()
    
    # print("path: ", path)

    

    # path = [([0,0,0], None), ([0, 0.0, -np.pi/2], 6), ([0, 0, -np.pi], 6)]
    # if method != 0:
    #     path = env.rrt_planning(node_s, node_t, contact_poses, constraints=constraints)
    #     env.path_repositioning(path, refinement=(method==2))

    if method == 0:
        success = method_1(env)
    elif method == 1:
        success = method_2(env, node_s, node_t, init_contact_poses)
    elif method == 2:
        # to use path planned by RRT*, run method_4 before method_3
        success = method_3(env, node_s, node_t, init_contact_poses)
    elif method == 3:
        success = method_4(env, node_s, node_t, init_contact_poses, reuse_path=True)



    # data_path = './data'
    # file name of target pose
    # round to near integer
    target_pose[1] = -target_pose[1]
    target_pose[2] = int(-np.round(np.rad2deg(target_pose[2])))

    folder = "P2P"
    if method == 0:
        folder = "P2P1"
    elif method == 1:
        folder = "P2P2"
    elif method == 2:
        folder = "P2P1_plus_Subgoal"
    elif method == 3:
        folder = "RRT_Star"

    save_path = './data/{}/{}.gif'.format(folder, str(target_pose).replace(' ', ''))

    env.end_record(save_path)

    actions = env.get_actions()
    print("Actions: ", actions)

    print("Success: ", success)

    # estimate final mean particle distance
    EDs = env.EDs()
    RDs = env.RDs()
    LDs = env.LDs()
    chamfer = env.CHAMFER()
    IOUs = env.IOUs()

    print("EDs: ", EDs)
    print("RDs: ", RDs)
    print("LDs: ", LDs)
    print("Chamfer: ", chamfer)
    print("IOUs: ", IOUs)

    # write mpd to file
    with open('./data/{}/{}.txt'.format(folder, str(target_pose).replace(' ', '')), 'w') as f:
        # write MPD to file
        f.write("EDs: " + str(EDs) + "\n")
        f.write("RDs: " + str(RDs) + "\n")
        f.write("LDs: " + str(LDs) + "\n")
        f.write("Chamfer: " + str(chamfer) + "\n")
        f.write("IOUs: " + str(IOUs) + "\n")
        # f.write("Actions: " + str(actions) + "\n")
        # f.write("Success: " + str(success) + "\n")
        

def exp(env, target, method=0):
    target = [target[1], target[0], -np.deg2rad(target[2])]
    test_planning(env, target, method=method)

if __name__ == '__main__':
    env = create_env(model_name="LongSleeve", contstrained=True)
    test_function(env)
    # test_sample_action(env)
    # take_pictures(env)
    # get argue from command line
    # method = int(sys.argv[1])



    # target = [0.01, 0.33, 10]
    # target = [-0.03, -0.30, 30]
    # target = [-0.06, 0.23, -15]
    # target = [0, 0.17, 83]
    # target = [0.03, 0.38, -68]
    # target = [0.04, 0.23, -18]
    # target = [0.02, -0.29, -59]
    # target = [0.02, 0.32, -47]
    # target = [-0.01, 0.39, 78]
    # target = [0.07, 0.38, 80]
    # target = [0.01, 0.33, 15]
    # target = [-0.03, -0.30, 25]
    # target = [-0.06, 0.31, -75]
    # target = [0, -0.17, -53]
    # target = [-0.03, 0.28, -78]
    # target = [0.01, 0.33, 5]
    # target = [-0.13, -0.30, 60]
    # target = [-0.16, 0.33, -75]
    # target = [0, 0.17, -63]
    # target = [-0.03, 0.28, -28]

    easy_configs = [
        # [0.01, 0.33, 10],
        # [-0.03, -0.30, 30],
        # [-0.06, 0.23, -15],
        # [0, 0.17, 83],
        # [0.03, 0.38, -68],
        # [0.04, 0.23, -18],
        # [0.02, -0.29, -59],
        # [0.02, 0.32, -47],
        # [-0.01, 0.39, 78],
        # [0.07, 0.38, 80],
        # [0.01, 0.33, 15],
        # [0.03, -0.20, 34],
        # [-0.06, 0.31, -75],
        [0, -0.17, -53],
        # [-0.03, 0.28, -78],
        # [0.01, 0.33, 5],
        # [-0.04,-0.15,60],
        # [-0.06,0.23,-75],
        # [0, 0.17, -63],
        # [-0.03, 0.28, -28]
    ]


    fail_configs = [
        # [0.02, -0.32, -144],
        # [0.02, -0.29, -59],
        # [0.04, -0.23, -98],
        # [0.02, -0.12, -148],
        # [0, 0, -156],
        # [0, 0, -153],
        # [0, 0, -143],
        # [0, 0, -141],
        # [0, 0, 133]
        # [0.07, 0.22, 170]
    ]







    hard_configs = [
        # [0, 0, -153],
        [0, 0, 117],
        [0, 0, -143],
        [0, 0, 155],
        [0, 0, -141],
        [0, 0, -120],
        [0, 0, 133],#
        [0, 0, -156],#
        # [0, 0, 180],
        [0, 0, -131],
        [0.04, -0.23, -98],
        [-0.02, -0.21, -129],#
        [0.02, -0.32, -144],
        [-0.01, 0.39, 168],
        [0.07, 0.28, 180],
        [0.04, -0.30, -92],#
        [-0.02, -0.21, -139],
        [0.02, -0.12, -148],
        [-0.01, 0.34, 167],
        [0.07, 0.22, 170] #
    ]



    

    fail_configs = [
        # [0, 0, 180],
        # [0, 0, 133],#
        # [0, 0, -156],#
        # [-0.02, -0.21, -129],#
        # [0.04, -0.30, -92],#
        # [0.07, 0.22, 170] #
        # [-0.06, 0.23, -75],
        # [-0.04, -0.15, 60]
        # [0, 0, -153],
        # [0.07, 0.22, 170]
        # [-0.02, -0.21, -129],
        # [0.07, 0.28, 180]
    ]
    
    # for target in easy_configs:
    #     exp(env, target, method=0)
        # exp(env, target, method=1)
        # exp(env, target, method=3)
        # exp(env, target, method=2)

    for target in hard_configs:
        exp(env, target, method=0)
        # exp(env, target, method=1)
        # exp(env, target, method=3)
        # exp(env, target, method=2)

    # exp(env, target, method=method)

