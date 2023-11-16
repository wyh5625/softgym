from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
import pyflex
import numpy as np

def create_env(env_name='PolygonPushPPP', contstrained=False):

    # env_name = 'PantsPushPPP'
    env_kwargs = env_arg_dict['PantsPushPPP']

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

    env_kwargs['constraints'] = contstrained

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

def test_function(env):
    env.snap_to_center()

    ### Test basic functions ###
    # env.set_vertices()
    # env.show_contact_poses()
    
    # env.set_init_pos()
    # env.find_corner_and_segments()
    # env.plot_corner_and_segments()
    
    # for i in range(21):
    #     env.place_contact_pose(i)
    # start_pose = env.place_contact_pose(0)
    # cp = env.place_contact_pose(2)
    # print("cp: ", cp)


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

def test_planning(env):

    # real_path = [([0.0, 0.0, 0.0], None), ([-0.19657929908312763, 0.05705603813970472, 1.3089969389957472], 18), ([0.17061203617598364, -0.028495357201568483, 2.6179938779914944], 5), ([0.0, 0.0, 3.141592566167013], 2)]

    # get default contact poses relative to the fabric frame(should call it when the fabric frame is the world frame)
    contact_poses = env.get_contact_poses()

    # env.snap_to_center()
    env.snap_to_center(center=[env.table_center[0], env.table_size[1], env.table_center[1]])

    start_pose = [0, 0, 0]
    target_pose = [-0.3, -0.03, np.deg2rad(30)] # relative to start_pose

    # start_pose = np.array(real_path[0][0])
    env.set_start_state([start_pose[0] + env.table_center[0], env.table_size[1], start_pose[1] + env.table_center[1]], start_pose[2])
    env.camera_on_cloth()
    env.set_pusher([env.table_center[0],0.1,env.table_center[1],0])

    # set_target_corner is a 2D transformation task
    env.set_target_corner([target_pose[0], target_pose[1]], target_pose[2], draw_target=False)

    # RRT planning
    node_s = [start_pose[0] + env.table_center[0], start_pose[1] + env.table_center[1], start_pose[2]]
    node_t = [target_pose[0] + env.table_center[0], target_pose[1] + env.table_center[1], target_pose[2]]

    offset = env.get_center_offset()
    constraints = {
        "table_center": (env.table_center[0], env.table_center[1]),
        "table_size": (env.table_size[0], env.table_size[2]),
        "object_size": (0.48, 0.4),
        "object_center_offset": (0, 0)
    }
    
    path = env.rrt_planning(node_s, node_t, contact_poses, constraints=constraints)
    # print("path: ", path)

    env.start_record()

    # path = [([0,0,0], None), ([0, 0.0, -np.pi/2], 6), ([0, 0, -np.pi], 6)]

    env.path_repositioning(path, refinement=True)
    env.p2p_repositioning()


    # data_path = './data'
    gif_path = './data/record.gif'

    env.end_record(gif_path)



if __name__ == '__main__':
    env = create_env(contstrained=True)
    # test_function(env)
    # test_sample_action(env)
    test_planning(env)

