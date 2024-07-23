from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
import pyflex
import numpy as np
import sys
import pickle

Tolerance = 0.015
def create_env(env_name='ClothPickAndDrop', model_name='TshirtPokeZp', drop_pos=[0, 1.5, 0], drop_ori=[np.pi/2, 0, np.pi]):

    # env_name = 'PantsPushPPP'
    env_kwargs = env_arg_dict[env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = 1
    env_kwargs['render'] = True
    env_kwargs['headless'] = 0
    env_kwargs['action_mode'] = 'pickerpickplace'
    env_kwargs['num_picker'] = 1
    env_kwargs['picker_radius'] = 0.01
    env_kwargs['tweak_panel'] = 0
    env_kwargs['model_name'] = model_name
    env_kwargs['pos'] = drop_pos
    env_kwargs['ori'] = drop_ori


    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = SOFTGYM_ENVS[env_name](**env_kwargs)

    # set cloth position

    # set cloth orientation

    # env.action_tool.reset([0,0.1,0,0])

    # frames = [env.get_image(720, 720)]

    # flatten_area = env._set_to_flatten()

    pyflex.step()

    
    return env

def dummy_function(env):
    # env.snap_to_center()

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

    # env.place_contact_poses()


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

    target_pose = [0.22, 0.07, np.deg2rad(-170)] # relative to start_pose
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
    
    env.start_record()
    gif_path = './data/pictures.gif'
    env.end_record(gif_path)

def take_frame(env):
    env.start_record()
    env.video_frames.append(env.render(mode='rgb_array'))

    png_path = './data/picture.png'
    env.save_png(png_path)

    gif_path = './data/pictures.gif'
    env.end_record(gif_path)

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

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def get_camera_params(objectPosition, l, pitch, yaw, roll):
    # Determine the distance from the object


    # Convert pitch, yaw, and roll to radians
    pitchRadians = pitch * (np.pi / 180)
    yawRadians = yaw * (np.pi / 180)
    rollRadians = roll * (np.pi / 180)

    # Calculate the camera position
    cameraX = objectPosition[0] + l * np.cos(yawRadians) * np.cos(pitchRadians)
    cameraY = objectPosition[1] + l * np.sin(pitchRadians)
    cameraZ = objectPosition[2] + l * np.sin(yawRadians) * np.cos(pitchRadians)

    camera_pos = np.array([cameraX, cameraY, cameraZ])

    # get orientation of camera
    directionVector = objectPosition - camera_pos
    directionVector = normalize(directionVector)

    # Calculate pitch, yaw, and roll
    pitch = np.arctan2(directionVector[1], np.sqrt(directionVector[0]**2 + directionVector[2]**2))
    yaw = np.arctan2(directionVector[0], directionVector[2])
    right = np.cross(np.array([0, 1, 0]), directionVector)
    right = normalize(right)
    roll = np.arctan2(right[1], right[0])

    # Convert to degrees
    pitchDegrees = pitch * (180 / np.pi)
    yawDegrees = yaw * (180 / np.pi)
    rollDegrees = roll * (180 / np.pi)

    camera_ori = [pitchDegrees, yawDegrees, rollDegrees]

    return camera_pos, camera_ori
        


if __name__ == '__main__':

    # Step 1: Drop a cloth in the environment
    drop_pos = [0.0, 1.0, 0.0]  # x, y(height), z
    drop_ori = [-np.pi/2, 0, 0] # rotation order: "yaw-pitch-roll" = "drop_ori[1]-drop_ori[0]-drop_ori[2]"
    env = create_env(drop_pos=drop_pos, drop_ori=drop_ori)

    # set camera config
    cam_name, _, _ = env.get_camera_params()
    # cam_pos, cam_angle = np.array([0.0, 2.3, 0.0]), np.array([0.0, -90 / 180. * np.pi, 0.0])
    # ----------- Hard code the camera position and angle ------------
    cam_pos, cam_angle = np.array([0.0, 2.8, 0.0]), np.array([0.0, -90 / 180. * np.pi, 0.0])
    cam_param = {'pos': cam_pos,
                'angle': cam_angle,
                'width': 720,
                'height': 720}
    
    # ----------- Find the camera position and angle relative to position of cloth ------------
    # cc = env.get_cloth_center()
    # camera_distance = 1.5
    # # find the camera position by finding the vector from the cloth center to the camera, with the length of 0.5, and the direction is set by shooting angles of three values
    # camera_pos, camera_ori = get_camera_params(cc, camera_distance, 0, 0, 0)
    # cam_param['pos'] = camera_pos
    # cam_param['angle'] = camera_ori



    # Viewing point of camera
    env.update_camera(cam_name, cam_param)

    # wait the cloth to be static
    for i in range(150):
        pyflex.step()

    # initialize picker
    env.init_picker(pos=[0, 0.01, 0])


    # Step 2: Pick up the cloth
    pos2d = [0.0, 0.0]  # pick up point

    # random_pos2d on the cloth

    height = 1.0    # raise height
    env.pick_up(pos2d=pos2d, height=height)
    


    # Step 3: Drop the cloth
    env.drop(pos2d=pos2d, height=height)

    # wait for the cloth to stop moving

    # Step 4: Take pictures
    # take_pictures(env)
    take_frame(env)
    

    dummy_function(env)


    

