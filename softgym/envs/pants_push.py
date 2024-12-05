import numpy as np
import random
import pyflex
from gym import error
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object
import os.path as osp
from softgym.utils.visualization import save_numpy_as_gif
from scipy.spatial.transform import Rotation
import scipy.spatial
from softgym.envs.cloth_push import ClothPushEnv

class PantsPushEnv(ClothPushEnv):
    # def __init__(self, **kwargs):
    #     """
    #     :param cached_states_path:
    #     :param num_picker: Number of pickers if the aciton_mode is picker
    #     :param kwargs:
    #     """
    #     super().__init__(**kwargs)

        # self.config = self.get_default_config()
        # self.set_scene(self.config)

        # self.get_cached_configs_and_states(
        #     cached_states_path, self.num_variations)
            
        # self.prev_covered_area = None  # Should not be used until initialized
        # self.cornerPos_init = None
        # self.init_covered_area = None

        # self.init_pos()
        # self.init_particles = pyflex.get_positions().reshape(-1, 4)

        # self.update_camera(self.config['camera_name'], self.config['camera_params'][self.config['camera_name']])

    def test_pusher(self):
        import pandas as pd
        cp_dict = {}
        cs_df = pd.read_csv("control_pants_50cm_2.csv")
        grouped = cs_df.groupby(['push_x', 'push_y', 'push_ori'])
        print("Num of contact pose: ", len(grouped.groups.keys()))
        for idx, g_key in enumerate(grouped.groups.keys()):
            group = grouped.get_group(g_key).reset_index(drop=True)
            cp_dict[idx] = list(g_key)

        # 
        push_pose = cp_dict[0]
        # insert 0 to the second position
        push_pose.insert(1, 0.02)

        print("push pose:", push_pose)
        self.set_pusher(push_pose)
        pyflex.step()

    def generate_pusher_poses(self):
        contact_poses = []

        top = [(self.cornerPos_init[0] + self.cornerPos_init[1])[[0,2]]/2, np.pi]
        right = [(self.cornerPos_init[1] + self.cornerPos_init[2])[[0,2]]/2, -np.pi/2]
        left = [(self.cornerPos_init[0] + self.cornerPos_init[6])[[0,2]]/2, np.pi/2]

        contact_poses.append(top)
        contact_poses.append(right)
        contact_poses.append(left)
        for side_pose in np.array(self.cornerCP_init)[[0,1,2,6]]:
            p0 = [side_pose[0], side_pose[1]]
            p1 = [side_pose[2], side_pose[3]]
            contact_poses.append(p0)
            contact_poses.append(p1)
        
        return contact_poses

    def get_corner_particles(self):
        # get for corners of the pants
        pos = pyflex.get_positions().reshape(-1, 4)[:, [0, 2]]
        
        min_x = np.min(pos[:, 0])
        max_x = np.max(pos[:, 0])
        min_y = np.min(pos[:, 1])
        max_y = np.max(pos[:, 1])

        # find 3 concave corners
        # find all particles that has the same y value of max_y
        bottom_particles = pos[pos[:, 1] == max_y]

        middle_x = (min_x + max_x)/2

        # the third corner
        # find the bottom_particles that has the x value greater than middle_x, get the one with the smallest x value
        right_leg_particles = bottom_particles[bottom_particles[:, 0] > middle_x]
        third_corner = right_leg_particles[right_leg_particles[:, 0] == np.min(right_leg_particles[:, 0])][0]
        print("third corner:", third_corner)

        # the fourth corner
        # find the particles that the difference of x value with middle_x is not greater than a particle radius and get the one with the largest y value
        middle_particles = pos[abs(pos[:, 0] - middle_x) < self.cloth_particle_radius]
        print("middle particles:", middle_particles)
        fourth_corner = middle_particles[middle_particles[:, 1] == np.max(middle_particles[:, 1])][0]
        print("fourth corner:", fourth_corner)

        # the fifth corner
        # find the bottom_particles that has the x value smaller than middle_x, get the one with the largest x value
        left_leg_particles = bottom_particles[bottom_particles[:, 0] < middle_x]
        fifth_corner = left_leg_particles[left_leg_particles[:, 0] == np.max(left_leg_particles[:, 0])][0]
        print("fifth corner:", fifth_corner)

        c0 = [min_x, min_y]
        c1 = [max_x, min_y]
        c2 = [max_x, max_y]
        c3 = third_corner
        c4 = fourth_corner
        c5 = fifth_corner
        c6 = [min_x, max_y]

        # four corners of the pants
        corner_pos = np.array([c0, c1, c2, c3, c4, c5, c6])

        # print("corner size:", len(corner_pos))


        self.cornerIdx = []

        # find index of particle is closest to the corner
        for i in range(len(corner_pos)):
            dists = scipy.spatial.distance.cdist(corner_pos[i].reshape(1, 2), pos)
            idx_dists = np.hstack([np.arange(pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
            idx_dists = idx_dists[idx_dists[:, 1].argsort()]
            self.cornerIdx.append(int(idx_dists[0, 0]))
            # print("idx:", int(idx_dists[0, 0]), "pos:", pos[int(idx_dists[0, 0])])

    def set_scene(self, config, state=None):
        # config = self.config
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 5
        render_mode = 2
        # scene_params = np.concatenate([ config['pos'][:], [config['scale'], config['rot']], config['vel'][:], [config['stiff'], config['mass'], config['radius']],
        #                         camera_params['pos'][:], camera_params['angle'][:], [camera_params['width'], camera_params['height']] ])
        scene_params = np.array([*config['ClothPos'], *config['ClothScale'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], config['mass'], config['flip_mesh'],
                                 config['static_friction'], config['dynamic_friction']],
                                 dtype=np.float32)
        pyflex.set_scene(env_idx, scene_params, 0)

    
    def get_default_config(self):
        cam_pos, cam_angle = np.array([0.0, 2.3, 0.00]), np.array([0, -np.pi/2., 0.])
        camera_width, camera_height = 720, 720
        config = {
            'ClothPos': [0.01, 0.01, 0.01],
            'ClothScale': [0.5, 0.5],
            'ClothSize': [1, 1],
            'ClothStiff': [0.8, 1, 0.8], # [2.9, 0.1, 2.8]
            'camera_name': 'default_camera',
            'stiff': 0.9,
            'radius': self.cloth_particle_radius,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': camera_width,
                                   'height': camera_height}},
            'mass': 0.2,
            'flip_mesh': 0,
            'static_friction': 0.2,
            'dynamic_friction': 0.2,
            'drop_height': 0.1,
            'flatten_area': 0,
            'env_idx': 5
        }

        return config

    def adjust_controls(self):
        import csv
        import pandas as pd

        
        cs_df = pd.read_csv("control_pants_50cm.csv")

        # Define the vector to be added to push_x and push_y
        # vector_to_add = [0.1, 0.2]  # Modify this as needed

        # find the center of four corners
        pos = pyflex.get_positions().reshape(-1, 4)[:, [0, 2]]
        min_x = np.min(pos[:, 0])
        max_x = np.max(pos[:, 0])
        min_y = np.min(pos[:, 1])
        max_y = np.max(pos[:, 1])
        middle_x = (min_x + max_x)/2
        middle_y = (min_y + max_y)/2
        center = [middle_x, middle_y]

        # find the center that is average of all the particles
        center_ = np.mean(pos, axis=0)

        vector_to_add = center - center_


        # Input and output file paths
        input_file_path = "control_pants_50cm_2.csv"
        output_file_path = "control_pants_50cm_3.csv"

        # Read the CSV file and modify the specified columns
        modified_rows = []
        with open(input_file_path, "r") as input_file:
            reader = csv.reader(input_file)
            header = next(reader)  # Read and ignore the header row

            for row in reader:
                push_x = float(row[0]) + vector_to_add[0]
                push_y = float(row[1]) + vector_to_add[1]
                push_ori = float(row[2])
                trans_x = float(row[3])
                trans_y = float(row[4])
                rot = float(row[5])
                deformation = float(row[6])

                modified_rows.append([push_x, push_y, push_ori, trans_x, trans_y, rot, deformation])

        # Save the modified data to a new CSV file
        with open(output_file_path, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)  # Write the header row
            writer.writerows(modified_rows)  # Write the modified rows

    # it's based on the default configuration of scene
    def test_sample(self, pusher_pos, action, record=False, img_size=None, save_video_dir=None):

        self.set_pusher(pusher_pos)

        # get virtual rigid position after action
        pos = pyflex.get_positions().reshape(-1, 4)
        rigid_pos = self.transform_particles(pos, translation=(action[:3]-pusher_pos[:3]),
                                             angle=(action[3]-pusher_pos[3]),
                                             center=pusher_pos[:3],
                                             set_position=False)
        
        actions = [action]
        for action in actions:
            _, _, _, info = self.step(
                action, record_continuous_video=record, img_size=img_size)
            # if record:
            #     frames.extend(info['flex_env_recorded_frames'])

        max_wait_step = 300
        stable_vel_threshold = 0.5

        
        now_pos = pyflex.get_positions().reshape(-1, 4)
        error = now_pos[:, :3] - rigid_pos[:, :3]
        error_norm = np.linalg.norm(error, axis=1)
        tot_errors = np.sum(error_norm)

        num_particle = pyflex.get_n_particles()
        avg_error = tot_errors/num_particle

        return avg_error


        
        