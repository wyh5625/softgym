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

class ClothPickEnv(ClothEnv):
    def __init__(self, pos=[0.0, 2.15, 0.0], ori=[0.0, 0.0, 0.0], cached_states_path='cloth_push_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)

        

        self.config = self.get_default_config()
        self.config['ClothPos'] = pos
        self.config['ClothOri'] = ori

        self.set_scene(self.config)

        # self.get_cached_configs_and_states(
        #     cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized
        self.cornerPos_init = None
        self.init_covered_area = None

        self.init_particles = pyflex.get_positions().reshape(-1, 4)
        
        self.init_pos(pos=self.config['ClothPos'])

        self.update_camera(self.config['camera_name'], self.config['camera_params'][self.config['camera_name']])

    def init_pos(self, pos=[0, 0, 0]):

        self.default_pos = pyflex.get_positions().reshape(-1, 4)
        self.init_pos = self.default_pos.copy()
        self.init_pos[:, 0] -= np.mean(self.init_pos, axis=0)[0] - pos[0]
        self.init_pos[:, 1] -= np.mean(self.init_pos, axis=0)[1] - pos[1]
        self.init_pos[:, 2] -= np.mean(self.init_pos, axis=0)[2] - pos[2]

        pyflex.set_positions(self.init_pos.flatten())

    def init_picker(self, pos=[0, 0, 0]):
        self.action_tool.reset(pos)


    def drop_at(self, x, y, z):
        curr_pos = pyflex.get_positions().reshape(-1, 4)
        curr_pos[:, :3] -= np.mean(self.init_pos, axis=0)[:3]
        curr_pos[:, :3] += [x, y, z]

        pyflex.set_positions(curr_pos.flatten())

    # def push(self, action, record=False, img_size=None, save_video_dir=None):
    #     # [x, y, z, rot, pick/drop]
    #     # init scene
    #     # default_config = self.get_default_config()
    #     actions = [action]
    #     for action in actions:
    #         _, _, _, info = self.step(
    #             action, record_continuous_video=record, img_size=img_size)

    def pick_up(self, pos2d=[0, 0], height=0.5):
        pos3d = [pos2d[0], 0.015, pos2d[1]]
        self.action_tool.set_picker_pos(pos3d)
        action = np.array([pos3d[0], height, pos3d[2], 1])
        self.step(action)
        
        # return curr_pos[pickpoint][:3]

    def drop(self, pos2d=[0, 0], height=0.5):
        action = np.array([pos2d[0], height, pos2d[1], 1])
        self.step(action)


        action[1] += 0.01
        action[3] = 0
        self.step(action)



    def get_touched_particle_idx(self, centered_x, centered_y):
        # given a planned action's push_x and push_y, the index of the nearest particle of fabric when its in the intial position is found
        picker_pos = np.array([centered_x, 0, centered_y]).reshape(1, 3)

        # find the index of closest point on the fabric
        dists = scipy.spatial.distance.cdist(picker_pos, self.init_particles[:, :3].reshape((-1, 3)))
        idx_dists = np.hstack([np.arange(self.init_particles.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
        mask = dists.flatten() <= self.action_tool.picker_threshold + self.action_tool.picker_radius + self.cloth_particle_radius
        idx_dists = idx_dists[mask, :].reshape((-1, 2))

        pick_id, pick_dist = None, None
        if idx_dists.shape[0] > 0:
            for j in range(idx_dists.shape[0]):
                if pick_id is None or idx_dists[j, 1] < pick_dist:
                    pick_id = idx_dists[j, 0]
                    pick_dist = idx_dists[j, 1]

        
        return int(pick_id)

 
        

    def transform_particles(self, particle_pos, translation=None, angle=None, center=None, set_position=False):

        new_pos = particle_pos.copy()
    
    
        if angle is not None:
            # rotation
            # print(new_pos)
            new_pos[:, :3] -= center
            centered_pos = new_pos.copy()
            new_pos[:, 0] = (np.cos(angle) * centered_pos[:, 0] -
                             np.sin(angle) * centered_pos[:, 2])
            new_pos[:, 2] = (np.sin(angle) * centered_pos[:, 0] +
                             np.cos(angle) * centered_pos[:, 2])
            new_pos[:, :3] += center


        if translation is not None:
            # translation
            translation = translation.reshape(-1, 3)
            new_pos[:, :3] += translation

        if set_position:
            pyflex.set_positions(new_pos)

        return new_pos
    
    def transform_cloth(self, translation, angle, center):
        pos = pyflex.get_positions().reshape(-1, 4)
        self.transform_particles(pos, translation, angle=angle, center=center, set_position=True)


    def set_working_area(self, width, height, center, color=np.array([1, 1, 1])):
        pyflex.set_shape_color(color)
        pyflex.add_box(np.array([width/2, 0.001, height/2]), center, np.array([0, 0, 0, 1]), 1)
        
        # pyflex.draw_rect(center[0], center[1], width, height, color)

    def distance(self, pos_a, pos_b):
        delta = np.array(pos_a) - np.array(pos_b)
        delta = np.linalg.norm(delta, axis=1)
        return np.sum(delta)/len(pos_a)

    #         push_x   push_y  push_ori       rot   trans_x   trans_y  deformation
    # 0     -0.23125 -0.11875 -1.570796 -1.570796  0.050000  0.000000     0.366831
    # 1     -0.23125 -0.11875 -1.570796 -1.570796  0.048296  0.012941     0.360867
    # 2     -0.23125 -0.11875 -1.570796 -1.570796  0.043301  0.025000     0.361200
    # 3     -0.23125 -0.11875 -1.570796 -1.570796  0.035355  0.035355     0.359203
    # 4     -0.23125 -0.11875 -1.570796 -1.570796  0.025000  0.043301     0.344591

    def get_center(self):
        self.get_current_corner_pos()
        return np.mean(self.cornerPos, axis=0)
    
    def get_cloth_center(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        return np.mean(pos[:, :3], axis=0)

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_covered_area = self._get_current_covered_area(
            pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            # self.action_tool.reset([cx, 0.02, cy, 0])
        pyflex.step()
        self.init_covered_area = None
        info = self._get_info()
        self.init_covered_area = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()
        return

    def _get_current_covered_area(self, pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(
            np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round(
            (offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(
            np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round(
            (offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

        # Method 2
        # grid_copy = np.zeros([100, 100])
        # for x_low, x_high, y_low, y_high in zip(slotted_x_low, slotted_x_high, slotted_y_low, slotted_y_high):
        #     grid_copy[x_low:x_high, y_low:y_high] = 1
        # assert np.allclose(grid_copy, grid)
        # return np.sum(grid_copy) * span[0] * span[1]

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area
        return r

    # @property
    # def performance_bound(self):
    #     dimx, dimy = self.current_config['ClothSize']
    #     max_area = dimx * self.cloth_particle_radius * dimy * self.cloth_particle_radius
    #     min_p = 0
    #     max_p = max_area
    #     return min_p, max_p

    def _get_info(self):
        # Duplicate of the compute reward function!
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        init_covered_area = curr_covered_area if self.init_covered_area is None else self.init_covered_area
        max_covered_area = self.get_current_config()['flatten_area']
        info = {
            'performance': curr_covered_area,
            'normalized_performance': (curr_covered_area - init_covered_area) / (max_covered_area - init_covered_area),
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info


    # def generate_a_pick_pos(self):
    #     curr_pos = pyflex.get_positions().reshape(-1, 4)
    #     num_particle = pyflex.get_n_particles()
    #     pickpoint = random.randint(0, num_particle - 1)
    #     return curr_pos[pickpoint][:3]
    

    # def pick

