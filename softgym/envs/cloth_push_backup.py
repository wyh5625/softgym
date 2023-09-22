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

class ClothPushEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_push_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.config = self.get_default_config()
        self.set_scene(self.config)

        # self.get_cached_configs_and_states(
        #     cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized
        self.cornerPos_init = None
        self.init_covered_area = None

        self.init_particles = pyflex.get_positions().reshape(-1, 4)
        self.init_pos()


        self.update_camera(self.config['camera_name'], self.config['camera_params'][self.config['camera_name']])

    def init_pos(self):
        self.get_corner_particles()
        
        self.get_current_corner_pos()
        self.cornerPos_init = self.cornerPos[:]

        self.default_pos = pyflex.get_positions().reshape(-1, 4)
        self.init_pos = self.default_pos.copy()
        self.init_pos[:, :3] -= np.mean(self.init_pos, axis=0)[:3]

        pyflex.set_positions(self.init_pos.flatten())

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
    
    def get_push_pos(self, planned_push_x, planned_push_y, planned_push_ori):
        particle_pos = pyflex.get_positions().reshape(-1, 4)[:,:3]
        # get the index of push center particle
        center_idx = self.get_touched_particle_idx(planned_push_x, planned_push_y)

        # get the index of push end particle
        toX = np.array([1, 0, 0])
        s, c = np.sin(planned_push_ori - 0), np.cos(planned_push_ori - 0)
        dist_dir = np.array([
            toX[0]*c - toX[2]*s,
            0,
            toX[0]*s + toX[2]*c
        ])
        push_center_3d = np.array([planned_push_x, 0, planned_push_y])
        endB = push_center_3d + 0.5*self.action_tool.pusher_length*dist_dir
        endA = push_center_3d - 0.5*self.action_tool.pusher_length*dist_dir

        endA_idx = self.get_touched_particle_idx(endA[0], endA[2])
        endB_idx = self.get_touched_particle_idx(endB[0], endB[2])

        # Use the two end points of pusher to find the orientation
        push_ori = self.action_tool.find_orientation(particle_pos[endA_idx,[0,2]], particle_pos[endB_idx,[0,2]])
        push_x = particle_pos[center_idx][0]
        push_y = particle_pos[center_idx][2]
        
        return push_x, push_y, push_ori
        

    # def get_current_ori_change(self):
    #     self.get_current_corner_pos()
    #     newPos = self.cornerPos[:]
    #     # Choose corresponding points
    #     A, B = self.cornerPos_init[1], self.cornerPos_init[2]
    #     A_prime, B_prime = newPos[1], newPos[2]

    #     # Calculate vectors and normalize
    #     u = B - A
    #     v = B_prime - A_prime
    #     u_normalized = u / np.linalg.norm(u)
    #     v_normalized = v / np.linalg.norm(v)

    #     # Calculate dot product
    #     dot_product = np.dot(u_normalized, v_normalized)

    #     # Calculate angle between vectors (in radians)
    #     angle = np.arccos(dot_product)

    #     # Calculate cross product
    #     # cross_product = np.cross(u_normalized[0,2], v_normalized[0,2])
    #     # Calculate the signed area (pseudo-cross product)
    #     signed_area = u_normalized[0] * v_normalized[2] - u_normalized[2] * v_normalized[0]


    #     # Determine the direction of rotation (clockwise or counterclockwise)
    #     rotation_direction = -1 if signed_area < 0 else 1

    #     # Convert angle to degrees
    #     # angle_degrees = np.degrees(angle)

    #     # print("Rotation angle:", angle_degrees)
    #     # print("Rotation direction:", rotation_direction)
    #     return rotation_direction*angle
    
    # def get_current_ori_change2(self, rect1, rect2):

    #     # Sample rectangle points (4 points per rectangle)
    #     rectangle1 = np.array(rect1)[:, [0,2]]
        
    #     rectangle2 = np.array(rect2)[:, [0,2]]

    #     # Calculate centroids of rectangles
    #     centroid1 = np.mean(rectangle1, axis=0)
    #     centroid2 = np.mean(rectangle2, axis=0)

    #     # Center points by subtracting centroids
    #     centered1 = rectangle1 - centroid1
    #     centered2 = rectangle2 - centroid2

    #     # Calculate covariance matrix
    #     covariance_matrix = np.dot(centered1.T, centered2)

    #     # Perform singular value decomposition
    #     U, _, Vt = np.linalg.svd(covariance_matrix)

    #     # Calculate rotation matrix
    #     rotation_matrix = np.dot(Vt.T, U.T)

    #     # Calculate angle of rotation in radians
    #     angle_radians = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    #     # Convert angle to degrees
    #     angle_degrees = np.degrees(angle_radians)

    #     # print("Rotation angle (degrees):", angle_degrees)
    #     return angle_degrees

    # def dir2ori(self, dir):
    #     # the ori is the angle between the direction and the x axis, in [-180, 180]
    #     ori = np.arctan2(dir[1], dir[0])
    #     return ori
        

    def get_corner_pose(self, cornerPos):
        self.get_current_corner_pos()

        # each corner has two sides, find the unit vector of each side, and output [(side1, side2), (side1, side2), (side1, side2), (side1, side2)]
        corner_side = []
        # get the length of first dimension of cornerPos which is a numpy array

        num = cornerPos.shape[0]
        for i in range(num):
            # the unit vector of the side
            len_left = np.linalg.norm(cornerPos[(i+1)%num] - cornerPos[i])
            len_right = np.linalg.norm(cornerPos[i-1] - cornerPos[i])

            two_side = ((cornerPos[(i+1)%num] - cornerPos[i])/len_left,
                        (cornerPos[i-1] - cornerPos[i])/len_right)
            corner_side.append(two_side)
        return corner_side


    def get_corner_side_contact_pose(self, cornerPos):
        corner_sides = self.get_corner_pose(cornerPos)
        contact_poses = []
        for i in range(cornerPos.shape[0]):

            # left dir
            # rotate corner_sides[i][0] by 90 degree about z axis

            left_center = cornerPos[i][[0,2]] + 0.5*self.action_tool.pusher_length*corner_sides[i][0][[0,2]]
            left_dir = self.rotate_vector(corner_sides[i][0][[0,2]], np.deg2rad(180))

            # right dir
            right_center = cornerPos[i][[0,2]] + 0.5*self.action_tool.pusher_length*corner_sides[i][1][[0,2]]
            right_dir = self.rotate_vector(corner_sides[i][1][[0,2]], np.deg2rad(0))

            contact_poses.append([left_center, left_dir, right_center, right_dir])
        return contact_poses
    
    def get_normal_of_push_pose(self, push_x, push_y, push_ori):
        # vector from push center to center of cloth
        v2o = np.array([0 - push_x, 0 - push_y])
        v_n_0 = self.rotate_vector(np.array([0, 1]), push_ori)
        v_n_1 = -v_n_0

        # print(v2o, v_n_0)
        # dot product of v2o and v_n_0
        dot0 = np.dot(v2o, v_n_0)
        if dot0 < 0:
            return v_n_0
        else:
            return v_n_1

    
    def rotate_vector(self, vector, angle_radians):
        # Create 2D rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])

        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector

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

    # abstract method
    def get_corner_particles(self):
        config = self.get_default_config()
        cloth_dimx, cloth_dimy = config['ClothSize']
        self.cornerIdx = []
        self.cornerIdx.append(0)
        self.cornerIdx.append(cloth_dimx-1)
        self.cornerIdx.append(cloth_dimx * cloth_dimy - 1)
        self.cornerIdx.append(cloth_dimx*(cloth_dimy-1))

        # pyflex.get_positions().reshape(-1, 4)[:, 3]
        # corners = {}

    def get_current_corner_pos(self):
        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        self.cornerPos = []
        for i in self.cornerIdx:
            self.cornerPos.append(particle_pos[i])
        self.cornerPos = np.array(self.cornerPos).reshape(-1, 3)

    def get_corners_of_pos(self, t, rot):
        translation = np.array([t[0], 0, t[1]])
        cornerPos = self.transform_particles(self.cornerPos_init, translation=translation,
                                                          angle=rot,
                                                          center=np.mean(
                                                              self.cornerPos_init),
                                                          set_position=False)
        return cornerPos
    
    def set_inter_corner(self, t, rot):
        corners_m = self.get_corners_of_pos(t, rot)

        center = t
        axis = np.array([0, -1, 0])
        rotation = Rotation.from_rotvec(rot * axis)
        quaternion = rotation.as_quat()

        box_state = np.array([center[0], 0.0001, center[1], center[0], 0.0001, center[1], quaternion[0], quaternion[1], quaternion[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2], quaternion[3]], dtype=np.float32)
        
        # pyflex.set_shape_state(box_state, self.box_id)
        pyflex.step()

        return corners_m
        
    def draw_contour(self, cornerPos):
        num = len(cornerPos)
        for i in range(num):
            center = (cornerPos[i] + cornerPos[(i+1)%num])/2
            center[1] = 0
            height = np.linalg.norm(cornerPos[i] - cornerPos[(i+1)%num])

            rot = -np.arctan2(cornerPos[i][0] - cornerPos[(i+1)%num][0], cornerPos[i][2] - cornerPos[(i+1)%num][2])

            axis = np.array([0, -1, 0])
            rotation = Rotation.from_rotvec(rot * axis)
            # Convert the Rotation object to a quaternion
            quaternion = rotation.as_quat()
            
            self.box_id = pyflex.add_box(np.array([0.001, 0.0001, height/2]), center, quaternion, 1)


    def set_target_corner(self, t, rot):
        self.get_corner_particles()
        self.get_current_corner_pos()
        translation = np.array([t[0], 0, t[1]])
        self.target_cornersPos = self.transform_particles(self.cornerPos, translation=translation,
                                                          angle=rot,
                                                          center=np.mean(
                                                              self.cornerPos),
                                                          set_position=False)
        
        center = np.mean(self.target_cornersPos, axis=0)
        # center[1] = 0
        dimx, dimy = self.current_config['ClothSize']
        self.cloth_particle_radius
        width, height = dimx * self.cloth_particle_radius, dimy * self.cloth_particle_radius
        
        # get quaternion of rotation
        # Create a Rotation object from the angle-axis representation
        axis = np.array([0, -1, 0])
        rotation = Rotation.from_rotvec(rot * axis)
        # Convert the Rotation object to a quaternion
        quaternion = rotation.as_quat()
        

        # pyflex.set_shape_color(np.array([0, 1, 0]))
        # center = (center[0], center[2])
        center[1] = 0
        # self.box_id = pyflex.add_box(np.array([width/2, 0.0001, height/2]), center, quaternion, 1)
        
        self.draw_contour(self.target_cornersPos)
        # box_state = np.array([0.5, 0.0001, 0.5, 0, 0, 0, quaternion[0], quaternion[1], quaternion[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2], quaternion[3]], dtype=np.float32)
        # pyflex.set_shape_state(box_state, box_id)
        # pyflex.step()



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
        input_file_path = "control_pants_50cm.csv"
        output_file_path = "control_pants_50cm_2.csv"

        # Read the CSV file and modify the specified columns
        modified_rows = []
        with open(input_file_path, "r") as input_file:
            reader = csv.reader(input_file)
            header = next(reader)  # Read and ignore the header row

            for row in reader:
                push_x = float(row[0]) - vector_to_add[0]
                push_y = float(row[1]) - vector_to_add[1]
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
    
    # def get_approach_vec(self):
    #     self.get_current_corner_pos()
    #     return np.mean(self.target_cornersPos, axis=0) - np.mean(self.cornerPos, axis=0)

    # def cost(self, action, cur_pos, deformation, a=10, b=0.1, c=1):
    #     self.get_current_corner_pos()
    #     # push_pos = [push_x, push_y, push_ori]
    #     # action = [trans_x, trans_y, rot]
    #     # calculate the cost of an action (one step)
    #     translation = np.array([action[0], 0, action[1]])

    #     rc = np.array([cur_pos[0], 0, cur_pos[1]])
    #     new_corners = self.transform_particles(
    #         self.cornerPos, translation=translation, angle=action[2], center=rc)
    #     # print(np.ravel(new_corners))
    #     dist_0 = self.distance(self.target_cornersPos, self.cornerPos)
    #     dist_1 = self.distance(self.target_cornersPos, new_corners)
    #     rot_error = abs(self.get_current_ori_change2(self.target_cornersPos, new_corners))
    #     # the approaching is negative of cost
    #     approaching = np.dot(translation, self.get_approach_vec())

    #     # approach = 

    #     # print('deformation')
    #     # print(deformation)
    #     # print('distance')
    #     # print(dist_1 - dist_0)
    #     return a*deformation + b*(dist_1) + c*rot_error, deformation, dist_1, np.mean(new_corners, axis=0)

    # def best_action(self):
    #     # load

    #     pass

    # def generate_env_variation(self, num_variations=2, vary_cloth_size=False):
    #     """ Generate initial states. Note: This will also change the current states! """
    #     max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
    #     # Cloth stable when all particles' vel are smaller than this
    #     stable_vel_threshold = 0.2
    #     generated_configs, generated_states = [], []
    #     default_config = self.get_default_config()
    #     default_config['flip_mesh'] = 1

    #     for i in range(num_variations):
    #         config = deepcopy(default_config)
    #         self.update_camera(
    #             config['camera_name'], config['camera_params'][config['camera_name']])
    #         if vary_cloth_size:
    #             cloth_dimx, cloth_dimy = self._sample_cloth_size()
    #             config['ClothSize'] = [cloth_dimx, cloth_dimy]
    #         else:
    #             cloth_dimx, cloth_dimy = config['ClothSize']

    #         self.set_scene(config)
    #         self.action_tool.reset([0., -1., 0., 0])
    #         pos = pyflex.get_positions().reshape(-1, 4)
    #         pos[:, :3] -= np.mean(pos, axis=0)[:3]
    #         # Take care of the table in robot case
    #         if self.action_mode in ['sawyer', 'franka']:
    #             pos[:, 1] = 0.57
    #         else:
    #             pos[:, 1] = 0.005
    #         pos[:, 3] = 1
    #         pyflex.set_positions(pos.flatten())
    #         pyflex.set_velocities(np.zeros_like(pos))
    #         for _ in range(5):  # In case if the cloth starts in the air
    #             pyflex.step()

    #         for wait_i in range(max_wait_step):
    #             pyflex.step()
    #             curr_vel = pyflex.get_velocities()
    #             if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
    #                 break

    #         center_object()
    #         # angle = (np.random.random() - 0.5) * np.pi / 2
    #         # self.rotate_particles(angle)

    #         generated_configs.append(deepcopy(config))
    #         # print('config {}: {}'.format(i, config['camera_params']))
    #         generated_states.append(deepcopy(self.get_state()))

    #         # Record the maximum flatten area
    #         generated_configs[-1]['flatten_area'] = self._set_to_flatten()
    #         print('config {}: camera params {}, flatten area: {}'.format(
    #             i, config['camera_params'], generated_configs[-1]['flatten_area']))
                

    #     return generated_configs, generated_states

    # def _set_to_flatten(self):
    #     # self._get_current_covered_area(pyflex.get_positions().reshape(-))
    #     cloth_dimx, cloth_dimz = self.get_current_config()['ClothSize']
    #     N = cloth_dimx * cloth_dimz
    #     px = np.linspace(
    #         0, cloth_dimx * self.cloth_particle_radius, cloth_dimx)
    #     py = np.linspace(
    #         0, cloth_dimz * self.cloth_particle_radius, cloth_dimz)
    #     xx, yy = np.meshgrid(px, py)
    #     new_pos = np.empty(shape=(N, 4), dtype=np.float)
    #     new_pos[:, 0] = xx.flatten()
    #     new_pos[:, 1] = self.cloth_particle_radius
    #     new_pos[:, 2] = yy.flatten()
    #     new_pos[:, 3] = 1.
    #     new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
    #     pyflex.set_positions(new_pos.flatten())
    #     return self._get_current_covered_area(new_pos)

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

    # def get_picked_particle(self):
    #     pps = np.ones(shape=self.action_tool.num_picker) * - \
    #         1  # -1 means no particles picked
    #     for i, pp in enumerate(self.action_tool.picked_particles):
    #         if pp is not None:
    #             pps[i] = pp
    #     return pps

    # def get_bounding_points(self):
    #     curr_pos = pyflex.get_positions().reshape(-1, 4)

    # def sample_points_on_cloth(self, dw, dh):
    #     pass

    # def project_action(self, planned_action):
        

    #     # get the curr pos of the closest point on the fabric

    #     # get the orientation change relative to the initial orientation of fabric
    #     pass

    # def particle_virtual_move(self, translation, rot, center):
    #     curr_pos = pyflex.get_positions().reshape(-1, 4)
    #     rotate_particles(self, rot, center)

    # def generate_push_actions(self, start_pos, target_pos, circular_rate=0.0, center_side=1, path_side=1, waypoint_size=1):
    #     # circular_rate is [0, 1], radius of path R = R_min / circular_rate(or cos(theta)), larger circular_rate, smaller R
    #     # center_side = -1/1, which means the center of circle is on the LHS/RHS of straight path
    #     # path_side = -1/1, which means the left/right part of circular path is chosen.
    #     waypoints = []
    #     actions = []
    #     # straight line
    #     if circular_rate == 0:
    #         diff = target_pos - start_pos
    #         if diff[3] > np.pi:
    #             diff[3] -= 2*np.pi
    #         elif diff[3] < -np.pi:
    #             diff[3] += 2*np.pi
    #         step = diff/waypoint_size

    #         for i in range(waypoint_size):
    #             waypoint = start_pos + (i+1)*step
    #             waypoints.append(waypoint)
    #             action = np.array([*waypoint, 1.0])
    #             actions.append(action)
    #     else:
    #         r_min = np.linalg.norm(target_pos - start_pos)/2
    #         r_path = r_min / circular_rate
    #         s = np.sqrt(1 - circular_rate**2)
    #         mid2center = s*r_path

    #         # find the rotation center
    #         mid_x = (start_pos + target_pos)/2
    #         move_l = target_pos - start_pos
    #         move_l_dir = move_l / np.linalg.norm(move_l)
    #         mid2center_dir = np.array(
    #             [move_l_dir[2], move_l_dir[1], -move_l_dir[0]])
    #         center = mid_x + mid2center * center_side * mid2center_dir

    #         # find the circular path
    #         angle = 2*(np.pi/2 - np.arccos(circular_rate))
    #         if center_side*path_side > 0:
    #             angle = 2*np.pi - angle
    #         if path_side == -1:
    #             angle = -angle
    #         step = angle/waypoint_size
    #         start_pos -= center
    #         print("Rotate angle:" + str(angle))
    #         # print(angle)
    #         for i in range(waypoint_size):
    #             new_pos = start_pos.copy()
    #             da = 0 + (i+1)*step
    #             new_pos[0] = (np.cos(da) * start_pos[0] -
    #                           np.sin(da) * start_pos[2])
    #             new_pos[2] = (np.sin(da) * start_pos[0] +
    #                           np.cos(da) * start_pos[2])
    #             waypoint = new_pos+center
    #             waypoints.append(waypoint)
    #             action = np.array([*waypoint, 1.0])
    #             actions.append(action)

    #     return actions

    # def generate_a_pick_pos(self):
    #     curr_pos = pyflex.get_positions().reshape(-1, 4)
    #     num_particle = pyflex.get_n_particles()
    #     pickpoint = random.randint(0, num_particle - 1)
    #     return curr_pos[pickpoint][:3]

    def init_pusher(self, pos):
        # pos: [x, y, z, rot]
        # action = np.array([*pos, 0.0])
        # curr_pos = pyflex.get_positions()
        # cx, cy = self._get_center_point(curr_pos)
        # self.action_tool.reset([cx, 0.02, cy, np.pi/6])
        self.action_tool.reset(pos)

    def set_pusher(self, pos):
        self.action_tool.unpick()
        self.action_tool.set_pos(pos)

    def get_pusher_pos(self):
        pose = [*self.action_tool.pos, self.action_tool.ori]
        return pose

    # it's based on the default configuration of scene
    # def test_sample(self, picker_pos, action, record=False, img_size=None, save_video_dir=None):
    #     # init scene
    #     # default_config = self.get_default_config()
    #     # self.reset()
    #     self._set_to_flat()
    #     # self.transform_particles(angle=np.pi, center=np.array([0,0,0]), set_position=True)
        
    #     self.set_pusher(picker_pos)

    #     # get virtual rigid position after action
    #     pos = pyflex.get_positions().reshape(-1, 4)
    #     rigid_pos = self.transform_particles(pos, translation=(action[:3]-picker_pos[:3]),
    #                                          angle=(action[3]-picker_pos[3]),
    #                                          center=picker_pos[:3],
    #                                          set_position=False)
    #     # actions = env.generate_push_actions(init_pusher_pos[:4], push_action[:4], waypoint_size=50)
    #     # frames = [self.get_image(img_size, img_size)]

    #     # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
    #     # intermediate frames. Only use this option for visualization as it increases computation.

    #     # drop_action = [0.5, 0, 0, np.pi/6, 1.0]
    #     # Divide the action by waypoint_size to make smooth transformation when displacement is very small and rotation is large
    #     # actions = self.generate_push_actions(picker_pos[:4], action[:4], waypoint_size=3)
    #     # print(actions)
    #     actions = [action]
    #     for action in actions:
    #         _, _, _, info = self.step(
    #             action, record_continuous_video=record, img_size=img_size)
    #         # if record:
    #         #     frames.extend(info['flex_env_recorded_frames'])

    #     max_wait_step = 300
    #     stable_vel_threshold = 0.5

    #     # wait to stablize cloth
    #     # for _ in range(max_wait_step):
    #     #     pyflex.step()
    #     #     curr_vel = pyflex.get_velocities()
    #     #     if np.alltrue(curr_vel < stable_vel_threshold):
    #     #         break

    #     # visualization
    #     # if record:
    #     #     save_name = osp.join(save_video_dir, 'ClothPushPPP' + '.gif')
    #     #     save_numpy_as_gif(np.array(frames), save_name)
    #     #     print('Video generated and save to {}'.format(save_name))

    #     # get the real position after action
    #     now_pos = pyflex.get_positions().reshape(-1, 4)
    #     error = now_pos[:, :3] - rigid_pos[:, :3]
    #     error_norm = np.linalg.norm(error, axis=1)
    #     tot_errors = np.sum(error_norm)

    #     num_particle = pyflex.get_n_particles()
    #     avg_error = tot_errors/num_particle

    #     return avg_error


    # it's based on the default configuration of scene
    def push(self, action, record=False, img_size=None, save_video_dir=None):
        # [x, y, z, rot, pick/drop]
        # init scene
        # default_config = self.get_default_config()

        actions = [action]
        for action in actions:
            print('action: ', action)
            _, _, _, info = self.step(
                action, record_continuous_video=record, img_size=img_size)
            # if record:
            #     frames.extend(info['flex_env_recorded_frames'])

        # max_wait_step = 300
        # stable_vel_threshold = 0.5

        # # wait to stablize cloth
        # for _ in range(max_wait_step):
        #     pyflex.step()
        #     curr_vel = pyflex.get_velocities()
        #     if np.alltrue(curr_vel < stable_vel_threshold):
        #         break

