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
from RRT_Star_MG import coordinate_to_matrix, matrix_to_coordinate
import ot
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

class ClothPushEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_push_init_states.pkl', constraints=False, **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)

        # self.get_cached_configs_and_states(
        #     cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized
        self.cornerPos_init = None
        self.init_covered_area = None
        self.constraints = constraints
        self.surface_height = 0

        
        self.table_size = (0, 0, 0)
        self.dist2table = 0

        

        if constraints:
            self.table_size = (1.4, 0.0001, 0.8)
            self.dist2table = 0.42
            self.surface_height = self.table_size[1]

        self.table_center = (0, -(self.dist2table + self.table_size[2]/2))
        self.init()

        self.inter_box_ids = []
        self.target_box_ids = []

        # max reachable length: 1.1
        max_reachable = 1.1

    def set_start_state(self, t, r):
        translation = np.array([t[0], t[1], t[2]])
        start_pos = self.transform_particles(self.init_pos, 
                            translation=translation, 
                            angle=r, 
                            center=np.mean(self.cornerPos_init, axis=0)[:3], 
                            set_position=True)
        
        self.init_pos = start_pos.copy()
        
        self.get_current_corner_pos()
        self.cornerPos_init = self.cornerPos[:]
        
    def set_init_pos(self):
        self.init_pos = pyflex.get_positions().reshape(-1, 4)

    def camera_on_cloth(self):
        center = self.get_center()
        camera_pos = [center[0], 2.3, center[2]]
        self.set_camera_pos(camera_pos)

    def init(self):
        self.config = self.get_default_config()
        self.set_scene(self.config)

        self.update_camera(self.config['camera_name'], self.config['camera_params'][self.config['camera_name']])

        self.init_particles = pyflex.get_positions().reshape(-1, 4)
        self.get_corner_particles()


        self.default_pos = pyflex.get_positions().reshape(-1, 4)
        self.init_pos = self.default_pos.copy()

        self.get_current_corner_pos()
        mean = np.mean(self.cornerPos, axis=0)[:3]
        # # mean[0] = 10
        self.init_pos[:, :3] -= mean
        pyflex.set_positions(self.init_pos.flatten())
        

        # self.init_pos[:, :3] += np.array([0, 2*self.surface_height, 0])


        if self.constraints:
            colors = [
                [0, 0, 0],
                [128, 128, 128],
                [160, 160, 160],
                [140, 140, 140],
                [ 90, 115, 165],
                [255, 255, 255]
            ]

            # self.draw_workspace(1.4, 1)

            # add a table box in the scene
            box_id = pyflex.add_box(np.array([self.table_size[0]/2, self.table_size[1]/2, self.table_size[2]/2]), np.array([self.table_center[0], self.table_size[1]/2, self.table_center[1]]), np.array([0, 0, 0, 1]), 1)
            # state = pyflex.get_shape_states()[box_id]
            # box_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
            # pyflex.set_shape_state(box_state, box_id)
            pyflex.set_shape_color(box_id, np.array(colors[0])/255.0)

            # add a background box
            # box_id = pyflex.add_box(np.array([self.table_size[0]/2, self.table_size[1]/2, self.table_size[2]/2])*2, np.array([0, 1, 0]), np.array([self.table_center[0], self.table_size[1]/2, self.table_center[1]]), np.array([0, 0, 0, 1]), 1)
            # pyflex.set_shape_color(box_id, np.array(colors[2])/255.0)

            # add robot arm base
            # robot_arm_pos = np.array([0.0, 0.0, 0.0])
            # robot_arm_size = np.array([0.1/2, 0.02, 0.1/2])
            # box_id = pyflex.add_box(robot_arm_size, robot_arm_pos, np.array([0, 0, 0, 1]), 0)
            # pyflex.set_shape_color(box_id, np.array(colors[0])/255.0)
            # print("robot arm box id: ", box_id)

        #     # print("num of particle: ", pyflex.get_n_particles())
        #     # print("num of shape: ", pyflex.get_n_shapes())
            
        #     # # print corner particle index
        #     # print("corner particle index: ", self.cornerIdx)

        # for i in range(100):
        #     pyflex.step()

        # pyflex.set_positions(self.init_pos.flatten())

        # for i in range(100):
        #     pyflex.step()
        
        self.get_current_corner_pos()
        self.cornerPos_init = self.cornerPos[:]

        # self.cornerCP_init = self.get_corner_side_contact_pose(self.cornerPos_init)

        # # evenly sample 1/10 of the particles idx
        # num_particles = self.init_pos.shape[0]
        # self.sampled_particles_idx = np.random.choice(num_particles, int(num_particles / 10), replace=False)

    def set_init_pos(self):
        pyflex.set_positions(self.init_pos.flatten())
        pyflex.step()

    def set_camera_pos(self, camera_pos):
        # params = pyflex.get_camera_params()
        # params[0] = camera_pos[0]
        # params[1] = camera_pos[1]
        # params[2] = camera_pos[2]

        self.config['camera_params'][self.config['camera_name']]['pos'] = [camera_pos[0], camera_pos[1], camera_pos[2]]

        self.update_camera(self.config['camera_name'], self.config['camera_params'][self.config['camera_name']])

        # pyflex.set_camera_params(params)

    def draw_workspace(self, width, height):
        # draw the workspace
        # draw the workspace
        # colors = [
        #     [128, 128, 128],
        #     [160, 160, 160]
        #     # [1, 0, 0]
        # ]

        # get the four corners of the workspace
        corners = np.array([
            [-width/2, 0, -height/2],
            [width/2, 0, -height/2],
            [width/2, 0, height/2],
            [-width/2, 0, height/2]
        ])

        for i in range(4):
            center = (corners[i] + corners[(i+1)%4])/2
            height = np.linalg.norm(corners[i] - corners[(i+1)%4])

            # rot = -np.arctan2(corners[(i+1)%4][2] - corners[i][2], corners[(i+1)%4][0] - corners[i][0])
            rot = -np.arctan2(corners[i][0] - corners[(i+1)%4][0], corners[i][2] - corners[(i+1)%4][2])
            axis = np.array([0, -1, 0])
            rotation = Rotation.from_rotvec(rot * axis)
            # Convert the Rotation object to a quaternion
            quaternion = rotation.as_quat()

            box_id = pyflex.add_box(np.array([0.005, 0.0001, height/2]), center, quaternion, 1)
            pyflex.set_shape_color(box_id, np.array([255, 255, 255])/255.0)

            

        
        



    def hide_pusher(self):
        ori = self.action_tool.ori
        pos = self.action_tool.pos[:]
        self.action_tool.set_pos([0,-0.1,0,0])

        pusher_pos = [*pos, ori]
        return pusher_pos

    def get_touched_particle_idx(self, centered_x, centered_y):
        # given a planned action's push_x and push_y, the index of the nearest particle of fabric when its in the intial position is found
        picker_pos = np.array([centered_x, 0, centered_y]).reshape(1, 3)

        # print("picker_pos", picker_pos)

        # set pusher to the picker_pos
        # self.action_tool.set_pos(picker_pos[0])
        # record the frame
        # for i in range(10000):
        #     pyflex.step()
        # self.shoot_frame()
        # save last frame to file

        # find the index of closest point on the fabric
        dists = scipy.spatial.distance.cdist(picker_pos, self.init_pos[:, :3].reshape((-1, 3)))
        idx_dists = np.hstack([np.arange(self.init_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
        # print("idx_dists", idx_dists)
        mask = dists.flatten() <= self.action_tool.picker_threshold + self.action_tool.picker_radius + self.cloth_particle_radius + 0.8
        idx_dists = idx_dists[mask, :].reshape((-1, 2))

        

        pick_id, pick_dist = None, None
        if idx_dists.shape[0] > 0:
            for j in range(idx_dists.shape[0]):
                if pick_id is None or idx_dists[j, 1] < pick_dist:
                    pick_id = idx_dists[j, 0]
                    pick_dist = idx_dists[j, 1]

        
        return int(pick_id)

    # define a set of possible pusher poses (x, z, theta)
    def generate_pusher_poses(env):
        # generate 12 pusher poses, 3 x 4, each side with 3 parallel pusher poses with the same orientation of the side, 
        # and attached to the border of the cloth, one of which is the center of the side, for the other two, their position 
        # should be the shift of the center position until one end touch the corner of the cloth, they should touch different corners in the same side.
        
        # get the length of the pusher
        pusher_length = env.action_tool.pusher_length
        # get the radius of the cloth particles
        cloth_particle_radius = env.cloth_particle_radius
        # get the size of the cloth
        dimx, dimy = env.get_current_config()['ClothSize']

        
        # find the index of the cloth particles closet to the center of the pusher pose on the upper side, and get the pusher orientation
        up_left = (int(0.5*pusher_length/cloth_particle_radius), np.pi)
        up_middle = (int(dimx/2), np.pi)
        up_right = (dimx -5 -int(0.5*pusher_length/cloth_particle_radius), np.pi)

        left_up = (int(0.5*pusher_length/cloth_particle_radius)*dimx, np.pi/2)
        left_middle = (int(dimy/2)*dimx, np.pi/2)
        left_down = ((dimy - 5 - int(0.5*pusher_length/cloth_particle_radius))*dimx, np.pi/2)

        down_left = (dimx*(dimy-1) + int(0.5*pusher_length/cloth_particle_radius), 0)
        down_middle = (dimx*(dimy-1) + int(dimx/2), 0)
        down_right = (dimx*dimy - 5-int(0.5*pusher_length/cloth_particle_radius), 0)

        right_up = (int(0.5*pusher_length/cloth_particle_radius)*dimx + dimx-1, -np.pi/2)
        right_middle = (int(dimy/2)*dimx + dimx-1, -np.pi/2)
        right_down = ((dimy-5-int(0.5*pusher_length/cloth_particle_radius))*dimx -1, -np.pi/2)

        pusher_pose = [up_left, up_middle, up_right, 
                            left_up, left_middle, left_down, 
                            down_left, down_middle, down_right, 
                            right_up, right_middle, right_down]
        
        return pusher_pose

    def generate_pusher_poses2(env):
        # generate 12 pusher poses, 3 x 4, each side with 3 parallel pusher poses with the same orientation of the side, 
        # and attached to the border of the cloth, one of which is the center of the side, for the other two, their position 
        # should be the shift of the center position until one end touch the corner of the cloth, they should touch different corners in the same side.
        
        # get the length of the pusher
        pusher_length = env.action_tool.pusher_length
        # get the radius of the cloth particles
        cloth_particle_radius = env.cloth_particle_radius
        # get the size of the cloth
        # dimx, dimy = env.get_current_config()['ClothSize']
        dimx, dimy = 0.6, 0.36875
        
        # find the index of the cloth particles closet to the center of the pusher pose on the upper side, and get the pusher orientation
        up_left = ((-dimx/2 + 0.5*pusher_length, -dimy/2), np.pi)
        up_middle = ((0, -dimy/2), np.pi)
        up_right = ((dimx/2 - 0.5*pusher_length, -dimy/2), np.pi)

        left_up = ((-dimx/2, -dimy/2 + 0.5*pusher_length), np.pi/2)
        left_middle = ((-dimx/2, 0), np.pi/2)
        left_down = ((-dimx/2, dimy - 0.5*pusher_length), np.pi/2)

        down_left = ((-dimx/2 + 0.5*pusher_length, dimy/2), 0)
        down_middle = ((0, dimy/2), 0)
        down_right = ((dimx/2 - 0.5*pusher_length, dimy/2), 0)

        right_up = ((dimx/2, -dimy/2 + 0.5*pusher_length), -np.pi/2)
        right_middle = ((dimx/2, 0), -np.pi/2)
        right_down = ((dimx/2, dimy/2 - 0.5*pusher_length), -np.pi/2)

        pusher_pose = [up_left, up_middle, up_right, 
                       right_up, right_middle, right_down,
                       down_right, down_middle, down_left,
                       left_down, left_middle, left_up]
        
        return pusher_pose
    
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
        endB_idx  = self.get_touched_particle_idx(endB[0], endB[2])

        # Use the two end points of pusher to find the orientation
        push_ori = self.action_tool.find_orientation(particle_pos[endA_idx,[0,2]], particle_pos[endB_idx,[0,2]])
        push_x = particle_pos[center_idx][0]
        push_y = particle_pos[center_idx][2]
        
        return push_x, push_y, push_ori
    
    # given pose of cloth, and the cp on the cloth, find the pusher pose
    def get_push_pos_rigid(self, t, r, cp):
        M_c = coordinate_to_matrix(t[0], t[1], r)
        M_p = coordinate_to_matrix(cp[0], cp[1], cp[2])

        M = np.dot(M_c, M_p)

        (push_x, push_y, push_ori) = matrix_to_coordinate(M)

        
        return np.array([push_x, push_y, push_ori])

    def dir2ori(self, dir):
        # the ori is the angle between the direction and the x axis, in [-180, 180]
        ori = np.arctan2(dir[1], dir[0])
        return ori
        

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
            left_ori = self.dir2ori(left_dir)

            # right dir
            right_center = cornerPos[i][[0,2]] + 0.5*self.action_tool.pusher_length*corner_sides[i][1][[0,2]]
            right_dir = self.rotate_vector(corner_sides[i][1][[0,2]], np.deg2rad(0))
            right_ori = self.dir2ori(right_dir)

            contact_poses.append([left_center, left_ori, right_center, right_ori])
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
            pyflex.step()

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
                                                          center=np.mean(self.cornerPos_init, axis=0)[:3],
                                                          set_position=False)
        return cornerPos
    
    def set_inter_corner(self, t, rot, draw=False, color=[255, 0, 0]):
        corners_m = self.get_corners_of_pos(t, rot)

        if draw:
            self.draw_inter(t, rot, color=color)
            self.shoot_frame()

        pyflex.step()

        return corners_m
    
    # get pose of cloth
    def align_inter_with_pusher(self, pose, cp):
        # get matrix of pusher pose
        M_p = coordinate_to_matrix(pose[0], pose[1], pose[2])
        # get transform matrix of cp
        M_cp = coordinate_to_matrix(cp[0], cp[1], cp[2])
        M_cp_ = np.linalg.inv(M_cp)
        S = np.dot(M_p, M_cp_)

        # get translation and rotation of S
        (x, y, theta) = matrix_to_coordinate(S)

        print("pose: ", pose)
        print("cp: ", cp)

        return np.array([x, y, theta])

        

    def draw_inter(self, t, r, color=[255, 0, 0]):
        cnr_inter = self.get_corners_of_pos(t, r)
        num = len(cnr_inter)
        inter_box_initialized = True
        # check if the inter_box is 
        if not self.inter_box_ids:
            inter_box_initialized = False

        for i in range(num):
            center = (cnr_inter[i] + cnr_inter[(i+1)%num])/2
            center[1] = 0
            height = np.linalg.norm(cnr_inter[i] - cnr_inter[(i+1)%num])
            print("height: ", height)


            rot = -np.arctan2(cnr_inter[i][0] - cnr_inter[(i+1)%num][0], cnr_inter[i][2] - cnr_inter[(i+1)%num][2])

            # rot -= r
            # center[[0,2]] += t

            axis = np.array([0, -1, 0])
            rotation = Rotation.from_rotvec(rot * axis)
            # Convert the Rotation object to a quaternion
            quaternion = rotation.as_quat()

            if not inter_box_initialized:
                box_id = pyflex.add_box(np.array([0.002, 0.0004, height/2]), center, quaternion, 1)
                pyflex.set_shape_color(box_id, np.array(color)/255.0)
                self.inter_box_ids.append(box_id)
            else:
                box_state = np.array([center[0], 0.0001, center[2], center[0], 0.0001, center[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2], quaternion[3]], dtype=np.float32)
                pyflex.set_shape_state(box_state, self.inter_box_ids[i])
                pyflex.set_shape_color(self.inter_box_ids[i], np.array(color)/255.0)
                pyflex.step()

    def remove_inter(self):
        # for box_id in self.inter_box_ids:
        #     pyflex.pop_box(box_id)
        pyflex.pop_box(len(self.inter_box_ids))
        self.inter_box_ids = []
        pyflex.step()
        
    def draw_target(self, cornerPos):
        target_box_initialized = True
        if not self.target_box_ids:
            target_box_initialized = False

        num = len(cornerPos)
        for i in range(num):
            center = (cornerPos[i] + cornerPos[(i+1)%num])/2
            # print("center: ", center)
            center[1] = self.surface_height
            height = np.linalg.norm(cornerPos[i] - cornerPos[(i+1)%num])

            rot = -np.arctan2(cornerPos[i][0] - cornerPos[(i+1)%num][0], cornerPos[i][2] - cornerPos[(i+1)%num][2])

            axis = np.array([0, -1, 0])
            rotation = Rotation.from_rotvec(rot * axis)
            # Convert the Rotation object to a quaternion
            quaternion = rotation.as_quat()

            if not target_box_initialized:
                box_id = pyflex.add_box(np.array([0.005, 0.0004, height/2]), center, quaternion, 1)
                pyflex.set_shape_color(box_id, np.array([0, 153, 0])/255.0)
                self.target_box_ids.append(box_id)
            else:
                box_state = np.array([center[0], center[1], center[2], center[0], center[1], center[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2], quaternion[3]], dtype=np.float32)
                pyflex.set_shape_state(box_state, self.target_box_ids[i])
                pyflex.step()
        # draw the bar of rectangle cloth on the shape
        if num == 4:
            # center of the bar is the center of line connecting 2/8 of top and bottom sides
            ratio = 1.0/8
            endA = (1-ratio)*cornerPos[0] + ratio*cornerPos[1]
            endB = (1-ratio)*cornerPos[3] + ratio*cornerPos[2]
            center = (endA + endB)/2
            center[1] -= 0.001

            rot = -np.arctan2(endA[0] - endB[0], endA[2] - endB[2])

            axis = np.array([0, -1, 0])
            rotation = Rotation.from_rotvec(rot * axis)
            # Convert the Rotation object to a quaternion
            quaternion = rotation.as_quat()

            if not target_box_initialized:
                box_id = pyflex.add_box(np.array([0.005, 0.0001, np.linalg.norm(endA - endB)/2]), center, quaternion, 1)
                pyflex.set_shape_color(box_id, np.array([0, 153, 0])/255.0)
                self.target_box_ids.append(box_id)
            else:
                box_state = np.array([center[0], center[1], center[2], center[0], 0.0001, center[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2], quaternion[3]], dtype=np.float32)
                pyflex.set_shape_state(box_state, self.target_box_ids[-1])
                pyflex.step()
        

    def get_center_offset(self):
        real_center = np.mean(self.cornerPos, axis=0)
        # get bounding box
        min_x = np.min(self.cornerPos[:, 0])
        max_x = np.max(self.cornerPos[:, 0])
        min_z = np.min(self.cornerPos[:, 2])
        max_z = np.max(self.cornerPos[:, 2])
        # center of bounding box
        rrt_center = np.array([(min_x + max_x)/2, 0, (min_z + max_z)/2])
        # offset of real center to rrt center
        offset = real_center - rrt_center
        return offset
        

    def set_target_corner(self, t, rot, draw_target=False):
        # print("set target corner: ", t, rot)
        # self.get_current_corner_pos()
        # print("Center: ", np.mean(self.cornerPos_init, axis=0))
        
        translation = np.array([t[0], 0, t[1]])
        self.target_cornersPos = self.transform_particles(self.cornerPos_init, translation=translation,
                                                          angle=rot,
                                                          center=np.mean(
                                                              self.cornerPos_init, axis=0),
                                                          set_position=False)
        
        # center = np.mean(self.target_cornersPos, axis=0)
        # # center[1] = 0
        # dimx, dimy = self.current_config['ClothSize']
        # self.cloth_particle_radius
        # width, height = dimx * self.cloth_particle_radius, dimy * self.cloth_particle_radius
        
        # # get quaternion of rotation
        # # Create a Rotation object from the angle-axis representation
        # axis = np.array([0, -1, 0])
        # rotation = Rotation.from_rotvec(rot * axis)
        # # Convert the Rotation object to a quaternion
        # quaternion = rotation.as_quat()
        

        # pyflex.set_shape_color(np.array([0, 1, 0]))
        # center = (center[0], center[2])
        # center[1] = 0
        # self.box_id = pyflex.add_box(np.array([width/2, 0.0001, height/2]), center, quaternion, 1)
        
        if draw_target:
            self.draw_target(self.target_cornersPos)

        # print("center: ", np.mean(self.init_pos, axis=0)[:3])

        # target particle pos
        self.t_pos = self.transform_particles(self.init_pos, 
                            translation=translation, 
                            angle=rot, 
                            center=np.mean(self.cornerPos_init, axis=0)[:3], 
                            set_position=False)
        



    def set_working_area(self, width, height, center, color=np.array([1, 1, 1])):
        pyflex.set_shape_color(color)
        pyflex.add_box(np.array([width/2, 0.001, height/2]), center, np.array([0, 0, 0, 1]), 1)
        
        # pyflex.draw_rect(center[0], center[1], width, height, color)

    def distance(self, pos_a, pos_b):
        delta = np.array(pos_a) - np.array(pos_b)
        delta = np.linalg.norm(delta, axis=1)
        return np.sum(delta)/len(pos_a)

    # pos_a is assumed to be the reference
    def CD(self, pos_a, pos_b):
        pos_a_sampled = pos_a[self.sampled_particles_idx]
        pos_b_sampled = pos_b[self.sampled_particles_idx]


        # Assume you have two sets of points: source_points and target_points
        # source_points = np.array([[x1, y1], [x2, y2], ...])  # Your source points
        # target_points = np.array([[x1, y1], [x2, y2], ...])  # Your target points

        # Apply Procrustes analysis to align the source points with the target points
        aligned_source_points, _, disparity = procrustes(pos_a_sampled, pos_b_sampled)
        # print("Aligned Source Points:", aligned_source_points)

        # Calculate the Chamfer distance between the aligned source points and target points
        # chamfer_distance = np.sum(np.min(np.linalg.norm(aligned_source_points - pos_a_sampled, axis=1)))
        # distances = np.linalg.norm(aligned_source_points - pos_a_sampled, axis=2)
        # chamfer_distance = np.sum(np.min(distances, axis=1)) + np.sum(np.min(distances, axis=0))
        print("Procrustes disparity:", disparity)


        return disparity


    def EMD(self, pos_a, pos_b):
        # create weights, a number of 1s equal to the number of particles
        pos_a_sampled = pos_a[self.sampled_particles_idx]
        pos_b_sampled = pos_b[self.sampled_particles_idx]
        weights_a = np.ones(len(pos_a_sampled))
        weights_b = np.ones(len(pos_b_sampled))
        # Create cost matrix based on pairwise distances
        cost_matrix = ot.dist(pos_a_sampled, pos_b_sampled)

        # Calculate the EMD using the Sinkhorn algorithm
        emd = ot.emd2(weights_a, weights_b, cost_matrix)

        return emd


    #         push_x   push_y  push_ori       rot   trans_x   trans_y  deformation
    # 0     -0.23125 -0.11875 -1.570796 -1.570796  0.050000  0.000000     0.366831
    # 1     -0.23125 -0.11875 -1.570796 -1.570796  0.048296  0.012941     0.360867
    # 2     -0.23125 -0.11875 -1.570796 -1.570796  0.043301  0.025000     0.361200
    # 3     -0.23125 -0.11875 -1.570796 -1.570796  0.035355  0.035355     0.359203
    # 4     -0.23125 -0.11875 -1.570796 -1.570796  0.025000  0.043301     0.344591

    def get_center(self):
        self.get_current_corner_pos()
        return np.mean(self.cornerPos, axis=0)[:3]
    
    

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

    

    def init_pusher(self, pos):
        self.action_tool.reset(pos)

    def set_pusher(self, pos):
        self.action_tool.unpick()
        self.action_tool.set_pos(pos)

    def get_pusher_pos(self):
        pose = [*self.action_tool.pos, self.action_tool.ori]
        return pose

    # it's based on the default configuration of scene
    def push(self, action, record=False, img_size=None, save_video_dir=None):
        # [x, y, z, rot, pick/drop]
        # init scene
        # default_config = self.get_default_config()
        actions = [action]
        for action in actions:
            _, _, _, info = self.step(
                action, record_continuous_video=record, img_size=img_size)


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