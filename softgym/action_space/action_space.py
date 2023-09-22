import abc
import numpy as np
from gym.spaces import Box
from softgym.utils.misc import rotation_2d_around_center, extend_along_center
import pyflex
import scipy.spatial
import math


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ Step funciton to change the action space states. Does not call pyflex.step() """


class Picker(ActionToolBase):
    def __init__(self, num_picker=1, picker_radius=0.05, init_pos=(0., -0.1, 0.), picker_threshold=0.005, particle_radius=0.05,
                 picker_low=(-0.4, 0.2, -0.4), picker_high=(0.4, 0.5, 0.4), init_particle_pos=None, spring_coef=1.2, **kwargs):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        self.spring_coef = spring_coef  # Prevent picker to drag two particles too far away

        space_low = np.array([-0.1, -0.1, -0.1, 0] * self.num_picker) * 0.1  # [dx, dy, dz, [0, 1]]
        space_high = np.array([0.1, 0.1, 0.1, 10] * self.num_picker) * 0.1
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    def update_picker_boundary(self, picker_low, picker_high):
        self.picker_low, self.picker_high = np.array(picker_low).copy(), np.array(picker_high).copy()

    def visualize_picker_boundary(self):
        halfEdge = np.array(self.picker_high - self.picker_low) / 2.
        center = np.array(self.picker_high + self.picker_low) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)

    def _apply_picker_boundary(self, picker_pos):
        clipped_picker_pos = picker_pos.copy()
        for i in range(3):
            clipped_picker_pos[i] = np.clip(picker_pos[i], self.picker_low[i] + self.picker_radius, self.picker_high[i] - self.picker_radius)
        return clipped_picker_pos

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        for i in (0, 2):
            offset = center[i] - (self.picker_high[i] + self.picker_low[i]) / 2.
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            # get id of add_sphereadd_sphere
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack([centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # pyflex.step() # Remove this as having an additional step here may affect the cloth drop env
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]
        # print('inv_mass_shape after reset:', self.particle_inv_mass.shape)

    @staticmethod
    def _get_pos(ids=None):
        """ Get the current pos of the pickers and the particles, along with the inverse mass of each particle """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        if ids is None:
            return picker_pos[:, :3], particle_pos
        else:
            return picker_pos[ids, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos, ids=None):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        if ids is None:
            shape_states[:, 3:6] = shape_states[:, :3]
            shape_states[:, :3] = picker_pos
        else:
            shape_states[ids, 3:6] = shape_states[ids, :3]
            shape_states[ids, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    @staticmethod
    def set_picker_pos(picker_pos, ids=None):
        """ Caution! Should only be called during the reset of the environment. Used only for cloth drop environment. """
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        if ids is None:
            shape_states[:, 3:6] = picker_pos
            shape_states[:, :3] = picker_pos
        else:
            shape_states[ids, 3:6] = picker_pos
            shape_states[ids, :3] = picker_pos

        pyflex.set_shape_states(shape_states)

    def step(self, action, ids=None):
        """ action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one, for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        action = np.reshape(action, [-1, 4])
        # pick_flag = np.random.random(self.num_picker) < action[:, 3]
        pick_flag = action[:, 3] > 0.5
        picker_pos, particle_pos = self._get_pos(ids)
        new_picker_pos, new_particle_pos = picker_pos.copy(), particle_pos.copy()

        # Un-pick the particles
        # print('check pick id:', self.picked_particles, new_particle_pos.shape, self.particle_inv_mass.shape)
        for i in range(self.num_picker):
            if not pick_flag[i] and self.picked_particles[i] is not None:
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]  # Revert the mass
                self.picked_particles[i] = None

        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i, :] = self._apply_picker_boundary(picker_pos[i, :] + action[i, :3])
            if pick_flag[i]:
                if self.picked_particles[i] is None:  # No particle is currently picked and thus need to select a particle to pick
                    dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                    idx_dists = np.hstack([np.arange(particle_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
                    mask = dists.flatten() <= self.picker_threshold + self.picker_radius + self.particle_radius
                    idx_dists = idx_dists[mask, :].reshape((-1, 2))
                    if idx_dists.shape[0] > 0:
                        pick_id, pick_dist = None, None
                        for j in range(idx_dists.shape[0]):
                            if idx_dists[j, 0] not in self.picked_particles and (pick_id is None or idx_dists[j, 1] < pick_dist):
                                pick_id = idx_dists[j, 0]
                                pick_dist = idx_dists[j, 1]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)

                if self.picked_particles[i] is not None:
                    # TODO The position of the particle needs to be updated such that it is close to the picker particle
                    new_particle_pos[self.picked_particles[i], :3] = particle_pos[self.picked_particles[i], :3] + new_picker_pos[i, :] - picker_pos[i,:]
                    new_particle_pos[self.picked_particles[i], 1] = 0.001
                    new_particle_pos[self.picked_particles[i], 3] = 0  # Set the mass to infinity

        # check for e.g., rope, the picker is not dragging the particles too far away that violates the actual physicals constraints.
        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i)

            l = len(picked_particle_idices)
            for i in range(l):
                for j in range(i + 1, l):
                    init_distance = np.linalg.norm(self.init_particle_pos[picked_particle_idices[i], :3] -
                                                   self.init_particle_pos[picked_particle_idices[j], :3])
                    now_distance = np.linalg.norm(new_particle_pos[picked_particle_idices[i], :3] -
                                                  new_particle_pos[picked_particle_idices[j], :3])
                    if now_distance >= init_distance * self.spring_coef:  # if dragged too long, make the action has no effect; revert it
                        new_picker_pos[active_picker_indices[i], :] = picker_pos[active_picker_indices[i], :].copy()
                        new_picker_pos[active_picker_indices[j], :] = picker_pos[active_picker_indices[j], :].copy()
                        new_particle_pos[picked_particle_idices[i], :3] = particle_pos[picked_particle_idices[i], :3].copy()
                        new_particle_pos[picked_particle_idices[j], :3] = particle_pos[picked_particle_idices[j], :3].copy()

        self._set_pos(new_picker_pos, new_particle_pos, ids)

class Pusher(Picker):
    def __init__(self, num_picker=3, pusher_length=0.2, env=None, picker_low=None, picker_high=None, **kwargs):
        super().__init__(num_picker=num_picker,
                         picker_low=picker_low,
                         picker_high=picker_high,
                         **kwargs)
        picker_low, picker_high = list(picker_low), list(picker_high)
        self.action_space = Box(np.array([*picker_low, 0 ,0.]),
                                np.array([*picker_high, 2*np.pi, 1.]), dtype=np.float32)
        self.delta_move = 0.003
        self.env = env
        # orientation of pusher
        self.ori = np.pi/2
        self.pos = np.array([0.0, 0.0, 0.0]) # x,y
        self.pusher_length = pusher_length
        self.color = np.array([0, 0, 255])/255.0

    def min_rotation(self, old_ori, new_ori):
        diff = new_ori - old_ori
        if diff > np.pi:
            diff -= 2*np.pi
        elif diff < -np.pi:
            diff += 2*np.pi

        return diff
    
    def get_rotation_angle(self, v1, v2):
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        determinant = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.atan2(determinant, dot_product)
        return angle
    
    # find the orientation of the pusher
    def find_orientation(self, endA, endB):
        vecA = endB - endA
        # find artan of vecA
        ori = math.atan2(vecA[1], vecA[0])

        return ori

        # vecB = endB - endA

        # # toX defined as 0 degree of orientation when pusher is parallel to x axis
        # toX = np.array([1,0])

        # angleA = self.get_rotation_angle(toX, vecA)
        # angleB = self.get_rotation_angle(toX, vecB)

        # # find minum angle
        # if abs(angleA) < abs(angleB):
        #     return angleA
        # else:
        #     return angleB
        

    def step(self, action):
        """
        action: Array of 5 . For each picker, the action should be [x, y, z, rot, pick/drop]. The picker will then first pick/drop, and keep
        the pick/drop state while moving towards x, y, x, rot
        """
        total_steps = 0
        # action = action.reshape(-1, 4)
        # curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[int(self.num_picker/2), :3]
        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3][self.pusher_particles]
        curr_mid_pos = curr_pos[int(self.num_picker/2)]

        end_pos = self._apply_picker_boundary(action[:3])
        dist = np.linalg.norm(curr_mid_pos - end_pos)
        num_step = np.ceil(dist / self.delta_move)
        if num_step < 0.1:
            return
        delta_t = (end_pos - curr_mid_pos) / num_step
        delta_r = self.min_rotation(self.ori, action[3])/num_step

        # print("dist: ", dist)
        # print("num_step: ", num_step)
        # print("delta_move: ", self.delta_move)

        # print("---------------")
        # print("Move to: ", action)

        #norm_delta = np.linalg.norm(delta)
        for i in range(int(num_step)):  # The maximum number of steps allowed for one pick and place
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3][self.pusher_particles]
            curr_mid_pos = curr_pos[int(self.num_picker/2)]
            dist = np.linalg.norm(end_pos - curr_mid_pos)
            if np.alltrue(dist < delta_t):
                delta_t = end_pos - curr_mid_pos

            # print("Current_mid_pos: ", curr_mid_pos)
            # print("End_pos: ", end_pos)
            # print("dist: ", dist)
            # print("delta_t: ", delta_t)
            # print("num_step: ", num_step)
            # print("delta_move: ", self.delta_move)

            actions = []
            for j in range(self.num_picker):
                if j == int(self.num_picker/2):
                    # translation of center point
                    actions.extend([*delta_t, action[4]])
                    # print("step: ", [*delta_t, action[4]])
                else:
                    # translation and rotation of other points
                    s, c = np.sin(delta_r), np.cos(delta_r)
                    centered_pos = curr_pos[j][:] - curr_mid_pos
                    r_t = [
                        centered_pos[0]*np.cos(delta_r) - centered_pos[2]*np.sin(delta_r) - centered_pos[0], 
                        0, 
                        centered_pos[0]*np.sin(delta_r) + centered_pos[2]*np.cos(delta_r) - centered_pos[2]
                    ]
                    actions.extend([delta_t[0]+r_t[0], delta_t[1]+r_t[1], delta_t[2]+r_t[2], action[4]])
            super().step(actions, ids=self.pusher_particles)
            pyflex.step()
            total_steps += 1
            if self.env is not None and self.env.recording:
                # print("recording in action_tool")
                self.env.video_frames.append(self.env.render(mode='rgb_array'))
            if np.alltrue(dist < self.delta_move):
                break

        self.ori = action[3]
        self.pos = end_pos

        return total_steps
    
    def unpick(self):
        new_picker_pos, particle_pos = self._get_pos()
        new_particle_pos = particle_pos.copy()

        for i in range(self.num_picker):
            if self.picked_particles[i] is not None:
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]  # Revert the mass
                self.picked_particles[i] = None
        
        self._set_pos(new_picker_pos, new_particle_pos)



    def reset(self, pos):
        """
        center: [x,y,z,rot]
        Update the position of middle picker as [x,y,z] other pickers as 
        [x(dist, rot), y, z(dist, rot)]
        """
        # the number of pickers should be odd, and evenly distributed on the pusher.
        # from left to right, pk0, pk1, ... pk_n-1; middle picker is int(num/2)
        # dist of pk_k is 
        # Thus, pos of pk_k is 
        self.ori = pos[3]
        self.pos = pos[:3]
        
        # pusher np.array([1, 0, 0]) parallel with x axis defined as 0 orientation angle(facing up)
        s, c = np.sin(pos[3] - 0), np.cos(pos[3] - 0)
        if self.num_picker > 1:
            toX = np.array([1, 0, 0])*self.pusher_length/(self.num_picker-1)
        else: 
            toX = np.array([0, 0, 0])
        # rotate the orientation
        dist_dir = np.array([
            toX[0]*c - toX[2]*s,
            0,
            toX[0]*s + toX[2]*c
        ])

        init_picker_poses = self._get_linear_picker_pos(pos[:3], dist_dir)

        self.pusher_particles = []
        for picker_pos in init_picker_poses:
            # shape_id = pyflex.add_box(np.array([self.picker_radius, self.picker_radius, self.picker_radius]), picker_pos, [1, 0, 0, 0], 0)
            shape_id = pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
            pyflex.set_shape_color(shape_id, self.color)
            self.pusher_particles.append(shape_id)
        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]

        self.picked_particles = [None] * self.num_picker

    def set_pos(self, pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)

        self.ori = pos[3]
        self.pos = pos[:3]

        s, c = np.sin(pos[3] - 0), np.cos(pos[3] - 0)
        toX = np.array([1, 0, 0])*self.pusher_length/(self.num_picker-1)
        # rotate the orientation
        dist_dir = np.array([
            toX[0]*c - toX[2]*s,
            0,
            toX[0]*s + toX[2]*c
        ])
        init_picker_poses = self._get_linear_picker_pos(pos[:3], dist_dir)
        # actions = []
        for (i, picker_pos) in enumerate(init_picker_poses):
            shape_states[self.pusher_particles[i], :3] = picker_pos
            shape_states[self.pusher_particles[i], 3:6] = picker_pos
            # actions.extend([*picker_pos, 0.0])
        pyflex.set_shape_states(shape_states)

        # super().step(actions)
        pyflex.step()
    
        
    def _get_linear_picker_pos(self, center, dist_dir):
        pos = [center + (i - int(self.num_picker/2))*dist_dir for i in range(self.num_picker)]
        return np.array(pos)


class PickerPickPlace(Picker):
    def __init__(self, num_picker, env=None, picker_low=None, picker_high=None, **kwargs):
        super().__init__(num_picker=num_picker,
                         picker_low=picker_low,
                         picker_high=picker_high,
                         **kwargs)
        picker_low, picker_high = list(picker_low), list(picker_high)
        self.action_space = Box(np.array([*picker_low, 0.] * self.num_picker),
                                np.array([*picker_high, 1.] * self.num_picker), dtype=np.float32)
        self.delta_move = 0.01
        self.env = env

    def step(self, action):
        """
        action: Array of pick_num x 4. For each picker, the action should be [x, y, z, pick/drop]. The picker will then first pick/drop, and keep
        the pick/drop state while moving towards x, y, x.
        """
        total_steps = 0
        action = action.reshape(-1, 4)
        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
        end_pos = np.vstack([self._apply_picker_boundary(picker_pos) for picker_pos in action[:, :3]])
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        for i in range(int(min(num_step, 300))):  # The maximum number of steps allowed for one pick and place
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            pyflex.step()
            total_steps += 1
            if self.env is not None and self.env.recording:
                self.env.video_frames.append(self.env.render(mode='rgb_array'))
            if np.alltrue(dist < self.delta_move):
                break
        return total_steps

    def get_model_action(self, action, picker_pos):
        """Input the action and return the action used for GNN model prediction"""
        action = action.reshape(-1, 4)
        curr_pos = picker_pos
        end_pos = np.vstack([self._apply_picker_boundary(picker_pos) for picker_pos in action[:, :3]])
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return [], curr_pos
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        model_actions = []
        for i in range(int(min(num_step, 300))):  # The maximum number of steps allowed for one pick and place
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            model_actions.append(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            curr_pos += delta
            if np.alltrue(dist < self.delta_move):
                break
        return model_actions, curr_pos


from softgym.utils.gemo_utils import intrinsic_from_fov, get_rotation_matrix


class PickerQPG(PickerPickPlace):
    def __init__(self, image_size, cam_pos, cam_angle, full=True, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.cam_pos = cam_pos
        self.cam_angle = cam_angle
        self.action_space = Box(np.array([-1., -1, -0.3, 0., -0.3]),
                                np.array([1., 1., *([0.3] * 3)]), dtype=np.float32)
        assert self.num_picker == 1
        self.full = full
        self.total_steps = None

    def _get_world_coor_from_image(self, u, v):
        height, width = self.image_size
        K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

        # Apply back-projection: K_inv @ pixels * depth
        cam_x, cam_y, cam_z = self.cam_pos
        cam_x_angle, cam_y_angle, cam_z_angle = self.cam_angle

        # get rotation matrix: from world to camera
        matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
        # matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [np.cos(cam_x_angle), 0, np.sin(cam_x_angle)])
        matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
        rotation_matrix = matrix2 @ matrix1

        # get translation matrix: from world to camera
        translation_matrix = np.zeros((4, 4))
        translation_matrix[0][0] = 1
        translation_matrix[1][1] = 1
        translation_matrix[2][2] = 1
        translation_matrix[3][3] = 1
        translation_matrix[0][3] = - cam_x
        translation_matrix[1][3] = - cam_y
        translation_matrix[2][3] = - cam_z
        matrix = np.linalg.inv(rotation_matrix @ translation_matrix)

        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]
        vec = ((u - u0) / fx, (v - v0) / fy)
        depth = self._get_depth(matrix, vec, self.particle_radius)  # Height to be the particle radius

        # Loop through each pixel in the image
        # Apply equation in fig 3
        x = (u - u0) * depth / fx
        y = (v - v0) * depth / fy
        z = depth
        cam_coords = np.array([x, y, z, 1])
        cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)

        world_coord = matrix @ cam_coords  # 4 x (height x width)
        world_coord = world_coord.reshape(4)
        return world_coord[:3]

    def _get_depth(self, matrix, vec, height):
        """ Get the depth such that the back-projected point has a fixed height"""
        return (height - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])

    def reset(self, *args, **kwargs):
        self.total_steps = 0
        super().reset(*args, **kwargs)

    def step(self, action):
        """ Action is in 5D: (u,v) the start of the pick in image coordinate; (dx, dy, dz): the relative position of the place w.r.t. the pick"""
        u, v = action[:2]
        u = ((u + 1.) * 0.5) * self.image_size[0]
        v = ((v + 1.) * 0.5) * self.image_size[1]
        x, y, z = self._get_world_coor_from_image(u, v)
        y += 0.01
        dx, dy, dz = action[2:]

        st_high = np.array([x, 0.2, z, 0])
        st = np.array([x, y, z, 0])
        en = st + np.array([dx, dy, dz, 1])
        # print('st:', st)
        if self.full:
            self.total_steps += super().step(st_high)
            self.total_steps += super().step(st)
            self.total_steps += super().step(en)
            en[3] = 0  # Drop cloth
            # Unpick all particles
            _, particle_pos = self._get_pos()
            new_particle_pos = particle_pos.copy()
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]  # Revert the mass
                    self.picked_particles[i] = None
            pyflex.set_positions(new_particle_pos)
            for i in range(20):
                pyflex.step()
                if self.env is not None and self.env.recording:
                    self.env.video_frames.append(self.env.render(mode='rgb_array'))
            self.total_steps += 20
        else:
            raise NotImplementedError
        return self.total_steps

    def get_model_action(self, action, curr_pos):
        u, v = action[:2]
        u = ((u + 1.) * 0.5) * self.image_size[0]
        v = ((v + 1.) * 0.5) * self.image_size[1]
        x, y, z = self._get_world_coor_from_image(u, v)
        y += 0.01
        dx, dy, dz = action[2:]

        st_high = np.array([x, 0.2, z, 0])
        st = np.array([x, y, z, 0])
        en = st + np.array([dx, dy, dz, 1])

        model_actions = []
        model_action, curr_pos = super().get_model_action(st_high, curr_pos)
        model_actions.extend(model_action)
        model_action, curr_pos = super().get_model_action(st, curr_pos)
        model_actions.extend(model_action)
        model_action, curr_pos = super().get_model_action(en, curr_pos)
        model_actions.extend(model_action)
        return model_actions, curr_pos
