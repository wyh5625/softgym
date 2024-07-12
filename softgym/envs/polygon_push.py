from softgym.envs.cloth_push import ClothPushEnv
from softgym.envs import obj_vertices_selector
from softgym.envs import draw_contact_pose
import pyflex
import numpy as np
import time
import pandas as pd
from RRT_Star_MG import coordinate_to_matrix, matrix_to_coordinate
import planner


# A class defining basic operation on the cloth

pusher_length = 0.125

# action_file_id = 0
action_file = [
    'control_rectangle_50cm_x.csv',
    'control_pants_50cm_x.csv',
    'control_pants_20cm_x.csv'
]

Env_Setting = {
    'Rectangle': {
        'polygon_name': 'rectangle',
        'scene_id': 0,
        'sampling_ingore_cp': [],
        'action_file': 'control_small_rectangle_50cm_x.csv'
    },
    'Pants': {
        'polygon_name': 'pants',
        'scene_id': 5,
        'sampling_ingore_cp': [12, 13, 14, 15, 16, 17],
        'action_file': 'control_pants_50cm_x.csv'
    },
    'Shirts': {
        'polygon_name': 't-shirt',
        'scene_id': 6,
        'sampling_ingore_cp': [3, 4, 5, 21, 22, 23],
        'action_file': 'control_t-shirt_50cm_x.csv'
    },
    'LongSleeve': {
        'polygon_name': 'long_sleeve',
        'scene_id': 7,
        'sampling_ingore_cp': [0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 33, 34, 35],
        'action_file': 'control_long_sleeve_50cm_x.csv'
    }
}

class PolygonPushEnv(ClothPushEnv):
    def __init__(self, model_name="Pants", **kwargs):
        self.polygon_name = Env_Setting[model_name]['polygon_name']
        self.scene_id = Env_Setting[model_name]['scene_id']
        self.action_file = Env_Setting[model_name]['action_file']
        self.sampling_ingore_cp = Env_Setting[model_name]['sampling_ingore_cp']
        super().__init__(**kwargs)
        

    ### Part 1. Helper function ###

    # set the vertices of the polygon model
    def set_vertices(self):
        obj_vertices_selector.main(self.polygon_name)

    # get the idx of corner particles
    def read_polygon_corners(self):
        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        corners = []
        for i in self.cornerIdx:
            corners.append(particle_pos[i])
        corners = np.array(corners).reshape(-1, 3)

        return corners

    # show contact poses
    def show_contact_poses(self):
        draw_contact_pose.show(self.polygon_name)

    # generate txt file of corner particle and segments particles of the corner, eg. pants.txt
    def find_corner_and_segments(self):
        draw_contact_pose.get_corner_and_segment(self.polygon_name)

    # read the corner particle and segments particles from the txt file
    def read_corner_and_segments(self):
        if self.polygon_name == 'rectangle':
            config = self.get_default_config()
            cloth_dimx, cloth_dimy = config['ClothSize']
            print("cloth_dimx: ", cloth_dimx, "cloth_dimy: ", cloth_dimy)
            sorted_boundary_segments = draw_contact_pose.find_corner_and_segments_rec(cloth_dimx, cloth_dimy)

            return sorted_boundary_segments
        else:
            sorted_boundary_segments = draw_contact_pose.read_sorted_boundary_segments_from_file(self.polygon_name)
            particle_corner_segments = {}
            # convert to particle index
            inMap = draw_contact_pose.get_vertice_particle_mapping(self.polygon_name)
            for corner, segments in sorted_boundary_segments.items():
                corner = inMap[corner]
                segments = [inMap[i] for i in segments]
                particle_corner_segments[corner] = segments
                    
            return particle_corner_segments

    # plot the corner and segments particles
    def plot_corner_and_segments(self, particle_pos=None):
        if particle_pos is None:
            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, [0, 2]]
        
        # read the corner and segments particles
        sorted_boundary_segments = self.read_corner_and_segments()

        # plot the corner and segments particles
        corner_pos = []
        segment_pos = []
        # i = 0
        # seg_idx = 6
        for corner, segments in sorted_boundary_segments.items():
            corner_pos.append(particle_pos[corner])
            for segment in segments:
                # if i == seg_idx:
                segment_pos.append(particle_pos[segment])
            # i += 1
        corner_pos = np.array(corner_pos)
        segment_pos = np.array(segment_pos)

        import matplotlib.pyplot as plt
        plt.scatter(corner_pos[:, 0], corner_pos[:, 1])
        plt.scatter(segment_pos[:, 0], segment_pos[:, 1])
        plt.show()


    ### Part 2. Class functions ###

    def get_default_config(self):
        config = super().get_default_config()
        config['env_idx'] = self.scene_id
        # config['ClothSize'] = [0.48, 0.4]
        return config

    def get_corner_particles(self):
        sorted_boundary_segments = self.read_corner_and_segments()
        self.cornerIdx = []
        for corner, segments in sorted_boundary_segments.items():
            self.cornerIdx.append(corner)

    def snap_to_center(self, center=[0, 0, 0]):
        corners = self.read_polygon_corners()
        # print(corners)
        # print(pyflex.get_positions())
        mean = np.mean(corners, axis=0)

        mean -= np.array(center)
        # mean[0] = 10
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        # print(particle_pos)
        print("size: ", len(particle_pos))
        particle_pos[:, :3] -= mean
        pyflex.set_positions(particle_pos.flatten())

    def get_contact_poses(self, particle_pos=None):
        if particle_pos is None:
            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, [0, 2]]
        
        # read the corner and segments particles
        sorted_boundary_segments = self.read_corner_and_segments()

        # get contact poses from the corner and segments particles
        contact_poses = draw_contact_pose.get_contact_poses(particle_pos, sorted_boundary_segments, pusher_length)

        # for each contact pose, insert y=0.03 to the second element
        for i in range(len(contact_poses)):
            contact_poses[i].insert(1, 0.025 + self.table_size[1])

        return contact_poses

    # put the pusher on the id-th contact pose
    def place_contact_pose(self, id):
        contact_poses = self.get_contact_poses()
        
        # get the pusher pose
        pusher_pose = contact_poses[id]

        self.set_pusher(pusher_pose)

        return pusher_pose
    
    def place_contact_poses(self):
        contact_poses = self.get_contact_poses()
        for i in range(len(contact_poses)):
            self.set_pusher(contact_poses[i])
            # wait for 1 second
            for j in range(100):
                pyflex.step()

    def p2p_repositioning(self, epsilon=0.01, cp_optimal=None):
        return planner.p2p_repositioning(self, epsilon=epsilon, cp_optimal=cp_optimal)

    def rrt_planning(self, start_config, target_config, cps, constraints=None):
        path = planner.rrt_planning(start_config, target_config, self.action_file, cps, constraints)
        return path

    def path_repositioning(self, path, refinement=False, cp_star=True):
        planner.execute_path(self, path, draw_inter=True, refinement=refinement, cp_star=cp_star)

    def reset_planner(self):
        planner.reset()

    def get_actions(self):
        return planner.actions

    def get_distances(self):
        return planner.MPD, planner.CHAMFER
    
    def sample_action(self, pusher_poses):
        # dir sampling
        dir_step = 15
        rot_step = 15
        move_len = 0.5
        # relative to the frame of the pusher
        dir_angles = np.deg2rad(np.arange(0, 181, dir_step))
        vectors = move_len * \
            np.stack((np.cos(dir_angles), np.sin(dir_angles)), axis=-1)
        rots = np.deg2rad(np.arange(-90, 91, rot_step))

        start_time = time.time()          
        deform_data = []

        # get the state of the flat configuration for discovering if the particle pos/vel is in nan state
        flat_state_dict = self.get_state()

        np.set_printoptions(precision=3, suppress=True, formatter={'separator': ','})

        # env.start_record()
        # env.shoot_frame()
        # frame_path = os.path.join("./data", 'deformation')
        # save_frame_as_image(env.video_frames[-1], os.path.join(frame_path, f"init.png"))

        for cp_id, push_pose in enumerate(pusher_poses):
            if cp_id in self.sampling_ingore_cp:
                continue

            push_pose = np.array(push_pose)
            print("pusher pose: ", push_pose)
            for i in range(len(vectors)):
            
                print("push direction: ", vectors[i])

                actions_all_rots = []
                for rot in rots:
                    print("rot: ", rot)

                    # set init state
                    self.set_state(flat_state_dict)

                    # do test sample
                    p = [push_pose[0], push_pose[2], push_pose[3]]
                    u = [vectors[i][0], vectors[i][1], rot]
                    M_p = coordinate_to_matrix(*p)
                    M_u = coordinate_to_matrix(*u)

                    M = np.matmul(M_p, M_u)
                    p_end = matrix_to_coordinate(M)

                    # get the action in world frame
                    end_pos = np.array([p_end[0], push_pose[1], p_end[1], p_end[2]])
                    action = [*end_pos, 1.0]

                    
                    # push_pose_init = np.array([*center_pos, push_pose[1]])
                    error = self.test_sample(push_pose, action, record=False,
                                                img_size=720, save_video_dir='./data/')
                    # env.shoot_frame()
                    # frame_path = os.path.join("./data", 'deformation')
                    # save_frame_as_image(env.video_frames[-1], os.path.join(frame_path, f"deform_{j}.png"))
                    # j += 1
                        
                    # env.get_current_corner_pos()

                    # result_points_str = "[" + ", ".join(["[{}, {}]".format(p[0], p[1]) for p in env.cornerPos[:,[0,2]]]) + "]"
                    # print(result_points_str)
                    data_item = (cp_id, u[0], u[1], u[2], error)
                    # deform_data.append(data_item)
                    actions_all_rots.append(data_item)
                    # print(result_points_str)
                         
                # sort the actions_all_rots by error
                actions_all_rots.sort(key=lambda x: x[4])
                # append first two actions to deform_data
                deform_data.append(actions_all_rots[0])
                deform_data.append(actions_all_rots[1])
                deform_data.append(actions_all_rots[2])
                

        columns = ['cp_id', 'trans_x', 'trans_y', 'rot', 'deformation']
        df = pd.DataFrame(deform_data, columns=columns)

        # df.to_csv('control_rectangle_50cm_x.csv', index=False)

        # save with name of self.polygon_name
        df.to_csv('control_small_{}_{}cm_x.csv'.format(self.polygon_name, int(move_len*100)), index=False)

        print("--- %s seconds ---" % (time.time() - start_time))


    # mean particle distance to the target position

    def EDs(self):
        return planner.EDs
    
    def RDs(self):
        return planner.RDs
    
    def LDs(self):
        return planner.LDs

    def CHAMFER(self):
        return planner.CHAMFER
    
    def IOUs(self):
        return planner.IOUs
