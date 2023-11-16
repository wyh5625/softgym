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
polygon_name = "pants"
scene_id = 5
pusher_length = 0.125

action_file_id = 1
action_file = [
    'control_cloth_50cm.csv',
    'control_pants_50cm_x.csv',
    'control_pants_20cm_x.csv'
]

class PolygonPushEnv(ClothPushEnv):
    # def __init__(self, **kwargs):
    #     pass

    ### Part 1. Helper function ###

    # set the vertices of the polygon model
    def set_vertices(self):
        obj_vertices_selector.main(polygon_name)

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
        draw_contact_pose.show(polygon_name)

    # generate txt file of corner particle and segments particles of the corner, eg. pants.txt
    def find_corner_and_segments(self):
        draw_contact_pose.get_corner_and_segment(polygon_name)

    # read the corner particle and segments particles from the txt file
    def read_corner_and_segments(self):
        sorted_boundary_segments = draw_contact_pose.read_sorted_boundary_segments_from_file(polygon_name)
        particle_corner_segments = {}
        # convert to particle index
        inMap = draw_contact_pose.get_vertice_particle_mapping(polygon_name)
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
        for corner, segments in sorted_boundary_segments.items():
            corner_pos.append(particle_pos[corner])
            for segment in segments:
                segment_pos.append(particle_pos[segment])
        corner_pos = np.array(corner_pos)
        segment_pos = np.array(segment_pos)

        import matplotlib.pyplot as plt
        plt.scatter(corner_pos[:, 0], corner_pos[:, 1])
        plt.scatter(segment_pos[:, 0], segment_pos[:, 1])
        plt.show()


    ### Part 2. Class functions ###

    def get_default_config(self):
        config = super().get_default_config()
        config['env_idx'] = scene_id
        config['ClothSize'] = [0.48, 0.4]
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

        # for each contact pose, insert y=0.001 to the second element
        for i in range(len(contact_poses)):
            contact_poses[i].insert(1, 0.03 + self.table_size[1])

        return contact_poses

    # put the pusher on the id-th contact pose
    def place_contact_pose(self, id):
        contact_poses = self.get_contact_poses()
        
        # get the pusher pose
        pusher_pose = contact_poses[id]

        self.set_pusher(pusher_pose)

        return pusher_pose

    def p2p_repositioning(self):
        planner.p2p_repositioning(self)

    def rrt_planning(self, start_config, target_config, cps, constraints=None):
        path = planner.rrt_planning(start_config, target_config, action_file[action_file_id], cps, constraints)
        return path

    def path_repositioning(self, path, refinement=False):
        planner.execute_path(self, path, draw_inter=True, refinement=refinement)
    
    def sample_action(env, pusher_poses):
        # dir sampling
        dir_step = 15
        rot_step = 15
        move_len = 0.2
        # relative to the frame of the pusher
        dir_angles = np.deg2rad(np.arange(0, 181, dir_step))
        vectors = move_len * \
            np.stack((np.cos(dir_angles), np.sin(dir_angles)), axis=-1)
        rots = np.deg2rad(np.arange(-90, 91, rot_step))

        start_time = time.time()          
        deform_data = []

        # get the state of the flat configuration for discovering if the particle pos/vel is in nan state
        flat_state_dict = env.get_state()

        np.set_printoptions(precision=3, suppress=True, formatter={'separator': ','})

        # env.start_record()
        # env.shoot_frame()
        # frame_path = os.path.join("./data", 'deformation')
        # save_frame_as_image(env.video_frames[-1], os.path.join(frame_path, f"init.png"))

        for cp_id, push_pose in enumerate(pusher_poses[2:3]):
            push_pose = np.array(push_pose)
            print("pusher pose: ", push_pose)
            for i in range(len(vectors)):
            
                print("push direction: ", vectors[i])

                actions_all_rots = []
                for rot in rots:
                    print("rot: ", rot)

                    # set init state
                    env.set_state(flat_state_dict)

                    # do test sample
                    p = [push_pose[0], push_pose[2], push_pose[3]]
                    u = [vectors[i][0], vectors[i][1], rot]
                    M_p = coordinate_to_matrix(*p)
                    M_u = coordinate_to_matrix(*u)

                    M = np.matmul(M_p, M_u)
                    p_end = matrix_to_coordinate(M)

                    # get the action in world frame
                    end_pos = np.array([p_end[0], 0.01, p_end[1], p_end[2]])
                    action = [*end_pos, 1.0]

                    
                    # push_pose_init = np.array([*center_pos, push_pose[1]])
                    error = env.test_sample(push_pose, action, record=False,
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

        df.to_csv('control_pants_20cm_x.csv', index=False)

        print("--- %s seconds ---" % (time.time() - start_time))