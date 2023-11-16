import pyflex
from controller import *
import RRT_Star_MG as rrt_star

# average distance between every pair of corresponding corners
def D(c_start, c_target):
    c_start_2d = c_start[:, [0,2]]
    c_target_2d = c_target[:, [0,2]]
    dist = 0
    for i in range(c_start.shape[0]):
        dist += np.linalg.norm(c_start_2d[i] - c_target_2d[i])
    return dist / c_start.shape[0]

def maxD(c_start, c_target):
    c_start_2d = c_start[:, [0,2]]
    c_target_2d = c_target[:, [0,2]]
    dist = 0
    for i in range(c_start.shape[0]):
        dist = max(dist, np.linalg.norm(c_start_2d[i] - c_target_2d[i]))
    return dist
    

def p2p_repositioning(env, epsilon = 0.01, no_penetration = True, middle_state = None, deformed_first = False):

    # Target position 
    if middle_state is None:
        # print("use middle state: ", middle_state)
        target_pos = env.t_pos
        target_cnr = env.target_cornersPos
    else:
        target_pos = env.transform_particles(env.init_pos, 
                                translation=np.array([middle_state[0], 0, middle_state[1]]), 
                                angle=middle_state[2], 
                                center=np.mean(env.cornerPos_init, axis=0)[:3], 
                                set_position=False)
        target_cnr = env.set_inter_corner([middle_state[0], middle_state[1]], middle_state[2], draw=False)

    # env.plot_corner_and_segments(env.t_pos[:, [0, 2]])
    env.get_current_corner_pos()
    

    # Current position
    curr_pos = pyflex.get_positions().reshape(-1, 4)

    refine_count = 0
    while maxD(env.cornerPos, target_cnr) > epsilon and refine_count < 10:
        print("D distance: ", D(curr_pos, target_pos))
        refine_count += 1

        # get current contact poses
        curr_cp = env.get_contact_poses()

        # get target contact poses
        target_cp = env.get_contact_poses(target_pos[:, [0, 2]])

        # print("curr_cp: ", curr_cp)
        # print("target_cp: ", target_cp)

        distance_threshold = 0.01

        # find the most distant corner pair
        distances = np.linalg.norm(np.array(env.cornerPos)[:, :3] - np.array(target_cnr)[:, :3], axis=1)
        max_distance_indices = np.unravel_index(np.argmax(distances), distances.shape)
        print("max_distance_indices: ", max_distance_indices)

        cp_id_1 = max_distance_indices[0] * 3 - 1
        cp_id_2 = max_distance_indices[0] * 3

    
        move_dir = [
            np.array(target_cp)[cp_id_1, [0, 2]] - np.array(curr_cp)[cp_id_1, [0, 2]],
            np.array(target_cp)[cp_id_2, [0, 2]] - np.array(curr_cp)[cp_id_2, [0, 2]]
        ]

        print("move_dir: ", move_dir)
        move_ori = [
            np.arctan2(move_dir[0][1], move_dir[0][0]),
            np.arctan2(move_dir[1][1], move_dir[1][0])
        ]
        print("move_ori: ", move_ori)
        print("pusher_ori: ", np.array(curr_cp)[cp_id_1, 3], np.array(curr_cp)[cp_id_2, 3])
        # angle between the move_ori and the orientation of curr_cp
        angle_diff = [
            np.abs(np.array(curr_cp)[cp_id_1, 3] + np.pi/2 - move_ori[0]),
            np.abs(np.array(curr_cp)[cp_id_2, 3] + np.pi/2 - move_ori[1])
        ]
        angle_diff = [
            np.rad2deg(np.minimum(angle_diff[0], 2 * np.pi - angle_diff[0])),
            np.rad2deg(np.minimum(angle_diff[1], 2 * np.pi - angle_diff[1]))
        ]  

        print("angle_diff: ", angle_diff)
        
        # argmin of angle_diff
        cp_id = cp_id_1 if angle_diff[0] < angle_diff[1] else cp_id_2


        # find the minimum orientation difference, since the orientation is in c, we need to consider the case when the difference is larger than pi
        # orientations = np.array(curr_cp)[cp_ids, 3] - np.array(target_cp)[cp_ids, 3]
        # orientations = np.abs(orientations)
        # orientations = np.minimum(orientations, 2 * np.pi - orientations)

        # # filter out the contact poses that are too close, those are not considered
        # orientations[distances < distance_threshold] = 1000
        # min_orientation_indices = np.unravel_index(np.argmin(orientations), orientations.shape)[0]

        # print("max_distance_indices: ", max_distance_indices)
        push_from = curr_cp[cp_id]
        push_to = target_cp[cp_id]
        # execute the push action
        execute_a_push_action(env, push_from, push_to)

        # record_frame(env)

        # env.push(push_action)

        # execute push action
        for i in range(5):
            pyflex.step()


        # update state
        env.get_current_corner_pos()
        # c_start_cnr = env.cornerPos

        curr_pos = pyflex.get_positions().reshape(-1, 4)

    if refine_count >= 10:
        print("refinement failed")
        return False
    
    return True

def execute_path(env, path, refinement=False, draw_inter=False):
    # prev_layer = 0
        
    # env.set_state(flat_state_dict)
    # env.init_pos()

    # wait until the cloth is stable
    # for i in range(100):
    #     pyflex.step()

    # give a sample path

    

    previous_node = path[0][0]
    u_path = path[1:]
    num_u = len(u_path)
    for i in range(num_u):
        (next_node, cp_id) = u_path[i]
        # next_node = next_state.node
            
        # if action is None:
        #     previous_node = next_node
        #     continue

        # pos = env.get_pusher_pos()
        # push_x, push_y, push_ori = pos[0], pos[2], pos[3]

        # if not touched:

        # get current contact poses
        curr_cp = env.get_contact_poses()

        # get target contact poses
        # target pos
        target_pos = env.transform_particles(env.init_pos, 
                                translation=np.array([next_node[0], 0, next_node[1]]), 
                                angle=next_node[2], 
                                center=np.mean(env.cornerPos_init, axis=0)[:3], 
                                set_position=False)
        target_cp = env.get_contact_poses(target_pos[:, [0, 2]])
        

        # draw intermediate state
        color = [255, 0, 0]
        if i == num_u - 1:
            color = [0, 255, 0]
        inter_corners = env.set_inter_corner([next_node[0], next_node[1]], next_node[2], draw=draw_inter , color=color)
        # record_frame(env)


        push_from = curr_cp[cp_id]
        push_to = target_cp[cp_id]
        execute_a_push_action(env, push_from, push_to)

        # record deformation error
        
        # record time
        # times.append(env.time_step)

        # if the next action's cp is different from the current one, do repositioning
        # check if cp change
        cp_change = False
        if i < num_u - 1:
            if u_path[i][1] != u_path[i+1][1]:
                print("cp change")
                cp_change = True
        elif i == num_u - 1:
            cp_change = True

        # refinement
        if refinement and cp_change:
            p2p_repositioning(env, epsilon=0.05, no_penetration=True, middle_state=next_node)
            

        # keep track of layer of last node, which is used to rotate the action
        # prev_layer = next_node[2]

        # previous_node = next_node
    


    # save_name = osp.join('./data/', "ML_{}.gif".format("rrt_star"))
    


    pos = env.get_pusher_pos()
    push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    # env.push(np.array([push_x, -0.01, push_y, push_ori, 0]))

    pyflex.step()


def rrt_planning(start_config, target_config, control_file, cps, constraints=None, save_plot=None, repeat=1):
    path = rrt_star.plan(start_config, target_config, control_file, cps, constraints, save_plot=save_plot, repeat=repeat)
    return path
