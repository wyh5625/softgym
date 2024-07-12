import pyflex
from controller import *
import RRT_Star_MG as rrt_star
import cv2

# average distance between every pair of corresponding corners
MAX_ACTIONS = 10
actions = 0
MAX_MPD = 0.1

# align with actions

CHAMFER = []
RDs = [] 
EDs = []
LDs = []
IOUs = []

def D(c_start, c_target):
    c_start_2d = c_start[:, [0, 2]]
    c_target_2d = c_target[:, [0, 2]]
    dist = 0
    for i in range(c_start.shape[0]):
        dist += np.linalg.norm(c_start_2d[i] - c_target_2d[i])
    return dist / c_start.shape[0]


def CD(pos_a, pos_b):
    total_distance = 0

    # Calculate the minimum distance from each point in pos_a to pos_b
    for point_a in pos_a:
        distances = np.linalg.norm(pos_b - point_a, axis=1)
        min_distance = np.min(distances)
        total_distance += min_distance

    # Calculate the minimum distance from each point in pos_b to pos_a
    for point_b in pos_b:
        distances = np.linalg.norm(pos_a - point_b, axis=1)
        min_distance = np.min(distances)
        total_distance += min_distance

    # Normalize the sum by dividing it by the total number of points
    num_points = len(pos_a) + len(pos_b)
    chamfer_distance = total_distance / num_points

    return chamfer_distance

def find_IOU(pos_a, pos_b):
    # find the polygon of A
    hull_A = cv2.convexHull(pos_a)
    # find interset(I) of particles of B that are inside A
    I_set = []
    # Perform the point in polygon test
    for point in pos_b: 
        # print(hull_A)
        # print(point)
        result = cv2.pointPolygonTest(hull_A, tuple(point), False)
        if result == 1:
            I_set.append(point)

    # get the area of I
    hull_I = cv2.convexHull(np.array(I_set))
    I_area = cv2.contourArea(hull_I)
    # get the area of A and B
    hull_B = cv2.convexHull(pos_b)
    A_area = cv2.contourArea(hull_A)
    B_area = cv2.contourArea(hull_B)
    # get the area of U = A + B - I
    U_area = A_area + B_area - I_area
    # get the IOU
    IOU = I_area / U_area
    print("IOU: ", IOU)
    return IOU
    

def maxD(c_start, c_target):
    c_start_2d = c_start[:, [0, 2]]
    c_target_2d = c_target[:, [0, 2]]
    dist = 0
    for i in range(c_start.shape[0]):
        dist = max(dist, np.linalg.norm(c_start_2d[i] - c_target_2d[i]))
    return dist

def allD(c_start, c_target):
    c_start_2d = c_start[:, [0, 2]]
    c_target_2d = c_target[:, [0, 2]]
    dist = []
    for i in range(c_start.shape[0]):
        dist.append(np.linalg.norm(c_start_2d[i] - c_target_2d[i]))
    return dist

def match_percentage(c_start, c_target, allowance):
    c_start_2d = c_start[:, [0, 2]]
    c_target_2d = c_target[:, [0, 2]]
    size = c_start.shape[0]
    count = 0
    for i in range(size):
        if np.linalg.norm(c_start_2d[i] - c_target_2d[i]) < allowance:
            print(np.linalg.norm(c_start_2d[i] - c_target_2d[i]))
            count += 1

    print("epsilon: ", allowance)
    print("match count: ", count)
    print("size: ", size)
    
    return count / size

def get_greedy_cps(env, target_pos, target_cnr):
    # get current contact poses
    curr_cp = env.get_contact_poses()

    # get target contact poses
    target_cp = env.get_contact_poses(target_pos[:, [0, 2]])

    env.get_current_corner_pos()
    # find the most distant corner pair
    distances = allD(env.cornerPos, target_cnr)
    # np.linalg.norm(np.array(env.cornerPos)[
    #                            :, :3] - np.array(target_cnr)[:, :3], axis=1)
    max_index = distances.index(max(distances))
    
    print("distances: ", distances)
    print("max_distance_indices: ", max_index)

    cp_id_1 = max_index * 3 - 1
    cp_id_2 = max_index * 3

    move_dir = [
        np.array(target_cp)[cp_id_1, [0, 2]] -
        np.array(curr_cp)[cp_id_1, [0, 2]],
        np.array(target_cp)[cp_id_2, [0, 2]] -
        np.array(curr_cp)[cp_id_2, [0, 2]]
    ]

    print("move_dir: ", move_dir)
    move_ori = [
        np.arctan2(move_dir[0][1], move_dir[0][0]),
        np.arctan2(move_dir[1][1], move_dir[1][0])
    ]
    print("move_ori: ", move_ori)
    print("pusher_ori: ", np.array(curr_cp)[
        cp_id_1, 3], np.array(curr_cp)[cp_id_2, 3])
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

    push_from = curr_cp[cp_id]
    push_to = target_cp[cp_id]

    return push_from, push_to


def p2p_repositioning(env, epsilon=0.01, no_penetration=True, middle_state=None, deformed_first=False, cp_optimal=None):
    global actions
    # Target position
    if middle_state is None:
        # print("use middle state: ", middle_state)
        target_pos = env.t_pos
        target_cnr = env.target_cornersPos
    else:
        target_pos = env.transform_particles(env.start_pos,
                                             translation=np.array(
                                                 [middle_state[0], 0, middle_state[1]]),
                                             angle=middle_state[2],
                                             center=np.mean(
                                                 env.cornerPos_start, axis=0)[:3],
                                             set_position=False)
        target_cnr = env.set_inter_corner(
            [middle_state[0], middle_state[1]], middle_state[2], draw=False)

    # env.plot_corner_and_segments(env.t_pos[:, [0, 2]])
    env.get_current_corner_pos()

    # Current position
    curr_pos = pyflex.get_positions().reshape(-1, 4)

    if cp_optimal:
        node_s = cp_optimal['node_s']
        node_t = cp_optimal['node_t']
        cps = cp_optimal['cps']
        control_file = cp_optimal['control_file']
        constraints = cp_optimal['constraints']
        _, u = rrt_star.steer(node_s, node_t, cps, control_file, constraints)

        # get current contact poses
        curr_cp = env.get_contact_poses()

        # get target contact poses
        target_cp = env.get_contact_poses(target_pos[:, [0, 2]])

        # approach the target using contact pose u[0]
        cp_id = u[0]

        push_from = curr_cp[cp_id]
        push_to = target_cp[cp_id]
        print("push_from: ", push_from)
        # execute the push action
        execute_a_push_action(env, push_from, push_to)
        actions += 1


        # execute push action
        for i in range(5):
            pyflex.step()

        curr_pos = pyflex.get_positions().reshape(-1, 4)
        mpd = D(curr_pos, target_pos)
        cd = CD(curr_pos[:, [0, 2]], target_pos[:, [0, 2]])
        iou = find_IOU(curr_pos[:, [0, 2]], target_pos[:, [0, 2]])
        ld = allD(env.cornerPos, env.target_cornersPos)
        LDs.append(ld)
        RDs.append(mpd)
        CHAMFER.append(cd)
        IOUs.append(iou)

        # update state
        env.get_current_corner_pos()
        # c_start_cnr = env.cornerPos

        curr_pos = pyflex.get_positions().reshape(-1, 4)

    # refine_count = 0
    # increase_count = 0
    # init_D = D(curr_pos, target_pos)
    # if (init_D > MAX_MPD):
    #     print("initial D distance: ", init_D)
    #     return False
    # while np.round(maxD(env.cornerPos, target_cnr), 3) > epsilon:
    while actions < MAX_ACTIONS:
        print("maxD distance: ", maxD(env.cornerPos, target_cnr))
        # mpd = D(curr_pos, target_pos)
        # if (actions >= 1 and mpd > MAX_MPD):
        #     print("exceed max mpd: ", mpd)
        #     return False

        print("D distance: ", D(curr_pos, target_pos))
        # refine_count += 1
        # new_D = D(curr_pos, target_pos)
        # if new_D > prev_D:
        #     increase_count += 1
        # prev_D = new_D
        # find the minimum orientation difference, since the orientation is in c, we need to consider the case when the difference is larger than pi
        # orientations = np.array(curr_cp)[cp_ids, 3] - np.array(target_cp)[cp_ids, 3]
        # orientations = np.abs(orientations)
        # orientations = np.minimum(orientations, 2 * np.pi - orientations)

        # # filter out the contact poses that are too close, those are not considered
        # orientations[distances < distance_threshold] = 1000
        # min_orientation_indices = np.unravel_index(np.argmin(orientations), orientations.shape)[0]

        # print("max_distance_indices: ", max_distance_indices)

        push_from, push_to = get_greedy_cps(env, target_pos, target_cnr)
        print("push_from: ", push_from)
        execute_a_push_action(env, push_from, push_to)

        actions += 1

        curr_pos = pyflex.get_positions().reshape(-1, 4)
        mpd = D(curr_pos, target_pos)
        cd = CD(curr_pos[:, [0, 2]], target_pos[:, [0, 2]])
        iou = find_IOU(curr_pos[:, [0, 2]], target_pos[:, [0, 2]])

        env.get_current_corner_pos()
        ld = allD(env.cornerPos, env.target_cornersPos)
        LDs.append(ld)
        RDs.append(mpd)
        CHAMFER.append(cd)
        IOUs.append(iou)

        # execute the push action
        if actions >= MAX_ACTIONS:
            print("exceed max actions")
            mp = match_percentage(env.cornerPos, target_cnr, epsilon)

            return mp

        
        

        # record_frame(env)

        # env.push(push_action)

        # execute push action
        for i in range(5):
            pyflex.step()

        # update state
        env.get_current_corner_pos()
        # c_start_cnr = env.cornerPos

        curr_pos = pyflex.get_positions().reshape(-1, 4)

    print("final D distance: ", D(curr_pos, target_pos))

    return 1


def execute_path(env, path, refinement=False, draw_inter=False, cp_star=True):
    global actions
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
        (next_node, cp_id, dm) = u_path[i]
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
        inter_pos = env.transform_particles(env.start_pos,
                                            translation=np.array(
                                                [next_node[0], 0, next_node[1]]),
                                            angle=next_node[2],
                                            center=np.mean(
                                                env.cornerPos_start, axis=0)[:3],
                                            set_position=False)
        target_cp = env.get_contact_poses(inter_pos[:, [0, 2]])

        # draw intermediate state
        color = [255, 255, 0]
        if i == num_u - 1:
            color = [0, 255, 0]
        inter_corners = env.set_inter_corner(
            [next_node[0], next_node[1]], next_node[2], draw=draw_inter, color=color)
        # record_frame(env)

        if cp_star:
            print("--------------using cp star--------------")
            push_from = curr_cp[cp_id]
            push_to = target_cp[cp_id]
        else:
            push_from, push_to = get_greedy_cps(env, inter_pos, inter_corners)
        execute_a_push_action(env, push_from, push_to)
        actions += 1

        # find distance to current subgoal
        curr_pos = pyflex.get_positions().reshape(-1, 4)
        mpd = D(curr_pos, inter_pos)
        cd = CD(curr_pos[:, [0, 2]], inter_pos[:, [0, 2]])
        iou = find_IOU(curr_pos[:, [0, 2]], inter_pos[:, [0, 2]])
        ld = allD(env.cornerPos, env.target_cornersPos)
        LDs.append(ld)
        EDs.append(dm)
        RDs.append(mpd)
        CHAMFER.append(cd)
        IOUs.append(iou)

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
            p2p_repositioning(env, epsilon=0.1,
                              no_penetration=True, middle_state=next_node)

        # keep track of layer of last node, which is used to rotate the action
        # prev_layer = next_node[2]

        # previous_node = next_node

    # save_name = osp.join('./data/', "ML_{}.gif".format("rrt_star"))

    pos = env.get_pusher_pos()
    push_x, push_y, push_ori = pos[0], pos[2], pos[3]
    # env.push(np.array([push_x, -0.01, push_y, push_ori, 0]))

    pyflex.step()


def rrt_planning(start_config, target_config, control_file, cps, constraints=None, save_plot=None, repeat=5):
    path = rrt_star.plan(start_config, target_config, control_file,
                         cps, constraints, save_plot=save_plot, repeat=repeat)
    return path

def reset():
    global actions, RDs, LDs, EDs, CHAMFER
    actions = 0
    RDs.clear()
    LDs.clear()
    EDs.clear()
    CHAMFER.clear()
    IOUs.clear()