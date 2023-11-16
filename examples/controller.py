import numpy as np
import datetime


hang_height = 0.05


def generate_push_actions(start_pos, target_pos, circular_rate=0.0, center_side=1, path_side=1, waypoint_size=1):
    # circular_rate is [0, 1], radius of path R = R_min / circular_rate(or cos(theta)), larger circular_rate, smaller R
    # center_side = -1/1, which means the center of circle is on the LHS/RHS of straight path
    # path_side = -1/1, which means the left/right part of circular path is chosen.
    waypoints = []
    actions = []
    # straight line
    if circular_rate == 0:
        diff = target_pos - start_pos
        if diff[3] > np.pi:
            diff[3] -= 2*np.pi
        elif diff[3] < -np.pi:
            diff[3] += 2*np.pi
        step = diff/waypoint_size

        for i in range(waypoint_size):
            waypoint = start_pos + (i+1)*step
            waypoints.append(waypoint)
            action = np.array([*waypoint, 1.0])
            actions.append(action)
    else:
        r_min = np.linalg.norm(target_pos - start_pos)/2
        r_path = r_min / circular_rate
        s = np.sqrt(1 - circular_rate**2)
        mid2center = s*r_path

        # find the rotation center
        mid_x = (start_pos + target_pos)/2
        move_l = target_pos - start_pos
        move_l_dir = move_l / np.linalg.norm(move_l)
        mid2center_dir = np.array(
            [move_l_dir[2], move_l_dir[1], -move_l_dir[0]])
        center = mid_x + mid2center * center_side * mid2center_dir

        # find the circular path
        angle = 2*(np.pi/2 - np.arccos(circular_rate))
        if center_side*path_side > 0:
            angle = 2*np.pi - angle
        if path_side == -1:
            angle = -angle
        step = angle/waypoint_size
        start_pos -= center
        print("Rotate angle:" + str(angle))
        # print(angle)
        for i in range(waypoint_size):
            new_pos = start_pos.copy()
            da = 0 + (i+1)*step
            new_pos[0] = (np.cos(da) * start_pos[0] -
                          np.sin(da) * start_pos[2])
            new_pos[2] = (np.sin(da) * start_pos[0] +
                          np.cos(da) * start_pos[2])
            waypoint = new_pos+center
            waypoints.append(waypoint)
            action = np.array([*waypoint, 1.0])
            actions.append(action)

    return actions


def execute_a_push_action(env, push_start, push_end, contact_pose=None, primary=False, with_standby=False):
    # push_start += np.array([0, env.surface_height, 0, 0])
    # push_end += np.array([0, env.surface_height, 0, 0])

    # global step, time_eplapsed

    # define a standby pusher pose
    standby_pusher_pos = np.array([0, 0.5, 0, 0])

    # assume pusher is in standby pose
    # env.set_pusher(standby_pusher_pos)

    # append 0 to the end of contact pose

    # set speed
    env.action_tool.delta_move = 0.004

    # print("move to contact pose: ", push_start)
    # move to contact pose
    # push_start[1] -= 0.01
    if with_standby:
        push_start = np.append(push_start, 0)
        time_prev = datetime.datetime.now()
        env.push(push_start)
        # time_eplapsed += datetime.datetime.now() - time_prev
    else:
        hang_over = push_start + np.array([0, hang_height, 0, 0])
        hang_over = np.append(hang_over, 0)

        # time_prev = datetime.datetime.now()
        env.push(hang_over)
        push_start = np.append(push_start, 0)
        env.push(push_start)
        # time_eplapsed += datetime.datetime.now() - time_prev

    # g_dt = 0.01s, the time interval of each step in the simulation, so 0.002/0.01 = 0.2 m/s
    env.action_tool.delta_move = 0.002


    push_end_action = np.append(push_end, 1)
    env.push(push_end_action)
    # execute push action
    # divide push action into 5 steps
    # actions = generate_push_actions(
    #     push_start[:4], push_end[:4], waypoint_size=10)

    # print("Start pushing")
    first_push = False
    # delta = (push_action[:4] - push_start) / 5
    # for ac in actions:
    #     # print("ac: ", ac)
    #     # if first_push:
    #     #     record_frame(env)
    #     #     first_push = False

    #     # if record_deform:
    #     #     record_deformation(env)
    #     # pa_small = push_start + delta * i
    #     # pa_small = np.append(pa_small, 1)
    #     time_prev = datetime.datetime.now()
    #     env.push(ac)
    #     # time_eplapsed += datetime.datetime.now() - time_prev

    #     # if primary:
    #     #     inter_pose = env.align_inter_with_pusher(ac[[0,2,3]], contact_pose)
    #     #     env.set_inter_corner([inter_pose[0], inter_pose[1]], inter_pose[2], draw=False)

    #     # if first_push:
    #     #     record_frame(env)
    #     #     first_push = False

    #     # record deformation w.r.t contact pose after each push action
    #     # record_deformation(env, ac[[0, 2, 3]], contact_pose)

    #         # calculate deformation for every mass point

    #         # calculate

    # critical_points.append(step)

    # env.push(push_action)

    env.action_tool.delta_move = 0.004
    # move back to standby pose
    if with_standby:
        standby_pusher_pos = np.append(standby_pusher_pos, 0)

        time_prev = datetime.datetime.now()
        env.push(standby_pusher_pos)
        # time_eplapsed += datetime.datetime.now() - time_prev
    else:
        hang_over = push_end + np.array([0, hang_height, 0, 0])
        # hang_over = np.append(hang_over, 0)
        hang_over = np.append(hang_over, 0)

        time_prev = datetime.datetime.now()
        env.push(hang_over)
        # time_eplapsed += datetime.datetime.now() - time_prev
