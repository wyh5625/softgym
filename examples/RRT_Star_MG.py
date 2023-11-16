import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time
import copy


class State:
    def __init__(self, pos):
        self.pos = pos

        self.cp_id = -1  # contact pose id
        self.cost = float('inf')
        self.parent = None
        self.control = None
        self.children = []


control_type = "TshirtPushPPP"

w1 = 1  # distance
w2 = 10  # deformation

# delta is usuaully smaller then B
delta = [0.1, 0.1, 0.2]  # x, y, theta

# work area
limit = [[-1.5, 1.5], [-1.5, 1.5], [-np.pi, np.pi]]  # x, y, theta

# define s_zero (including its corners' positions of s_zero state),
# which is the state of the cloth where the controls are sampled from
# n_zero = State([0, 0, 0])

# control samples, a tuple of (cp_id, control)
# U = [(0, [0.1, 0, 0])]

# max
# B = [0.1, 0.1, 0.1]


def close(s1, s2):
    if abs(s1.pos[0] - s2.pos[0]) < delta[0] and abs(s1.pos[1] - s2.pos[1]) < delta[1] and abs(s1.pos[2] - s2.pos[2]) < delta[2]:
        return True
    else:
        return False


def coordinate_to_matrix(x, y, theta):
    # Homogeneous transformation matrix for 2D
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), x],
                                    [np.sin(theta), np.cos(theta), y],
                                    [0, 0, 1]])
    return transformation_matrix


def matrix_to_coordinate(transformation_matrix):
    # Extract translation (x, y)
    x = transformation_matrix[0, 2]
    y = transformation_matrix[1, 2]
    # Extract rotation angle (theta)
    theta = np.arctan2(
        transformation_matrix[1, 0], transformation_matrix[0, 0])
    return (x, y, theta)


class MG_RRTStar:
    def __init__(self, start, goal, max_iterations=400, goal_tolerance=0.02):
        self.start = State(start)
        self.goal = State(goal)
        self.max_iterations = max_iterations
        self.goal_tolerance = goal_tolerance

        self.start.cost = 0
        self.nodes = [self.start]

        self.goal_rate = 0.02

        self.constrained = False

        # cp_dict and contact poses
        self.cp_dict = {}
        self.cps = None

        # control space
        self.U = []

        # step_box
        self.B = [0.2, 0.2, 0.8]

    # set the center of table and size, relative to robot
    def set_constraints(self, table_center=(0, -0.9), table_size=(1, 1), reachable_length=1.1, object_size=(0.6, 0.36), object_center_offset=(0, 0)):
        self.constrained = True
        self.table_center = table_center
        self.table_size = table_size
        self.reachable_length = reachable_length
        self.object_size = object_size
        self.object_center_offset = object_center_offset

        # set limit area for sampling
        limit[0][0] = table_center[0] - table_size[0]/2
        limit[0][1] = table_center[0] + table_size[0]/2
        limit[1][0] = table_center[1] - table_size[1]/2
        limit[1][1] = table_center[1] + table_size[1]/2

    # find the u and s_transform, given cp_id s_1 and s_2
    def find_u_and_delta_s(self, s_1, s_2, cp_id):
        # find the transformation from s_1 to s_2
        T_1 = coordinate_to_matrix(s_1.pos[0], s_1.pos[1], s_1.pos[2])
        T_2 = coordinate_to_matrix(s_2.pos[0], s_2.pos[1], s_2.pos[2])

        P = coordinate_to_matrix(
            self.cps[cp_id][0], self.cps[cp_id][2], self.cps[cp_id][3])
        P_1 = np.dot(T_1, P)
        P_2 = np.dot(T_2, P)

        # find the transformation from P_1 to P_2
        P_1_inv = np.linalg.inv(P_1)
        P_1_to_2 = np.dot(P_1_inv, P_2)

        p_transform = matrix_to_coordinate(P_1_to_2)

        # find the transformation from s_1 to s_2
        T_1_inv = np.linalg.inv(T_1)
        T_1_to_2 = np.dot(T_1_inv, T_2)

        s_transform = matrix_to_coordinate(T_1_to_2)

        return p_transform, s_transform

    def set_cps(self, cps):
        self.cps = cps

    def populate_control_space(self, cs_file):
        ignore_cp = range(12, 18)
        cs_df = pd.read_csv(cs_file)
        grouped = cs_df.groupby(['cp_id'])
        print("Num of contact pose: ", len(grouped.groups.keys()))
        for idx, g_key in enumerate(grouped.groups.keys()):
            group = grouped.get_group(g_key).reset_index(drop=True)
            self.cp_dict[idx] = g_key
            print(g_key)
            if g_key in ignore_cp:
                continue
            for index, row in group.iterrows():
                self.U.append(
                    (g_key, [row['trans_x'], row['trans_y'], row['rot'], row['deformation']]))

        # (-0.29374998807907104, -0.11874999850988388, 1.5707963267948966)
        # (-0.29374998807907104, 2.7755575615628914e-17, 1.5707963267948966)
        # (-0.29374998807907104, 0.09375, 1.5707963267948966)
        # (-0.23125000298023224, -0.18125000596046448, 0.0)
        # (-0.23125000298023224, 0.18125000596046448, 0.0)
        # (0.0, -0.18125000596046448, 0.0)
        # (0.0, 0.18125000596046448, 0.0)
        # (0.20624999701976776, -0.18125000596046448, 0.0)
        # (0.20624999701976776, 0.18125000596046448, 0.0)
        # (0.29374998807907104, -0.11874999850988388, 1.5707963267948966)
        # (0.29374998807907104, 2.7755575615628914e-17, 1.5707963267948966)
        # (0.29374998807907104, 0.08749999850988388, 1.5707963267948966)

        # print(self.cp_dict[3][2])
        # self.cp_dict[3][2] = np.pi
        # self.cp_dict[5][2] = np.pi
        # self.cp_dict[7][2] = np.pi

        # self.cp_dict[9][2] = -np.pi/2
        # self.cp_dict[10][2] = -np.pi/2
        # self.cp_dict[11][2] = -np.pi/2

        # convert control from world to local frame of gripper
        # for i in range(len(self.U)):
        #     (cp_id, u) = self.U[i]
        #     theta = self.cp_dict[cp_id][2]
        #     R_w_2_p = np.array([[np.cos(theta), -np.sin(theta)],
        #                             [np.sin(theta), np.cos(theta)]])
        #     R_p_2_w = np.linalg.inv(R_w_2_p)

        #     v_p = np.dot(R_p_2_w, u[0:2])
        #     u_p = [v_p[0], v_p[1], u[2]]

        #     self.U[i] = (cp_id, u_p)

        # append transformation of cloth
        for i in range(len(self.U)):
            (cp_id, u) = self.U[i]


            T_w_2_p = coordinate_to_matrix(
                self.cps[cp_id][0], self.cps[cp_id][2], self.cps[cp_id][3])
            T_u = coordinate_to_matrix(u[0], u[1], u[2])
            T_p_2_w = np.linalg.inv(T_w_2_p)

            T_w_2_o_ = np.dot(np.dot(T_w_2_p, T_u), T_p_2_w)

            s_transform = matrix_to_coordinate(T_w_2_o_)

            if u[3] == 0.09363753940417882:
                print("all cps: ", self.cps)
                print("cp_id: ", cp_id)
                print("contact pose: ", self.cps[cp_id])
                print("u: ", u)
                print("s_transform: ", s_transform)
                

                P_ = np.dot(T_w_2_p, T_u)
                p_ = matrix_to_coordinate(P_)
                print("p_: ", p_)

            # plot the initial contour of cloth, the contact pose self.cps[cp_id] and the transformed contour of s_transform
            # if u[3] == 0.10553718993341986:
            #     corner_pos = [[-self.object_size[0]/2, self.object_size[1]/2],
            #                     [self.object_size[0]/2, self.object_size[1]/2],
            #                     [self.object_size[0]/2, -self.object_size[1]/2],
            #                     [-self.object_size[0]/2, -self.object_size[1]/2]]
            #     corner_pos = np.array(corner_pos)
            #     corner_pos[:, 0] += self.cps[cp_id][0]
            #     corner_pos[:, 1] += self.cps[cp_id][1]
            #     plt.scatter(corner_pos[:, 0], corner_pos[:, 1])
            #     plt.scatter(self.cps[cp_id][0], self.cps[cp_id][1])
            #     corner_pos[:, 0] += s_transform[0]
            #     corner_pos[:, 1] += s_transform[1]
            #     plt.scatter(corner_pos[:, 0], corner_pos[:, 1])

            self.U[i] = (cp_id, u, s_transform)

        # init step_box, which is the max absolute value of the control space
        # max_x, max_y, max_r = 0, 0, 0
        # for (cp_id, u, s_transform) in self.U:
        #     if abs(s_transform[0]) > max_x:
        #         max_x = abs(s_transform[0])
        #     if abs(s_transform[1]) > max_y:
        #         max_y = abs(s_transform[1])
        #     if abs(s_transform[2]) > max_r:
        #         max_r = abs(s_transform[2])
        # self.B = [max_x, max_y, max_r]
    def moving_distance(self, pos_1, pose_2):
        dist = np.linalg.norm(np.array(pos_1)[:2] - np.array(pose_2)[:2])
        return dist

    def distance(self, pos_1, pos_2):
        s1_tilde = [(pos_1[0] - limit[0][0])/(limit[0][1] - limit[0][0]),
                    (pos_1[1] - limit[1][0])/(limit[1][1] - limit[1][0]),
                    (pos_1[2] - limit[2][0])/(limit[2][1] - limit[2][0])]

        s2_tilde = [(pos_2[0] - limit[0][0])/(limit[0][1] - limit[0][0]),
                    (pos_2[1] - limit[1][0])/(limit[1][1] - limit[1][0]),
                    (pos_2[2] - limit[2][0])/(limit[2][1] - limit[2][0])]
        return 0.1*math.sqrt((s1_tilde[0] - s2_tilde[0]) ** 2 + (s1_tilde[1] - s2_tilde[1]) ** 2) + 0.9*abs(s1_tilde[2] - s2_tilde[2])

    def nearest(self, state):
        distances = [self.distance(state.pos, n.pos) for n in self.nodes]
        nearest_idx = np.argmin(distances)
        return self.nodes[nearest_idx]

        # get the minimum cost, and the nodes with the minimum cost
        # min_cost = float('inf')
        # min_nodes = []
        # for n in self.nodes:
        #     if n.cost < min_cost:
        #         min_cost = n.cost
        # for n in self.nodes:
        #     if n.cost == min_cost:
        #         min_nodes.append(n)
        # # get the nearest node among the nodes with the minimum cost
        # distances = [self.distance(state.pos, n.pos) for n in min_nodes]
        # nearest_idx = np.argmin(distances)
        # return min_nodes[nearest_idx]

    def f(self, s, u):
        cp_id = u[0]
        # T_s_2_p = coordinate_to_matrix(self.cps[cp_id][0], self.cps[cp_id][1], self.cps[cp_id][2])
        S = coordinate_to_matrix(s.pos[0], s.pos[1], s.pos[2])
        # U = coordinate_to_matrix(u[1][0], u[1][1], u[1][2])
        # T_p_2_s = np.linalg.inv(T_s_2_p)

        # S_ = np.dot(S, np.dot(np.dot(T_s_2_p, U), T_p_2_s))

        S_ = np.dot(S, coordinate_to_matrix(u[2][0], u[2][1], u[2][2]))

        pos_new = matrix_to_coordinate(S_)

        # Return new_state and control
        s_new = State(pos_new)
        s_new.cp_id = u[0]
        s_new.control = u
        s_new.parent = s
        # if the same contact pose, just extend a M, else extend a composite action (GMR + Refinement)
        # s_new.cost = s.cost + w2 if u[0] == s.cp_id else s.cost + w0 + w1 + w2
        s_new.cost = s.cost + self.c(s, s_new)

        return s_new

    def steer(self, s1, s2):
        # get tranformation from s1 to s2
        T_1 = coordinate_to_matrix(s1.pos[0], s1.pos[1], s1.pos[2])
        T_2 = coordinate_to_matrix(s2.pos[0], s2.pos[1], s2.pos[2])

        T_1_inv = np.linalg.inv(T_1)
        T_1_to_2 = np.dot(T_1_inv, T_2)

        s_delta = matrix_to_coordinate(T_1_to_2)

        # Find the closest s_transform to s_delta
        # if s2.cp_id == -1:
        #     print("new sample")

        control_set = self.U if s2.cp_id == - \
            1 else [u for u in self.U if u[0] == s2.cp_id]
        distances = [self.distance(s_delta, s_transform)
                                   for (cp_id, u, s_transform) in control_set]
        closest_u_idx = np.argmin(distances)
        control = control_set[closest_u_idx]

        # u_new_w = np.dot(rotation_matrix, u_new[2])

        # Find the closest state to s1 + u_new

        s_new = self.f(s1, control)

        # if the same contact pose, just extend a M, else extend a composite action (GMR + Refinement)
        return s_new, control

    def within_cs(self, state):

        # # get the corner positions when the cloth is at the origin with orientation of state.pos[2]
        corner_pos = [[-self.object_size[0]/2, self.object_size[1]/2],
                        [self.object_size[0]/2, self.object_size[1]/2],
                        [self.object_size[0]/2, -self.object_size[1]/2],
                        [-self.object_size[0]/2, -self.object_size[1]/2]]
        
        # for each corner position, add with self.object_center_offset
        corner_pos = np.array(corner_pos)
        corner_pos[:, 0] -= self.object_center_offset[0]
        corner_pos[:, 1] -= self.object_center_offset[1]

        # rotate the corner positions
        rotation_matrix = np.array([[np.cos(state.pos[2]), -np.sin(state.pos[2])],
                                    [np.sin(state.pos[2]), np.cos(state.pos[2])]])
        corner_pos = np.dot(rotation_matrix, np.array(corner_pos).T).T

        # translate the corner positions
        corner_pos[:, 0] += state.pos[0]
        corner_pos[:, 1] += state.pos[1]

        # # calculate the distance between the corner positions and origin
        dist = np.linalg.norm(corner_pos, axis=1).max()
        if dist > self.reachable_length:
            return False

        # check if state is within limit
        # if (state.pos < np.array(limit)[:, 0]).any() or (state.pos > np.array(limit)[:, 1]).any():
        #     return False

        # # check if the corner positions are within the limit and the distance is within the reachable length
        # or dist > self.reachable_length
        if (corner_pos < np.array(limit)[:2, 0]).any() or (corner_pos > np.array(limit)[:2, 1]).any():
            return False

        return True

    def near_vertices(self, n):
        # find the neighbors of the new node that is within B = [1,1,1] of the new node, where B specify helf length of the box
        neighbors = []
        for node in self.nodes:
            if abs(node.pos[0] - n.pos[0]) <= self.B[0] and abs(node.pos[1] - n.pos[1]) <= self.B[1] and abs(node.pos[2] - n.pos[2]) <= self.B[2]:
                neighbors.append(node)
        return neighbors

    def close(self, s1, s2):
        if abs(s1.pos[2] - s2.pos[2]) < delta[2]:
            return True
        else:
            return False

    def repropagate(self, s, s_new):
        for child in s.children[:]:
            s_new_ = self.f(s_new, child.control)
            if (not self.constrained or self.within_cs(s_new_)):
                self.repropagate(child, s_new_)
                s_new.children.append(s_new_)
            s.children.remove(child)
        # print("remove s: ", s.pos)
        self.nodes.remove(s)
        self.nodes.append(s_new)

    def print_tree(self, state):
        print(state.pos)
        for child in state.children:
            self.print_tree(child)

    def get_cps_from_state(self, state, cp_id=None):
        if cp_id == None:
            cp = self.cps[state.cp_id]
        else:
            cp = self.cps[cp_id]
        P = coordinate_to_matrix(cp[0], cp[2], cp[3])
        S = coordinate_to_matrix(state.pos[0], state.pos[1], state.pos[2])
        P_ = np.dot(S, P)
        pos = matrix_to_coordinate(P_)
        return pos

    def c(self, state1, state2):
        if state1.cp_id == state2.cp_id:
            # moving distance
            pos_from = self.get_cps_from_state(state1)
            pos_to = self.get_cps_from_state(state2)
            dist = self.moving_distance(pos_from, pos_to)
        else:
            pos_from = self.get_cps_from_state(state1)
            pos_to = self.get_cps_from_state(state1, state2.cp_id)
            dist1 = self.moving_distance(pos_from, pos_to)
            pos_from = pos_to
            pos_to = self.get_cps_from_state(state2)
            dist2 = self.moving_distance(pos_from, pos_to)
            dist = dist1 + dist2
        # print("dist: ", dist)
        # print("deformation: ", state2.control[1][3])
        return w1*dist + w2*state2.control[1][3]

    def plan(self):
        
        for i in range(self.max_iterations):
            print("Iteration: ", i)
            # sample a random state
            # self.goal_rate of sample the goal state
            if random.random() < self.goal_rate:
                rand = self.goal.pos
            else:
                rand = [random.uniform(limit[0][0], limit[0][1]), random.uniform(limit[1][0], limit[1][1]), random.uniform(limit[2][0], limit[2][1])]
            s_rand = State(rand)

            # find the nearest node to the random state
            s_nearest = self.nearest(s_rand)

            # steer from the nearest node to the random state
            s_new, u_new = self.steer(s_nearest, s_rand)

            # check if the new node is within the configuration space
            if self.constrained and not self.within_cs(s_new):
                continue
            # print("num of state in tree: ", len(self.nodes))

            s_min = s_nearest
            J_min = s_new.cost
            S_near = self.near_vertices(s_new)
            s_add = s_new

            # extend the tree
            for s_near in S_near:
                s_new_, u_new_ = self.steer(s_near, s_new)
                new_cost = s_near.cost + self.c(s_near, s_new_)
                if self.close(s_new_, s_new) and (not self.constrained or self.within_cs(s_new_)) and new_cost < J_min:
                    # print("reconnect--")
                    J_min = new_cost
                    s_min = s_near
                    s_add = s_new_

            # add the new node to the tree
            self.nodes.append(s_add)
            s_min.children.append(s_add)

            # rewire the tree
            for s_near in S_near:
                if s_near == s_min or s_near not in self.nodes:
                    continue
                # have to use the same cp_id to steer
                # s_add has to approach s_near with the same cp_id
                s_near_, _ = self.steer(s_add, s_near)

                cost_add = self.c(s_add, s_near_)

                # if u_near_[0] == s_add.cp_id, then moving distance from same cp
                new_cost = s_add.cost + cost_add

                # new_cost = s_add.cost + w2 if u_near_[0] == s_add.cp_id else s_add.cost + w0 + w1 + w2
                if self.close(s_near_, s_near) and (not self.constrained or self.within_cs(s_near_)) and new_cost < s_near.cost:
                    # print("rewire--")
                    # print(s_near.cost, new_cost)
                    # print(s_near.pos)
                    # s_near.parent.children.remove(s_near)
                    try:
                        s_near.parent.children.remove(s_near)
                    except ValueError:
                        print("s_near not found in the list")
                    # print("Print tree ----")
                    # self.print_tree(s_near)
                    # print("end ---------")
                    self.repropagate(s_near, s_near_)
                    s_add.children.append(s_near_)



    def find_path(self):
        # get all the nodes that are close to the goal
        # close_nodes = []
        # for node in self.nodes:
        #     if self.close(node, self.goal):
        #         close_nodes.append(node)


        close_nodes = self.near_vertices(self.goal)

        print("Goal node: ", self.goal)
        print("Num of close nodes: ", len(close_nodes))

        # find the node with the minimum cost to the goal
        min_cost = float('inf')
        min_node = None
        for node in close_nodes:
            print("node pos: ", node.pos)
            print("node cost: ", node.cost)
            if node.cost != 0 and node.cost < min_cost:
                min_cost = node.cost
                min_node = node

        print("Optimal cost: ", min_cost)
        
        # back track to find the path
        path = []
        node = min_node

        if node is not None:
            # create a new node to store the goal state
            last_node = State(self.goal.pos)

            # set the parent of the goal node to the node with the minimum cost
            last_node.parent = node

            # only the cp_id of the goal node is used
            s_new, u_new = self.steer(node, last_node)
            u, st = self.find_u_and_delta_s(node, last_node, u_new[0])
            print(u_new)
            last_node.control = (u_new[0], [*u, u_new[1][3]], st)


            while last_node is not None:
                # if node.control is not None:
                #     print("cp_ID: ", node.control[0], "control: ", node.control[1])
                path.append(last_node)
                last_node = last_node.parent
        
        path.reverse()
        return path, min_cost
    
    def construct_control(self, u):
        control = {}
        control['cp_id'] = self.cp_dict[u[0]]
        control['trans_x'] = u[1][0]
        control['trans_y'] = u[1][1]
        control['rot'] = u[1][2]

        return control
                
def a_hard_code_path(self):
    # a executable path
    path = [
        State([0, 0, 0]),

    ]


def set_env(env_name):
    control_type = env_name


def plan(node_s, node_t, control_file, cps, constraints=None, save_plot=None, repeat=1):
    min_cost = float('inf')
    path = []
    for i in range(repeat):
        print("Repeat: ", i)

        rrt_star = MG_RRTStar(node_s, node_t)
        if constraints is not None:
            rrt_star.set_constraints(**constraints)
        # rrt_star.set_constraints(table_center=(0, -0.82), table_size=(1.4, 0.8), object_size=(0.4, 0.4))
        # if control_type == "TshirtPushPPP":
        #     rrt_star.populate_control_space("control_pants_50cm.csv")
        # elif control_type == "ClothPushPPP":
        rrt_star.set_cps(cps)
        rrt_star.populate_control_space(control_file)
        

        rrt_star.plan()
        this_path, cost = rrt_star.find_path()

        if cost != 0 and cost < min_cost:
            min_cost = cost
            path = this_path

    print("Path length: ", len(path))
    # print state and control of path
    # for state in path:
    #     print("state: ", state.pos, "control: ", state.control)

    # draw the path in cs
    # use color of path to indicate the contact pose
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(limit[0][0], limit[0][1])
    ax.set_ylim3d(limit[1][0], limit[1][1])
    ax.set_zlim3d(limit[2][0], limit[2][1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('theta')
    ax.set_title('Path in configuration space')

    # Set the view angle (rotate 180 degrees around x-axis)
    ax.view_init(elev=-145, azim=-125)  # Change elev=180 to azim=180 if you want to rotate around y-axis

    colors = ["red", "green", "blue", "purple", "orange", "pink", "brown", "gray", "olive", "yellow", "cyan", "magenta"]
    c_id = 0

    last_push_pose = None

    for i in range(len(path)):
        if i == 0:
            ax.scatter(path[i].pos[0], path[i].pos[1], path[i].pos[2], c='orange', marker='o', label='Start', s=50)
        elif i == 1:
            ax.scatter(path[i].pos[0], path[i].pos[1], path[i].pos[2], c='black', marker='x', label='Sub-goal')
        else:
            ax.scatter(path[i].pos[0], path[i].pos[1], path[i].pos[2], c='black', marker='x')

        current_push_pose = path[i].cp_id
        use_label = False
        if last_push_pose is not None and current_push_pose != last_push_pose:
            c_id += 1
            use_label = True
            # each different contact pose, label the path with the number of contact pose, ie. 1st, 2nd cp

        if i > 0:
            if use_label:
                ax.plot([path[i].pos[0], path[i-1].pos[0]], [path[i].pos[1], path[i-1].pos[1]], [path[i].pos[2], path[i-1].pos[2]], c=colors[c_id], label="CP: "+str(c_id))
            else:
                ax.plot([path[i].pos[0], path[i-1].pos[0]], [path[i].pos[1], path[i-1].pos[1]], [path[i].pos[2], path[i-1].pos[2]], c=colors[c_id])
        last_push_pose = current_push_pose

    # plot the goal, with larger size
    ax.scatter(node_t[0], node_t[1], node_t[2], c='g', marker='o', label='Goal', s=50)
    # legend
    # ax.legend()

    

    plt.show()

    # save the plot
    if save_plot is not None:
        plt.savefig(save_plot)


    # convert control from local frame of gripper to world frame
    # for i in range(len(path)):
    #     if path[i].control is not None:
    #         (cp_id, u, s_transform) = path[i].control
    #         T_w_2_p = rrt_star.coordinate_to_matrix(rrt_star.cp_dict[cp_id][0], rrt_star.cp_dict[cp_id][1], rrt_star.cp_dict[cp_id][2])
    #         u_w = np.dot(T_w_2_p, np.array([u[0], u[1], u[2]]))

    #         path[i].control = (cp_id, u_w, s_transform)

    for state in path:
        print("state: ", state.pos, "control: ", state.control)

    path_ = [([*state.pos], None if state.control is None else state.control[0]) for state in path]

    # substract each pose by node_s
    for i in range(len(path_)):
        path_[i][0][0] -= node_s[0]
        path_[i][0][1] -= node_s[1]
        path_[i][0][2] -= node_s[2]

    return path_

if __name__ == '__main__':

    start = (0,0,0)
    goal = (0,0.3,20)

    rrt_star = MG_RRTStar(start, goal)
    # rrt_star.populate_control_space("deform_data_small_step_ctl.csv")
    rrt_star.populate_control_space("control_pants_50cm_2.csv")

    rrt_star.plan()
    path = rrt_star.find_path()

    print("Path length: ", len(path))

    # draw the path in cs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(limit[0][0], limit[0][1])
    ax.set_ylim3d(limit[1][0], limit[1][1])
    ax.set_zlim3d(limit[2][0], limit[2][1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('theta')
    ax.set_title('Path in configuration space')
    for i in range(len(path)):
        ax.scatter(path[i].pos[0], path[i].pos[1], path[i].pos[2], c='r', marker='o')
        if i > 0:
            ax.plot([path[i].pos[0], path[i-1].pos[0]], [path[i].pos[1], path[i-1].pos[1]], [path[i].pos[2], path[i-1].pos[2]], c='b')
    plt.show()

    # draw all explored states in cs
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(limit[0][0], limit[0][1])
    # ax.set_ylim3d(limit[1][0], limit[1][1])
    # ax.set_zlim3d(limit[2][0], limit[2][1])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('theta')
    # ax.set_title('Explored states in configuration space')
    # for node in rrt_star.nodes:
    #     ax.scatter(node.pos[0], node.pos[1], node.pos[2], c='r', marker='o')
    # plt.show()

    # draw path to a random explored state in cs
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(limit[0][0], limit[0][1])
    # ax.set_ylim3d(limit[1][0], limit[1][1])
    # ax.set_zlim3d(limit[2][0], limit[2][1])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('theta')
    # ax.set_title('Path to a random explored state in configuration space')
    # for i in range(len(path)):
    #     ax.scatter(path[i].pos[0], path[i].pos[1], path[i].pos[2], c='r', marker='o')
    #     if i > 0:
    #         ax.plot([path[i].pos[0], path[i-1].pos[0]], [path[i].pos[1], path[i-1].pos[1]], [path[i].pos[2], path[i-1].pos[2]], c='b')
    # rand_node = random.choice(rrt_star.nodes)
    # ax.scatter(rand_node.pos[0], rand_node.pos[1], rand_node.pos[2], c='g', marker='o')
    # rand_path = []
    # node = rand_node
    # while node is not None:
    #     rand_path.append(node)
    #     node = node.parent
    # rand_path.reverse()
    # for i in range(len(rand_path)):
    #     ax.scatter(rand_path[i].pos[0], rand_path[i].pos[1], rand_path[i].pos[2], c='g', marker='o')
    #     if i > 0:
    #         ax.plot([rand_path[i].pos[0], rand_path[i-1].pos[0]], [rand_path[i].pos[1], rand_path[i-1].pos[1]], [rand_path[i].pos[2], rand_path[i-1].pos[2]], c='g')
    # plt.show()






