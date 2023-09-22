import numpy as np
import math
import time
from collections import deque
import pickle
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import pandas as pd

def are_nodes_equal(node1, node2):
    if not set(node1.keys()) == set(node2.keys()):
        return False

    for key in node1.keys():
        value1 = node1[key]
        value2 = node2[key]

        if isinstance(value1, list) or isinstance(value1, dict):
            if not value1 == value2:
                return False
        elif value1 != value2:
            return False

    return True


def get_current_ori_change(rect1, rect2):
    # Calculate centroids of rectangles
    centroid1 = np.mean(rect1, axis=0)
    centroid2 = np.mean(rect2, axis=0)

    # Center points by subtracting centroids
    centered1 = rect1 - centroid1
    centered2 = rect2 - centroid2

    # Calculate covariance matrix
    covariance_matrix = np.dot(centered1.T, centered2)

    # Perform singular value decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Calculate angle of rotation in radians
    angle_radians = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # print("Rotation angle (degrees):", angle_degrees)
    return angle_degrees



# Define a function to calculate the center and orientation values
def calculate_center_orientation(row):
    # Split the result_points string into separate values
    curr_corners = ast.literal_eval(row['result_points'])
    center = np.mean(curr_corners, axis=0)

    init_corners = np.array([[-0.294345, -0.18128946],
                            [ 0.29434484, -0.18128933],
                            [ 0.29434484, 0.18128933],
                            [-0.29434484, 0.18128951]])
    ori_change = get_current_ori_change(init_corners, curr_corners)

    # Calculate the orientation value as the angle in degrees between the x-axis and the orientation vector

    return pd.Series({"delta_p": center, "delta_r": ori_change})

def calculate_center_orientation3(row):
    # Split the result_points string into separate values
    curr_corners = ast.literal_eval(row['result_points'])
    center = np.mean(curr_corners, axis=0)

    init_corners = np.array([[-0.294345, -0.18128946],
                            [ 0.29434484, -0.18128933],
                            [ 0.29434484, 0.18128933],
                            [-0.29434484, 0.18128951]])
    ori_change = get_current_ori_change(init_corners, curr_corners)

    # Calculate the orientation value as the angle in degrees between the x-axis and the orientation vector

    return center, ori_change

def calculate_center_orientation2(cornersPos):
    # Split the result_points string into separate values
    # curr_corners = ast.literal_eval(row['result_points'])
    # print("cornersPos: ", cornersPos)
    center = np.mean(cornersPos[:,[0,2]], axis=0)

    init_corners = np.array([[-0.294345, -0.18128946],
                            [ 0.29434484, -0.18128933],
                            [ 0.29434484, 0.18128933],
                            [-0.29434484, 0.18128951]])
    ori_change = get_current_ori_change(init_corners, cornersPos[:,[0,2]])

    # Calculate the orientation value as the angle in degrees between the x-axis and the orientation vector

    return center, ori_change



def get_current_ori_change2(curr_corners):

    init_corners = np.array([[-0.294345, -0.18128946],
                            [ 0.29434484, -0.18128933],
                            [ 0.29434484, 0.18128933],
                            [-0.29434484, 0.18128951]])
    ori_change = get_current_ori_change(init_corners, curr_corners)
    return ori_change
    

def process_dataframe(df, push_idx=None):
    # Filter out rows where 'deformation' is greater than 0.1
    # df = df[df['deformation'] <= 0.05]

    # print number of rows in df
    print("Num of rows: ", len(df))
    
    # Apply the function to each row of the DataFrame and add the resulting columns to the DataFrame
    df[["delta_p", "delta_r"]] = df.apply(calculate_center_orientation, axis=1)

    if push_idx is not None:
        grouped = df.groupby(['push_x', 'push_y', 'push_ori'])

        # Get the first group's key
        group_keys = list(grouped.groups.keys())

        print("Num of group keys: ", len(group_keys))
        print("Group keys: ")
        print(group_keys)

        push_pose = group_keys[push_idx]

        # Output the first group using the key
        group = grouped.get_group(push_pose)

        return group, push_pose
        
    else:
        return df
    
def test_database(env, action_sample_df, flat_state_dict):

    env.set_state(flat_state_dict)

    
    for _, action in action_sample_df.iterrows():
        env.set_state(flat_state_dict)

        init_pusher_pos = np.array([action['push_x'], 0, action['push_y'], action['push_ori']])
        end_pos = init_pusher_pos + \
                        np.array([action['trans_x'], 0, action['trans_y'], action['rot']])
        action_ = [*end_pos, 1.0]
        error = env.test_sample(init_pusher_pos, action_, record=False,
                                            img_size=720, save_video_dir='./data/')
        env.get_current_corner_pos()
        delta_p, delta_r = calculate_center_orientation2(env.cornerPos)
        print("delta_p database: ", action['delta_p'])
        print("delta_p sim: ", delta_p)
        print("delta_r database: ", action['delta_r'])
        print("delta_r sim: ", delta_r)
        # result_points_str = "[" + ", ".join(["[{}, {}]".format(p[0], p[1]) for p in env.cornerPos[:,[0,2]]]) + "]"
    

class MultiLayerTransitionGraph:
    def __init__(self, area_size=1.0, grid_step_size=0.1, layer_step_size=5):
        self.area_size = area_size
        self.grid_step_size = grid_step_size
        self.layer_step_size = layer_step_size
        self.width = int(area_size / grid_step_size)
        self.height = int(area_size / grid_step_size)
        self.transition_dict = {layer: self.create_empty_2d_array(self.width, self.height) for layer in range(-180, 181, layer_step_size)}
        self.frontier = deque([])
        self.max_delta_r = 0
        self.max_delta_t = 0

    @staticmethod
    def create_empty_node():
        return {"incoming": [], "outgoing": [], "expanded": False}

    def create_empty_2d_array(self, width, height):
        return [[self.create_empty_node() for _ in range(height)] for _ in range(width)]
    
    @staticmethod
    def create_action(push_x, push_y, push_ori, trans_x, trans_y, rot):
        return {"push_x":push_x, "push_y": push_y, "push_ori": push_ori, "trans_x": trans_x, "trans_y": trans_y, "rot": rot}
    
    def create_edge(self, node, action, deformation, delta_p, delta_r):
        next_layer = node['layer'] + self.angle_to_layer(delta_r)
        # get position of the center of the node
        node_position = (node['index_x'] * self.grid_step_size - self.area_size / 2 + self.grid_step_size / 2, node['index_y'] * self.grid_step_size - self.area_size / 2 + + self.grid_step_size / 2)
        # the next node position should be the current node position + delta_p rotated by layer
        next_node_position = (node_position[0] + delta_p[0] * np.cos(np.deg2rad(node['layer'])) - delta_p[1] * np.sin(np.deg2rad(node['layer'])), node_position[1] + delta_p[0] * np.sin(np.deg2rad(node['layer'])) + delta_p[1] * np.cos(np.deg2rad(node['layer'])))
        
        # if the next node position is outside of the area, return None
        if next_node_position[0] < -self.area_size / 2 or next_node_position[0] > self.area_size / 2 or next_node_position[1] < -self.area_size / 2 or next_node_position[1] > self.area_size / 2:
            return None, None
        
        # get the index of the next node
        next_x, next_y = self.coord_to_index(next_node_position[0], next_node_position[1])

        # get the layer of the next node
        next_layer = self.angle_to_layer(node['layer'] + delta_r)

        if next_layer < -180 or next_layer > 180:
            return None, None

        next_node = {"layer": next_layer, "index_x": next_x, "index_y": next_y}
        # the next node may be the same as current node
        if are_nodes_equal(node, next_node):
            return None, None

        # may need to check if the next layer is between the start layer and the target layer
        outgoing_edge = {"action": action, "deformation": deformation, "next_node": next_node}
        incoming_edge = {"action": action, "deformation": deformation, "next_node": node}
        
        return outgoing_edge, incoming_edge
    
    def next_node(self, node, delta_p, delta_r):
        next_layer = node['layer'] + self.angle_to_layer(delta_r)
        # get position of the center of the node
        node_position = (node['index_x'] * self.grid_step_size - self.area_size / 2 + self.grid_step_size / 2, node['index_y'] * self.grid_step_size - self.area_size / 2 + + self.grid_step_size / 2)
        # the next node position should be the current node position + delta_p rotated by layer
        next_node_position = (node_position[0] + delta_p[0] * np.cos(np.deg2rad(node['layer'])) - delta_p[1] * np.sin(np.deg2rad(node['layer'])), node_position[1] + delta_p[0] * np.sin(np.deg2rad(node['layer'])) + delta_p[1] * np.cos(np.deg2rad(node['layer'])))
        
        # if the next node position is outside of the area, return None
        if next_node_position[0] < -self.area_size / 2 or next_node_position[0] > self.area_size / 2 or next_node_position[1] < -self.area_size / 2 or next_node_position[1] > self.area_size / 2:
            return None
        
        # get the index of the next node
        next_x, next_y = self.coord_to_index(next_node_position[0], next_node_position[1])

        # get the layer of the next node
        next_layer = self.angle_to_layer(node['layer'] + delta_r)

        if next_layer < -180 or next_layer > 180:
            return None

        next_node = {"layer": next_layer, "index_x": next_x, "index_y": next_y}
        # the next node may be the same as current node
        if are_nodes_equal(node, next_node):
            return None

        
        return next_node

    def add_edge(self, layer, x, y, edge, is_incoming):
        if is_incoming:
            self.transition_dict[layer][x][y]["incoming"].append(edge)
        else:
            self.transition_dict[layer][x][y]["outgoing"].append(edge)

    def coord_to_index(self, x, y):
        index_x = int((x + self.area_size / 2) / self.grid_step_size)
        index_y = int((y + self.area_size / 2) / self.grid_step_size)
        # constrains the index to be within the area
        index_x = min(max(index_x, 0), self.width - 1)
        index_y = min(max(index_y, 0), self.height - 1)
        return index_x, index_y
    
    def index_to_coord(self, index_x, index_y):
        x = index_x * self.grid_step_size - self.area_size / 2
        y = index_y * self.grid_step_size - self.area_size / 2
        return x, y
    
    def angle_to_layer(self, angle):
        adjusted_angle = angle + self.layer_step_size/2
        approx_angle = int(adjusted_angle // self.layer_step_size) * self.layer_step_size
        return approx_angle
    
    def approximate_node_location(self, x, y, angle):
        index_x, index_y = self.coord_to_index(x, y)
        layer = self.angle_to_layer(angle)
        return layer, index_x, index_y
    
    def nearest_expanded_node(self, node):
        # node = {"layer": layer, "index_x": x, "index_y": y}
        # check if the node is expanded
        if self.transition_dict[node["layer"]][node["index_x"]][node["index_y"]]["expanded"]:
            return node
        
        # find the nearest expanded node by go through every node in the transition_dict
        min_distance = float('inf')
        nearest_node = None
        for layer in range(-180, 180, self.layer_step_size):
            for x in range(self.width):
                for y in range(self.height):
                    if self.transition_dict[layer][x][y]["expanded"]:
                        distance = (layer - node["layer"])**2 + (x - node["index_x"])**2 + (y - node["index_y"])**2
                        if distance < min_distance:
                            min_distance = distance
                            nearest_node = {"layer": layer, "index_x": x, "index_y": y}
        return nearest_node
    
    def update_max_step_size(self, action_samples_df):
        for _, action_sample in action_samples_df.iterrows():
            if abs(action_sample['delta_r']) > self.max_delta_r:
                self.max_delta_r = abs(action_sample['delta_r'])
            delta_t = (action_sample['delta_p'][0]**2 + action_sample['delta_p'][1]**2)**0.5
            if delta_t > self.max_delta_t:
                self.max_delta_t = delta_t

    def expand_node(self, node, action_samples_df):
        # sample: ['push_x', 'push_y', 'push_ori', 'rot', 'trans_x', 'trans_y', 'deformation', 'result_points']
        # node = {"layer": layer, "index_x": x, "index_y": y}  
        if self.transition_dict[node["layer"]][node["index_x"]][node["index_y"]]["expanded"]:
            return
        
        self.transition_dict[node["layer"]][node["index_x"]][node["index_y"]]["expanded"] = True


        # for every action sample, expand the node to the next layer
        for _, action_sample in action_samples_df.iterrows():
            # (push_x, push_y, push_ori, trans_x, trans_y, rot)
            action = self.create_action(action_sample['push_x'], action_sample['push_y'], action_sample['push_ori'], 
                                        action_sample['trans_x'], action_sample['trans_y'], action_sample['rot'])
            deformation = action_sample['deformation']

            # Test the runtime of the create_edge function
            # start_time = time.monotonic()
            outgoing_edge, incoming_edge = self.create_edge(node, action, deformation, action_sample['delta_p'], action_sample['delta_r'])
            # end_time = time.monotonic()
            # print(f"create_edge runtime: {end_time - start_time:.6f} seconds")



            # If the next node is outside of the area or the same as node, return None
            # outgoing_edge, incoming_edge = self.create_edge(node, action, deformation, action_sample['delta_p'], action_sample['delta_r'])
            if outgoing_edge is None:
                continue
        
            next_node = outgoing_edge["next_node"]
            
            # add two edges in each direction
            self.add_edge(node['layer'], node['index_x'], node['index_y'], outgoing_edge, False)
            self.add_edge(next_node['layer'], next_node['index_x'], next_node['index_y'], incoming_edge, True)
            # search for every leaf node, and expand the leaf node

            # if the next node is not expanded
            if not self.transition_dict[next_node["layer"]][next_node["index_x"]][next_node["index_y"]]["expanded"]:
                # if the next node is not in the frontier, add it to the frontier
                not_in_frontier = True
                for f_node in self.frontier:
                    if are_nodes_equal(f_node, next_node):
                        not_in_frontier = False
                        break
                if not_in_frontier:
                    self.frontier.append(next_node)
        

    def expand_frontier(self, action_samples_df, max_expansion=None):
        # expand the frontier nodes, until its empty
        count = 0
        while len(self.frontier) > 0:
            # print("expand count: ", count)
            f_node = self.frontier.popleft()
            
            # Test the runtime of the expand_node function
            # start_time = time.monotonic()
            self.expand_node(f_node, action_samples_df)
            # end_time = time.monotonic()
            # print(f"expand_node runtime: {end_time - start_time:.6f} seconds")
            
            count += 1
            if max_expansion is not None and count > max_expansion:
                break

    # generate the graph with the start configuration
    def generate_graph(self, start_config, action_samples_df, one_step=False):
        # get the start node
        start_layer, start_index_x, start_index_y = self.approximate_node_location(start_config[0], start_config[1], start_config[2])
        start_node = {"layer": start_layer, "index_x": start_index_x, "index_y": start_index_y}
        # add the start node to the frontier
        self.frontier.append(start_node)
        # expand the frontier

        # Test the runtime of the expand_frontier function
        start_time = time.monotonic()
        if not one_step:
            self.expand_frontier(action_samples_df)
        else:
            self.expand_node(start_node, action_samples_df)
        end_time = time.monotonic()
        print(f"expand_frontier runtime: {end_time - start_time:.6f} seconds")

    def save_graph(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.transition_dict, file)

    def load_graph(self, filename):
        with open(filename, 'rb') as file:
            self.transition_dict = pickle.load(file)

        


    def node_distance(self, pos1, pos2, layer1, layer2, layer_weight=1, location_weight=1):
        layer_dist = abs(layer1 - layer2) * layer_weight
        loc_dist = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * location_weight
        return layer_dist + loc_dist


    # Optional: expand the frontier nodes, by looping through steps
    def find_closest_node_with_incoming_edges(self, layer_angle, index_x, index_y):
        
        min_distance = float("inf")
        closest_node = None
        # closest_node_position = None
        
        for angle, layer in self.transition_dict.items():
            for x in range(len(layer)):
                for y in range(len(layer[0])):
                    node = layer[x][y]
                    if node["incoming"]:
                        current_distance = self.node_distance((index_x, index_y), (x, y), layer_angle, angle)
                        if current_distance < min_distance:
                            min_distance = current_distance
                            closest_node = (angle, x, y)
                            # closest_node_position = (x, y)


        return closest_node[0], closest_node[1], closest_node[2]
    

    # def find_all_paths(self, start_node, target_node, path=None):
    #     if path is None:
    #         path = []

    #     if are_nodes_equal(start_node, target_node):
    #         yield path[::-1]
    #     else:
    #         for edge in self.transition_dict[target_node["layer"]][target_node["index_x"]][target_node["index_y"]]["incoming"]:
    #             parent_node = edge["next_node"]
    #             action = edge["action"]
    #             new_path = path.copy()
    #             new_path.append((target_node, action))
    #             yield from self.find_all_paths(start_node, parent_node, new_path)

    def find_all_paths(self, start_node, target_node, max_length=None):
        paths = []
        stack = [(target_node, [])]

        while stack:
            node, path = stack.pop()
            if are_nodes_equal(node, start_node):
                paths.append(path[::-1])
            else:
                if max_length is None or len(path) < max_length:
                    for edge in self.transition_dict[node["layer"]][node["index_x"]][node["index_y"]]["incoming"]:
                        parent_node = edge["next_node"]
                        action = edge["action"]
                        new_path = path.copy()
                        new_path.append((node, action))
                        stack.append((parent_node, new_path))

        return paths
    

    # def heuristic(self, node1, node2):
    #     position1 = self.index_to_coord(node1["index_x"], node1["index_y"])
    #     position2 = self.index_to_coord(node2["index_x"], node2["index_y"])
    #     euclidean_distance = ((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)**0.5

    #     orientation_diff = abs(node1["layer"] - node2["layer"])
    #     orientation_diff = min(orientation_diff, 360 - orientation_diff)

    #     # You can adjust the weight of orientation_diff to balance the importance of position and orientation
    #     return euclidean_distance + orientation_diff

    def init_sampled_actions(self, action_file):
        samples_db = pd.read_csv(action_file)
        samples_db = process_dataframe(samples_db)
        self.actions_db = samples_db
        self.update_max_step_size(samples_db)

    def init_sampled_actions2(self, action_db):
        self.actions_db = action_db
        self.update_max_step_size(action_db)

    
    def plan(self, start_config, target_config):
        s_layer, s_index_x, s_index_y = self.approximate_node_location(start_config[0], start_config[1], start_config[2])
        start_node = {"layer": s_layer, "index_x": s_index_x, "index_y": s_index_y}
        t_layer, t_index_x, t_index_y = self.approximate_node_location(target_config[0], target_config[1], target_config[2])   
        target_node = {"layer": t_layer, "index_x": t_index_x, "index_y": t_index_y}
        path = self.multi_grasp_a_star_search_in_expanding(start_node, target_node, self.actions_db, cp_weight=20)

        return path
    
    def find_path_from_target(self, start_node, target_node, max_length=None):
        def reconstruct_path(came_from, current_node, path):
            node_key = self.node_to_key(current_node)
            if node_key in came_from:
                parent_node, action = came_from[node_key]
                path.append((parent_node, action))
                return reconstruct_path(came_from, parent_node, path)
            return path
        
        def update_open_set(open_set, new_tuple):
            new_f_score, _, _, new_key = new_tuple
            found = False

            for i, (f_score, _, _, node_key) in enumerate(open_set):
                if node_key == new_key:
                    found = True
                    if new_f_score < f_score:
                        open_set[i] = new_tuple
                    break

            if not found:
                heapq.heappush(open_set, new_tuple)
            else:
                heapq.heapify(open_set)



        counter = 0
        open_set = [(0, counter, target_node, self.node_to_key(target_node))]
        came_from = {}
        g_score = {self.node_to_key(target_node): cp_weight}

        while open_set:
            _, _, current_node, current_key = heapq.heappop(open_set)

            if are_nodes_equal(current_node, start_node):
                return reconstruct_path(came_from, current_node, [])

            if max_length is None or len(came_from) < max_length:
                for edge in self.transition_dict[current_node["layer"]][current_node["index_x"]][current_node["index_y"]]["incoming"]:
                    parent_node = edge["next_node"]
                    action = edge["action"]
                    parent_key = self.node_to_key(parent_node)
                    tentative_g_score = g_score[current_key] + 1  # Assuming all edge costs are 1

                    if parent_key not in g_score or tentative_g_score < g_score[parent_key]:
                        came_from[parent_key] = (current_node, action)
                        g_score[parent_key] = tentative_g_score
                        f_score = tentative_g_score
                        # + self.heuristic(parent_node, start_node)

                        # check if parent_key is already in open_set
                        counter += 1
                        
                        update_tuple = (f_score, counter, parent_node, parent_key)
                        # heapq.heappush(open_set, (f_score, counter, parent_node))
                        update_open_set(open_set, update_tuple)

        return []  # No path found
    
    def heuristic(self, node, target_node):
        # Define your heuristic function here
        r_diff = abs(target_node["layer"] - node["layer"])
        # compute distance between node and target_node
        x_t, y_t = self.index_to_coord(target_node["index_x"], target_node["index_y"])
        x_n, y_n = self.index_to_coord(node["index_x"], node["index_y"])
        t_diff = ((x_t - x_n)**2 + (y_t - y_n)**2)**0.5

        return int(max(r_diff/self.max_delta_r, t_diff/self.max_delta_t))
        
    

    def a_star_search(self, start_node, target_node):
        def reconstruct_path(came_from, current_node):
            node_key = self.node_to_key(current_node)
            if node_key in came_from:
                parent_node, action = came_from[node_key]
                path = reconstruct_path(came_from, parent_node)
                path.append((current_node, action))
                return path
            return [(current_node, None)]

        # (f_score, unique_id, node, node_key)
        open_set = [(self.heuristic(start_node, target_node), 0, start_node, self.node_to_key(start_node))]
        came_from = {}
        g_score = {self.node_to_key(start_node): 0}
        counter = 0

        while open_set:
            _, _, current_node, current_key = heapq.heappop(open_set)

            if are_nodes_equal(current_node, target_node):
                return reconstruct_path(came_from, current_node)

            for edge in self.transition_dict[current_node["layer"]][current_node["index_x"]][current_node["index_y"]]["outgoing"]:
                child_node = edge["next_node"]
                action = edge["action"]
                child_key = self.node_to_key(child_node)
                tentative_g_score = g_score[current_key] + 1  # Assuming edge cost is stored in edge["cost"]

                if child_key not in g_score or tentative_g_score < g_score[child_key]:
                    came_from[child_key] = (current_node, action)
                    g_score[child_key] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(child_node, target_node)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, child_node, child_key))

        return []  # No path found

    def a_star_search_in_expanding(self, start_node, goal_node, action_samples_df):
        def reconstruct_path(came_from, current_node):
            node_key = self.node_to_key(current_node)
            if node_key in came_from:
                parent_node, action = came_from[node_key]
                path = reconstruct_path(came_from, parent_node)
                path.append((current_node, action))
                return path
            return [(current_node, None)]

        open_set = [(self.heuristic(start_node, goal_node), 0, start_node, self.node_to_key(start_node))]
        came_from = {}
        g_score = {self.node_to_key(start_node): 0}
        counter = 0

        explored_nodes = []
        counter1 = 0

        while open_set:
            _, _, current_node, current_key = heapq.heappop(open_set)
            distance_to_goal = self.node_distance((current_node['index_x'], current_node['index_y']), (goal_node['index_x'], goal_node['index_y']), current_node['layer'], goal_node['layer'])
            heapq.heappush(explored_nodes, (distance_to_goal, counter1, current_node))
            counter1 += 1

            if are_nodes_equal(current_node, goal_node):
                print("found target_node: ", goal_node)
                return reconstruct_path(came_from, current_node)
            
            # Expand the current node
            # for action in sample_actions:
            for _, action_sample in action_samples_df.iterrows():
                action = self.create_action(action_sample['push_x'], action_sample['push_y'], action_sample['push_ori'], 
                                        action_sample['trans_x'], action_sample['trans_y'], action_sample['rot'])
                neighbor_node = self.next_node(current_node, action_sample['delta_p'], action_sample['delta_r'])
                
                if neighbor_node is not None:
                    # Calculate the g_score for the neighbor node
                    tentative_g_score = g_score[current_key] + 1

                    neighbor_key = self.node_to_key(neighbor_node)
                    if neighbor_key not in g_score or tentative_g_score < g_score[neighbor_key]:
                        # This path to the neighbor node is better than any previous one. Record it!
                        came_from[neighbor_key] = (current_node, action)
                        g_score[neighbor_key] = tentative_g_score

                        f_score = tentative_g_score + self.heuristic(neighbor_node, goal_node)
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor_node, neighbor_key))

        # can exactly find a path to the goal node, but can find a path to a node that is close to the goal node
        # print("len(explored_nodes): ", len(explored_nodes))
        # for _ in range(100):
        #     _, _, close_node = heapq.heappop(explored_nodes)
        #     print("close_node: ", close_node)
        
        # _, _, close_node = heapq.heappop(explored_nodes)
        _, _, close_node = heapq.heappop(explored_nodes)
        print("close_node: ", close_node)
        print("target_node: ", goal_node)
        
        
        return reconstruct_path(came_from, close_node)

    def multi_grasp_a_star_search_in_expanding(self, start_node, goal_node, action_samples_df, cp_weight=5):
        def reconstruct_path(came_from, current_node, current_cp_key):
            node_cp_key = (self.node_to_key(current_node), current_cp_key)
            if node_cp_key in came_from:
                parent_node, parent_cp_key, action = came_from[node_cp_key]
                path = reconstruct_path(came_from, parent_node, parent_cp_key)
                path.append((current_node, action))
                return path
            return [(current_node, None)]

        # Initialize with cp as None
        start_cp = None
        open_set = [(self.heuristic(start_node, goal_node), 0, start_node, start_cp, self.node_to_key(start_node), start_cp)]
        came_from = {}
        g_score = {(self.node_to_key(start_node), start_cp): cp_weight}
        counter = 0

        explored_nodes = []
        counter1 = 0

        while open_set:
            fs, _, current_node, current_cp, current_key, current_cp_key = heapq.heappop(open_set)
            distance_to_goal = self.node_distance((current_node['index_x'], current_node['index_y']), (goal_node['index_x'], goal_node['index_y']), current_node['layer'], goal_node['layer'])
            heapq.heappush(explored_nodes, (distance_to_goal, counter1, current_node, current_cp_key))
            counter1 += 1

            print("estimated score: ", fs)

            # Check times that CP has changed
            # if (current_key, current_cp_key) in came_from:
            #     _, action = came_from[(current_key, current_cp_key)]
            #     last_cp = (action['push_x'], action['push_y'], action['push_ori'])
            # else:
            #     # cp_change_counter = 0
            #     last_cp = None  # Initialize last CP

            if are_nodes_equal(current_node, goal_node):
                print("target_node found: ", goal_node)
                print("cost of path(g_score): ", g_score[(self.node_to_key(goal_node), current_cp_key)])
                return reconstruct_path(came_from, current_node, current_cp_key)
            
            # Expand the current node
            # for action in sample_actions:
            for _, action_sample in action_samples_df.iterrows():
                action = self.create_action(action_sample['push_x'], action_sample['push_y'], action_sample['push_ori'], 
                                        action_sample['trans_x'], action_sample['trans_y'], action_sample['rot'])
                neighbor_node = self.next_node(current_node, action_sample['delta_p'], action_sample['delta_r'])
                
                if neighbor_node is not None:
                    # Check if the CP has changed
                    neighbor_cp = (action_sample['push_x'], action_sample['push_y'], action_sample['push_ori'])
                    # if last_cp is not None and current_cp != last_cp:
                    #     cp_change_counter += 1

                    neighbor_key = self.node_to_key(neighbor_node)
                    neighbor_cp_key = self.cp_to_key(neighbor_cp)

                    # Calculate the g_score for the neighbor node
                    # tentative_g_score = g_score[current_key] + 1
                    tentative_g_score = g_score[(current_key, current_cp_key)] + 1
                    if current_cp_key is not None and neighbor_cp_key != current_cp_key:
                        tentative_g_score = tentative_g_score + cp_weight * 1


                    
                    if (neighbor_key, neighbor_cp_key) not in g_score or tentative_g_score < g_score[(neighbor_key, neighbor_cp_key)]:
                        # This path to the neighbor node is better than any previous one. Record it!
                        came_from[(neighbor_key, neighbor_cp_key)] = (current_node, current_cp_key, action)
                        g_score[(neighbor_key, neighbor_cp_key)] = tentative_g_score

                        f_score = tentative_g_score + self.heuristic(neighbor_node, goal_node)
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor_node, neighbor_cp, neighbor_key, neighbor_cp_key))

        # can exactly find a path to the goal node, but can find a path to a node that is close to the goal node
        # print("len(explored_nodes): ", len(explored_nodes))
        # for _ in range(100):
        #     _, _, close_node = heapq.heappop(explored_nodes)
        #     print("close_node: ", close_node)
        
        # _, _, close_node = heapq.heappop(explored_nodes)
        _, _, close_node, close_cp_key = heapq.heappop(explored_nodes)
        print("close_node: ", close_node)
        print("target_node: ", goal_node)
        
        
        return reconstruct_path(came_from, close_node, close_cp_key)

    def node_to_key(self, node):
        return f"{node['layer']}_{node['index_x']}_{node['index_y']}"

    def cp_to_key(self, cp):
        return f"{cp[0]}_{cp[1]}_{cp[2]}"


    # need to provide a function for extracting center and orientation from input corners
    # need to process dataframe to get the center and orientation of samples and save it back to the dataframe
    # given the starting location and orientation, find the closest node, append node to frontier and expand the frontier
    # we assume the exploring start from the given configuration
    # after termination of expansion, find the node in the multi-layer graph that has incoming edges and is closest to the target configuration

    @staticmethod
    def visualize_graph(graph):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through layers and nodes to collect node positions
        x = []
        y = []
        z = []
        for layer, layer_data in graph.transition_dict.items():
            for index_x, row_data in enumerate(layer_data):
                for index_y, node_data in enumerate(row_data):
                    if node_data is not None and node_data["expanded"]:
                        px, py = graph.index_to_coord(index_x, index_y)
                        # if px == -0 and py == 0 and layer == -180:
                        #     print("target reachable")
                        x.append(px)
                        y.append(py)
                        z.append(layer)

        # Plot the nodes as points
        ax.scatter(x, y, z, alpha=0.5, s=1)

        # plot start node with green color
        start_node = (0,0,0)
        goal_node = (0,0,-180)
        # z0, x0, y0 = graph.approximate_node_location(start_node[0], start_node[1], start_node[2])
        # z1, x1, y1 = graph.approximate_node_location(goal_node[0], goal_node[1], goal_node[2])
        ax.scatter([start_node[0]], [start_node[1]], [start_node[2]], color="green")
        ax.scatter([goal_node[0]], [goal_node[1]], [goal_node[2]], color="red")
        

        count = 0

        # Plot edges
        for layer, layer_data in graph.transition_dict.items():
            # if layer != 0:
            #     continue
            for index_x, row_data in enumerate(layer_data):
                for index_y, node_data in enumerate(row_data):
                    if node_data is not None:
                        # if count > 1000:
                        #     break
                        for edge in node_data["outgoing"]:
                            next_node = edge["next_node"]
                            count += 1
                            # convert index to coordinates
                            px, py = graph.index_to_coord(index_x, index_y)
                            px_, py_ = graph.index_to_coord(next_node["index_x"], next_node["index_y"])

                            ax.plot([px, px_], [py, py_], [layer, next_node["layer"]], color="red")
                            # plot the line with color, alpha and linewidth
                            # ax.plot([px, px_], [py, py_], [layer, next_node["layer"]], color="blue", alpha=0.5, linewidth=0.5)

        # print("How many edges: ", count)

        # ax.view_init(elev=15., azim=-75)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer (Orientation)')

        plt.show()


    @staticmethod
    def visualize_graph2(graph):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through layers and nodes to collect node positions
        x = []
        y = []
        z = []
        ex = []
        ey = []
        ez = []

        for layer, layer_data in graph.transition_dict.items():
            for index_x, row_data in enumerate(layer_data):
                for index_y, node_data in enumerate(row_data):
                    if node_data is not None:
                        px, py = graph.index_to_coord(index_x, index_y)
                        # Check if the node has been expanded
                        if node_data["expanded"]:
                            ex.append(px)
                            ey.append(py)
                            ez.append(layer)
                        else:
                            x.append(px)
                            y.append(py)
                            z.append(layer)

        # Plot the non-expanded nodes as points
        ax.scatter(x, y, z, alpha=0.1, s=1)
        
        # Plot the expanded nodes in a different color
        ax.scatter(ex, ey, ez, color='red', alpha=0.5, s=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer (Orientation)')

        plt.show()

    @staticmethod
    def visualize_path(path, x_lim, y_lim):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the nodes as points
        x = []
        y = []
        z = []
        for node, _ in path:
            x.append(node["index_x"])
            y.append(node["index_y"])
            z.append(node["layer"])

        ax.scatter(x, y, z)

        colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "gray", "olive", "cyan", "magenta"]
        c_id = 0

        last_push_pose = None
        # Plot edges
        for i in range(len(path) - 1):
            node, _ = path[i]
            next_node, action = path[i + 1]

            current_push_pose = (action['push_x'], action['push_y'], action['push_ori'])

            # Change color if push_pose has changed
            if last_push_pose is not None and current_push_pose != last_push_pose:
                c_id += 1

            ax.plot([node["index_x"], next_node["index_x"]], [node["index_y"], next_node["index_y"]], [node["layer"], next_node["layer"]], color=colors[c_id])
            last_push_pose = current_push_pose

        # ax.view_init(elev=15., azim=-75)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer (Orientation)')

        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_zlim(-180, 180)

        plt.show()

    @staticmethod
    def visualize_path_on_graph(path, graph):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through layers and nodes to collect node positions
        x = []
        y = []
        z = []
        for layer, layer_data in graph.transition_dict.items():
            for index_x, row_data in enumerate(layer_data):
                for index_y, node_data in enumerate(row_data):
                    if node_data is not None:
                        # px, py = graph.index_to_coord(index_x, index_y)

                        x.append(index_x)
                        y.append(index_y)
                        z.append(layer)

        # Plot the nodes as points
        ax.scatter(x, y, z, alpha=0.1, s=1)

        count = 0

        # Plot edges
        for layer, layer_data in graph.transition_dict.items():
            # if layer != 0:
            #     continue
            for index_x, row_data in enumerate(layer_data):
                for index_y, node_data in enumerate(row_data):
                    if node_data is not None:
                        # if count > 1000:
                        #     break
                        for edge in node_data["outgoing"]:
                            next_node = edge["next_node"]
                            count += 1
                            # convert index to coordinates
                            px, py = index_x, index_y
                            px_, py_ = next_node["index_x"], next_node["index_y"]

                            ax.plot([px, px_], [py, py_], [layer, next_node["layer"]], color="gray")

        # print("How many edges: ", count)

        # Plot the nodes as points
        x = []
        y = []
        z = []
        for node, _ in path:
            x.append(node["index_x"])
            y.append(node["index_y"])
            z.append(node["layer"])

        ax.scatter(x, y, z)

        # Plot edges
        for i in range(len(path) - 1):
            node, _ = path[i]
            next_node, _ = path[i + 1]

            ax.plot([node["index_x"], next_node["index_x"]], [node["index_y"], next_node["index_y"]], [node["layer"], next_node["layer"]], color="red")

        # ax.view_init(elev=15., azim=-75)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer (Orientation)')
        
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([-180, 180])


        plt.show()
    

# Example usage
# multi_layer_motion_planner = MultiLayerTransitionGraph(area_size=2, grid_step_size=0.1, layer_step_size=15)

# action = ...
# connected_node_position = ...
# x, y = ...  # coordinates
# index_x, index_y = multi_layer_motion_planner.coord_to_index(x, y)
# multi_layer_motion_planner.add_edge(0, index_x, index_y, is_incoming=True, edge=multi_layer_motion_planner.create_edge(action, connected_node_position))
