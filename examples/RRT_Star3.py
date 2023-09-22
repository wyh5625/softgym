import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

class State:
    def __init__(self, node, cp_id):
        self.node = node  # a tuple (x, y, theta)
        self.cp_id = cp_id  # contact pose id
        self.reached = False
        self.cost = float('inf')
        self.parent = None
        self.control = None
        self.children = []
        self.num_of_grasp = float('inf')
        # self.used_controls = set()
        self.is_expanded = False

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

class MultiLayerDict:
    def __init__(self, area_size=1, grid_step_size=0.1, layer_step_size=5, pose_step_size=1):
        self.area_size = area_size
        self.grid_step_size = grid_step_size
        self.layer_step_size = layer_step_size
        self.pose_step_size = pose_step_size
        self.width = int(area_size / grid_step_size)
        self.height = int(area_size / grid_step_size)
        
        # access a state: eg. multiLayer[-15][i][j][cp_id]
        self.transition_dict = {layer: self.create_empty_2d_array(self.width, self.height, layer) for layer in range(-180, 181, self.layer_step_size)}

    # your other methods (coord_to_index, index_to_coord, angle_to_layer, approximate_node_location) go here

    def generate_random_node(self):
        random_node = {'index_x': random.randint(0, self.width - 1),
                        'index_y': random.randint(0, self.height - 1),
                        'layer': random.choice(range(-180, 181, self.layer_step_size))}
        return random_node
    
    def state_to_key(self, state):
        return f"{state.node['index_x']},{state.node['index_y']},{state.node['layer']},{state.cp_id}"

    
    def index_to_coord(self, index_x, index_y):
        x = index_x * self.grid_step_size - self.area_size / 2
        y = index_y * self.grid_step_size - self.area_size / 2
        return x, y

    def create_empty_node():
        return {"incoming": [], "outgoing": [], "expanded": False}

    def create_empty_2d_array(self, width, height, layer):
        # each layer is a 2d array of nodes contraining states with 13 contact poses
        return [[[State({"index_x": index_x, "index_y": index_y, "layer": layer}, cp_id) for cp_id in range(12)] for index_y in range(height)] for index_x in range(width)]

    def angle_to_layer(self, angle):
        adjusted_angle = angle + self.layer_step_size/2
        approx_angle = int(adjusted_angle // self.layer_step_size) * self.layer_step_size
        return approx_angle

    def coord_to_index(self, x, y):
        index_x = int((x + self.area_size / 2) / self.grid_step_size)
        index_y = int((y + self.area_size / 2) / self.grid_step_size)
        # constrains the index to be within the area
        index_x = min(max(index_x, 0), self.width - 1)
        index_y = min(max(index_y, 0), self.height - 1)
        return index_x, index_y

    def config_to_node(self, config):
        index_x, index_y = self.coord_to_index(config[0], config[1])
        layer = self.angle_to_layer(config[2])
        return {"index_x": index_x, "index_y": index_y, "layer": layer}

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

class RRTStar:
    def __init__(self, w1=1, w2=20, alpha=0.5, rpho=0.3, max_iter=1000):
        self.w1 = w1    # cost of path
        self.w2 = w2    # cost of switching contact pose
        # alpha is for probability of choosing goal node
        self.alpha = alpha
        # rpho is for probability of switching cp_id
        self.rpho = rpho
        self.max_iter = max_iter
        
        self.frontier = {}

        # init self.states as a dictionary of {key: state}, for every states in this 4D space, 
        self.configuration_space = MultiLayerDict()

        self.control_space = {}

        # define grasp_layer
        self.grasp_layer = {}

        # define unused_controls
        self.unused_controls = {}
        

    def set_start_and_goal(self, start_node, goal_node):
        self.start_node = start_node
        self.goal_node = goal_node
        

    def propagate(self, node, u):
        next_node = self.configuration_space.next_node(node, [u['delta_x'], u['delta_y']], u['delta_r'])
        return next_node

    def init_control_space(self, cs_file):
        cs_df = pd.read_csv(cs_file)
        grouped = cs_df.groupby(['push_x', 'push_y', 'push_ori'])
        print("Num of group keys: ", len(grouped.groups.keys()))
        for idx, g_key in enumerate(grouped.groups.keys()):
            group = grouped.get_group(g_key).reset_index(drop=True)
            self.control_space[idx] = group

        # populate unused_controls
        for layer in self.configuration_space.transition_dict:
            for index_x in range(self.configuration_space.width):
                for index_y in range(self.configuration_space.height):
                    for cp_id, state in enumerate(self.configuration_space.transition_dict[layer][index_x][index_y]):
                        # initialize the control space for this state
                        self.unused_controls[state] = list(range(len(self.control_space[cp_id])))
            

    def node_to_key(self, node):
        return f"{node['index_x']},{node['index_y']},{node['layer']}"

    def eu_distance(self, node1, node2):
        dx = node1['index_x'] - node2['index_x']
        dy = node1['index_y'] - node2['index_y']
        dz = node1['layer'] - node2['layer']
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def distance(self, node1, node2):
        return self.eu_distance(node1, node2)
        # # Calculate the rotational difference
        # r = abs(node1['layer'] - node2['layer'])

        # # Calculate the Manhattan distance of (index_x, index_y)
        # d = abs(node1['index_x'] - node2['index_x']) + abs(node1['index_y'] - node2['index_y'])

        # if d != 0:
        #     # Calculate the distance as r/d
        #     return r / d
        # else:
        #     return float('inf')  # In case d is zero to avoid division by zero


    def nearest_node(self, node):
        # find the nearest node
        nearest_node_key = min(self.frontier.keys(), key=lambda key: self.eu_distance(self.frontier[key], node))
        return self.frontier[nearest_node_key]

    def random_control(self, state):
        control_space_df = self.control_space[state.cp_id]
        unused_controls = self.unused_controls.get(state)
        if not unused_controls:
            return None
        rand_index = random.choice(unused_controls)
        unused_controls.remove(rand_index)
        row = control_space_df.loc[rand_index]
        state.used_controls.add(rand_index)
        return row.to_dict()

    def all_controls(self, state):
        control_space_df = self.control_space[state.cp_id]
        control_list = control_space_df.to_dict('records')
        return control_list


    def propagate_cost(self, parent_state):
        for child_state in parent_state.children:
            new_cost = parent_state.cost + self.w1 if parent_state.cp_id == child_state.cp_id else parent_state.cost + self.w2
            if new_cost < child_state.cost:
                child_state.parent = parent_state
                child_state.cost = new_cost
                self.propagate_cost(child_state)


    def update_and_propagate_cost(self, new_state):
        # If new state has not been reached yet
        state = self.configuration_space.transition_dict[new_state.node['layer']][new_state.node['index_x']][new_state.node['index_y']][new_state.cp_id]
        if not state.reached:
            state.reached = True
            state.cost = new_state.cost
            state.parent = new_state.parent
            state.control = new_state.control

        # If new state has been reached before
        else:
            # If new cost is smaller
            if new_state.cost < state.cost:
                # Update cost and parent state
                state.cost = new_state.cost
                state.parent = new_state.parent
                state.control = new_state.control
                
                # Propagate cost
                self.propagate_cost(state)

    def select_grasp_layer_for_expansion(self):
        # Calculate total number of unused controls for each layer
        # print("num of states in each layer: ", {layer: len(self.grasp_layer[layer]) for layer in self.grasp_layer})

        total_unused_controls_per_layer = {layer: sum(len(self.unused_controls[state]) for state in self.grasp_layer[layer]) 
                                            for layer in self.grasp_layer}
        
        # print("total_unused_controls_per_layer: ", total_unused_controls_per_layer)

        # Calculate sum of all unused controls
        total_unused_controls = sum(total_unused_controls_per_layer.values())

        # Calculate probabilities for each layer
        layer_probabilities = {layer: total_unused_controls_per_layer[layer] / total_unused_controls
                            for layer in self.grasp_layer}

        # Make a list with layer number repeated according to its probability
        layer_list = [layer for layer in layer_probabilities for _ in range(int(layer_probabilities[layer]*100))]

        # Select a layer randomly
        selected_layer = random.choice(layer_list)

        return selected_layer

    # def select_grasp_layer_for_expansion2(self):
    #     # calculate total number of unexpanded states in all layers
    #     total_states = sum(len(states) for states in self.grasp_layer.values() if state.is_expanded == False)

    #     if total_states == 0:
    #         return None

    #     # calculate the relative number of unexpanded states in each layer as a weight
    #     weights = [len(states)/total_states for states in self.grasp_layer.values() if state.is_expanded == False]

    #     # select a layer with probability proportional to the number of states in the layer
    #     layers = list(self.grasp_layer.keys())
    #     chosen_layer = random.choices(layers, weights)[0]

    #     return chosen_layer

    def select_grasp_layer_for_expansion2(self):
        """A function to select which grasp layer to expand next based on a probability distribution over the layers."""

        # Calculate the number of unexpanded states in each layer
        unexpanded_states_per_layer = {grasp_num: sum(1 for state in states if not state.is_expanded) 
                                    for grasp_num, states in self.grasp_layer.items()}

        # Calculate the total number of unexpanded states
        total_unexpanded_states = sum(unexpanded_states_per_layer.values())

        # Check if there are no unexpanded states left
        if total_unexpanded_states == 0:
            return None

        # Calculate the probability of selecting each layer
        layer_probabilities = {grasp_num: unexpanded_states / total_unexpanded_states 
                            for grasp_num, unexpanded_states in unexpanded_states_per_layer.items()}

        # Select a layer based on the calculated probabilities
        selected_layer = np.random.choice(list(layer_probabilities.keys()), p=list(layer_probabilities.values()))

        return selected_layer




    def get_min_cost_state(self):
        five_closest_nodes = heapq.nsmallest(5, self.frontier.values(), key=lambda node: self.distance(node, self.goal_node))

            # Find the node with the smallest cost among these 5 nodes
            # smallest_cost_node = min(five_closest_nodes, key=lambda node: node.cost)
            # Extract all states from these 5 nodes
        states = [state for node in five_closest_nodes for state in self.configuration_space.transition_dict[node['layer']][node['index_x']][node['index_y']]]

            # Find the state with the smallest cost
        smallest_cost_state = min(states, key=lambda state: state.cost)

        return smallest_cost_state

    def get_min_cost_path(self):
        # if the goal node is in the tree, return the path from start to goal, the goal state is the state with minimum cost
        if self.node_to_key(self.goal_node) in self.frontier:
            goal_state = min(self.configuration_space.transition_dict[self.goal_node['layer']][self.goal_node['index_x']][self.goal_node['index_y']], key=lambda state: state.cost)
            
            print('Goal found!')
            # print('Path cost: ', path[0][0].cost)
            print('Goal cost: ', goal_state.cost)

            cost = goal_state.cost

            path = []
            while goal_state is not None:
                path.append((goal_state.node, goal_state.control))
                goal_state = goal_state.parent

            print('Path length: ', len(path))

            return path[::-1], cost
        else:
            # find the nodes in the tree that are closest to the goal node
            closest_node = self.nearest_node(self.goal_node)
            closest_state = min(self.configuration_space.transition_dict[closest_node['layer']][closest_node['index_x']][closest_node['index_y']], key=lambda state: state.cost)
            
            print('Nearest goal found!')
            print('Nearest goal cost: ', closest_state.cost)

            cost = closest_state.cost

            path = []
            while closest_state is not None:
                path.append((closest_state.node, closest_state.control))
                closest_state = closest_state.parent

            print('Path length: ', len(path))
            
            return path[::-1], cost


    def get_closest_reached_state(self):
        close_nodes = [node for node in self.frontier.values() if self.grid_distance(node, self.goal_node) <= 5]
        min_cost_state = None

        for node in close_nodes:
            reached_states = [state for state in self.configuration_space.transition_dict[node['layer']][node['index_x']][node['index_y']] if state.reached]
            if reached_states:
                local_min_state = min(reached_states, key=lambda state: state.cost)
                if min_cost_state is None or local_min_state.cost < min_cost_state.cost:
                    min_cost_state = local_min_state

        return min_cost_state


    def grid_distance(self, node1, node2):
        dx = abs(node1['index_x'] - node2['index_x'])
        dy = abs(node1['index_y'] - node2['index_y'])
        dz = abs(node1['layer'] - node2['layer']) / 15
        return dx + dy + dz

    def expand_state(self, state_to_expand):
        state_to_expand.is_expanded = True

        controls = self.all_controls(state_to_expand)

        for u in controls:
            new_node = self.propagate(state_to_expand.node, u)  # Function 'propagate' must be implemented based on your specific motion model.
            if new_node is None:  # propagation failed
                continue

            new_node_key = self.node_to_key(new_node)

            # print("new_node_key: ", new_node_key)
            # expand the tree
            if new_node_key not in self.frontier:
                self.frontier[new_node_key] = new_node
                
            new_state = State(new_node, state_to_expand.cp_id)
            new_state.parent = state_to_expand
            new_state.control = u
            new_state.num_of_grasp = state_to_expand.num_of_grasp
            new_state.cost = state_to_expand.cost + self.w1  # w1 is the cost for maintaining the same contact pose
            new_state.reached = True

            # get the state in the configuration space
            state = self.configuration_space.transition_dict[new_state.node['layer']][new_state.node['index_x']][new_state.node['index_y']][new_state.cp_id]
                
            # update the grasp_layer of the state
            if new_state.num_of_grasp < state.num_of_grasp:
                if state.num_of_grasp != float('inf'):
                    self.grasp_layer[state.num_of_grasp].remove(state)
                state.num_of_grasp = new_state.num_of_grasp
                self.grasp_layer[state.num_of_grasp].append(state)

            # update the cost of the state
            if state not in state_to_expand.children:
                state_to_expand.children.append(state)
                # use the new state to update the cs_dict
                self.update_and_propagate_cost(new_state)

    def find_path(self, duration, view_paths=False):
        # init tree
        self.frontier = {self.node_to_key(self.start_node): self.start_node}
        
        # Initialize grasp_layer
        self.grasp_layer = {1: []}

        # Initialize the grasp_layer[1] with 12 states on start node with cp_id from 0 to 11
        for cp_id in range(12):  # cp_ids from 0 to 11
            # get the states from dict
            start_state = self.configuration_space.transition_dict[self.start_node['layer']][self.start_node['index_x']][self.start_node['index_y']][cp_id]
            start_state.reached = True
            start_state.cost = self.w2
            start_state.num_of_grasp = 1

            self.grasp_layer[1].append(start_state)
        
        # cost_records = []  # To store cost records
        cost_of_path = []
        paths = []
        times = []

        interval = duration/10
        interval_count = 0

        overall_start_time = time.time()

        # record time
        start_time = time.time()

        itr = 0

        while True:

            # print("iteration: ", itr)
            itr += 1
            # choose a random node
            if random.random() < self.alpha:
                random_node = self.goal_node
            else:
                random_node = self.configuration_space.generate_random_node()

            # get the nearest frontier node
            # nearest_node = self.nearest_node(random_node)

            # print number of states in each grasp_layer
            # print("num of states in each layer: ", {layer: len(self.grasp_layer[layer]) for layer in self.grasp_layer})


            # Find the grasp layer to expand
            # # gamma probability for expanding the grasp_layer[1], (1-gamma) for expanding other grasp_layer that is more than 1.
            # if random.random() < self.gamma:
            #     grasp_layer_to_expand = self.grasp_layer[1]
            # else:
            #     prob = random.random()
            #     for j in range(2, len(self.grasp_layer)):
            #         if prob < ((1-self.gamma) ** (j - 1)):
            #             grasp_layer_to_expand = self.grasp_layer[j]
            #             break 
            
            layer_key = self.select_grasp_layer_for_expansion2()
            # print("layer_key: ", layer_key)
            grasp_layer_to_expand = self.grasp_layer[layer_key]

            # Filter states that still have unused controls
            unexpanded_state = [state for state in grasp_layer_to_expand if not state.is_expanded]
            # print("num of states with unused controls: ", len(unexpanded_state))
            if not unexpanded_state:
                continue

            # The state to expand is the one that is closest to the random_node
            state_to_expand = min(unexpanded_state, key=lambda state: self.distance(state.node, random_node))
            
            node_to_expand = state_to_expand.node
            reached_states = [state for state in self.configuration_space.transition_dict[node_to_expand['layer']][node_to_expand['index_x']][node_to_expand['index_y']] if state.reached]
            unreached_states = [state for state in self.configuration_space.transition_dict[node_to_expand['layer']][node_to_expand['index_x']][node_to_expand['index_y']] if not state.reached]

            # unused_controls = self.unused_controls.get(state_to_expand, [])
            # num_of_total_controls = len(self.control_space[state_to_expand.cp_id])
            # unused_ratio = len(unused_controls) / num_of_total_controls

            if unreached_states and random.random() < self.rpho:
                # Switch to a different contact pose, leading to a new reached state to the tree
                nearest_state = random.choice(unreached_states)

                # Find the state in reached_states that has the minimum cost
                min_cost_parent = min(reached_states, key=lambda state: state.cost)
                nearest_state.parent = min_cost_parent
                nearest_state.cost = min_cost_parent.cost + self.w2 # The cost of switching contact pose is w2
                nearest_state.reached = True
                nearest_state.control = None
                nearest_state.num_of_grasp = min_cost_parent.num_of_grasp + 1
                min_cost_parent.children.append(nearest_state)

                # check if the grasp_layer[nearest_state.num_of_grasp] exists
                if nearest_state.num_of_grasp not in self.grasp_layer:
                    self.grasp_layer[nearest_state.num_of_grasp] = []
                self.grasp_layer[nearest_state.num_of_grasp].append(nearest_state)

                state_to_expand = nearest_state

            # else:
                # keeping the same contact pose


            # generate a control
            # u = self.random_control(state_to_expand)
            # if u is None:  # No more controls available for this state
            #     continue

            # expand_node
            self.expand_state(state_to_expand)

            # Call get_min_cost_state every 1/10 of total iterations and record the cost

            # if i % (self.max_iter // 10) == 0:
            #     min_cost_state = self.get_closest_reached_state()
            #     if min_cost_state is None:
            #         cost_records.append(99)
            #     else:
            #         cost_records.append(min_cost_state.cost)

            # Check if we've passed an interval, and if so record the cost
            if time.time() - start_time >= interval:
                min_cost_path, cost = self.get_min_cost_path()
                paths.append(min_cost_path)
                cost_of_path.append(cost)
                times.append(int(time.time() - overall_start_time))

                print(f"Cost after {int(time.time() - overall_start_time)} seconds: {cost}")
                # min_cost_state = self.get_min_cost_state()
                # cost_records.append(cost)
                # Update the start_time to the current time
                start_time = time.time()

                interval_count += 1

                if interval_count == 10:
                    break
            
            # Check if we've passed the duration, if so, break the loop
            # if time.time() - overall_start_time >= duration:
            #     break
        
        # # record time
        # end_time = time.time()
        # print("time: ", end_time - start_time)


        
                
            # if the goal_node is not in the frontier
            # find the 5 most closest nodes in the frontier

        # At the end of find_path, plot cost over iterations
        plt.plot(times, cost_of_path)
        plt.xlabel('time (s)')
        plt.ylabel('minimum cost of path')
        plt.title('Minimum Cost over Iterations')
        plt.show()

        print("last record: ", cost_of_path[-1])


        # smallest_cost_state = self.get_closest_reached_state()
            
        # path = []
        # while smallest_cost_state is not None:
        #     path.append((smallest_cost_state, smallest_cost_state.control))
        #     smallest_cost_state = smallest_cost_state.parent

        # print('Path found!')
        # print('Path length: ', len(path))
        # print('Path cost: ', path[0][0].cost)

        # return path[::-1]

        if view_paths:
            visualize_paths(paths, times, cost_of_path, 10, 10)

        return paths[-1]


    @staticmethod
    def visualize_cs(ml_dict):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        count = 0

        x_ = []
        y_ = []
        z_ = []

        for layer, layer_data in ml_dict.items():
            # if layer != 0:
            #     continue
            for index_x, row_data in enumerate(layer_data):
                for index_y, state_list in enumerate(row_data):
                    # if state_list contains a state that is reached, plot the point as red color
                    if any(state.reached for state in state_list):
                        x_.append(index_x)
                        y_.append(index_y)
                        z_.append(layer)
        
        # print("How many edges: ", count)
        ax.scatter(x_, y_, z_, color="red", s=1)

        # ax.view_init(elev=15., azim=-75)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer (Orientation)')

        plt.show()

    def check_state_in_multiple_layers(self):
        for i in range(1, len(self.grasp_layer) + 1):
            for state in self.grasp_layer[i]:
                for j in range(i+1, len(self.grasp_layer) + 1):
                    if state in self.grasp_layer[j]:
                        print(f"State {state} exists in layer {i} and {j}")


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
        next_node, control = path[i + 1]

        # current_push_pose = (control['push_x'], control['push_y'], control['push_ori'])

        # Change color if push_pose has changed
        # if last_push_pose is not None and current_push_pose != last_push_pose:
        #     c_id += 1
        if control is None:
            # switch contact pose
            c_id += 1

        ax.plot([node["index_x"], next_node["index_x"]], [node["index_y"], next_node["index_y"]], [node["layer"], next_node["layer"]], color=colors[c_id])
        # last_push_pose = current_push_pose

    # ax.view_init(elev=15., azim=-75)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer (Orientation)')

    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_zlim(-180, 180)

    plt.show()

def visualize_paths(paths, times, cost_of_path, x_lim, y_lim):
    fig = plt.figure()

    for i in range(len(paths)):
        path = paths[i]

        print("paths length: ", len(paths))
        ax = fig.add_subplot(2, len(paths)//2, i+1, projection='3d')

        # Add the title to the subplot
        ax.set_title(f'Path {i+1} - Cost: {cost_of_path[i]}, Time: {times[i]}s')

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
            next_node, control = path[i + 1]

            # current_push_pose = (control['push_x'], control['push_y'], control['push_ori'])

            # Change color if push_pose has changed
            # if last_push_pose is not None and current_push_pose != last_push_pose:
            #     c_id += 1
            if control is None:
                # switch contact pose
                c_id += 1

            ax.plot([node["index_x"], next_node["index_x"]], [node["index_y"], next_node["index_y"]], [node["layer"], next_node["layer"]], color=colors[c_id])
            # last_push_pose = current_push_pose

        # ax.view_init(elev=15., azim=-75)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer (Orientation)')

        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_zlim(-180, 180)

    plt.show()


def plan(node_s, node_t):
    rrt_star = RRTStar(max_iter=10000)
    rrt_star.init_control_space("deform_data_small_step_ctl.csv")

    start_node = rrt_star.configuration_space.config_to_node(node_s)
    goal_node = rrt_star.configuration_space.config_to_node(node_t)

    rrt_star.set_start_and_goal(start_node, goal_node)

    path = rrt_star.find_path(10, True)

    return path




# the entry of the program
if __name__ == '__main__':
    
    rrt_star = RRTStar(max_iter=10000)
    rrt_star.init_control_space("deform_data_small_step_ctl.csv")
    start = (0,0,0)
    goal = (0,0,180)

    start_node = rrt_star.configuration_space.config_to_node(start)
    goal_node = rrt_star.configuration_space.config_to_node(goal)

    rrt_star.set_start_and_goal(start_node, goal_node)

    path = rrt_star.find_path()
    if path is not None:
        print("Path found!")
    
    rrt_star.check_state_in_multiple_layers()

    # plot the configuration space
    # rrt_star.visualize_cs(rrt_star.configuration_space.transition_dict)

    # plot the path
    print(path)
    visualize_path(path, 10, 10)

