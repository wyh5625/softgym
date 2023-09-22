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
    def __init__(self, area_size=0.5, grid_step_size=0.1, layer_step_size=15, pose_step_size=1):
        self.area_size = area_size
        self.grid_step_size = grid_step_size
        self.layer_step_size = layer_step_size
        self.pose_step_size = pose_step_size
        self.width = int(area_size / grid_step_size)
        self.height = int(area_size / grid_step_size)
        
        # access a state: eg. multiLayer[-15][i][j][cp_id]
        self.transition_dict = {layer: self.create_empty_2d_array(self.width, self.height, layer) for layer in range(-180, 181, layer_step_size)}

    # your other methods (coord_to_index, index_to_coord, angle_to_layer, approximate_node_location) go here

    def generate_random_node(self):
        random_node = {'index_x': random.randint(0, self.width - 1),
                        'index_y': random.randint(0, self.height - 1),
                        'layer': random.choice(range(-180, 181, 15))}
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
            group = grouped.get_group(g_key)
            self.control_space[idx] = group
            

    def node_to_key(self, node):
        return f"{node['index_x']},{node['index_y']},{node['layer']}"

    def distance(self, node1, node2):
        dx = node1['index_x'] - node2['index_x']
        dy = node1['index_y'] - node2['index_y']
        dz = node1['layer'] - node2['layer']
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def nearest_node(self, node):
        # find the nearest node
        nearest_node_key = min(self.frontier.keys(), key=lambda key: self.distance(self.frontier[key], node))
        return self.frontier[nearest_node_key]

    def random_control(self, state):
        control_space_df = self.control_space[state.cp_id]
        rand_index = random.choice(control_space_df.index.tolist())
        row = control_space_df.loc[rand_index]
        return row.to_dict()


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

    def grid_distance(self, node1, node2):
        dx = abs(node1['index_x'] - node2['index_x'])
        dy = abs(node1['index_y'] - node2['index_y'])
        dz = abs(node1['layer'] - node2['layer']) / 15
        return dx + dy + dz

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



    def find_path(self):
        # init tree
        self.frontier = {self.node_to_key(self.start_node): self.start_node}

        # Init the start states
        for cp_id in self.control_space.keys():
            state = self.configuration_space.transition_dict[self.start_node['layer']][self.start_node['index_x']][self.start_node['index_y']][cp_id]
            state.cost = self.w2
            state.reached = True

        cost_records = []  # To store cost records

        # record time
        start_time = time.time()

        for i in range(1, self.max_iter + 1):

            if i % (self.max_iter // 10) == 0:
                min_cost_state = self.get_closest_reached_state()
                if min_cost_state is None:
                    cost_records.append(99)
                else:
                    cost_records.append(min_cost_state.cost)
            
            print("iteration: ", i)

            # choose a random node
            if random.random() < self.alpha:
                random_node = self.goal_node
            else:
                random_node = self.configuration_space.generate_random_node()

            # get the nearest frontier node
            nearest_node = self.nearest_node(random_node)

            # get the frontier state
            reached_states = [state for state in self.configuration_space.transition_dict[nearest_node['layer']][nearest_node['index_x']][nearest_node['index_y']] if state.reached]
            unreached_states = [state for state in self.configuration_space.transition_dict[nearest_node['layer']][nearest_node['index_x']][nearest_node['index_y']] if not state.reached]


            

            # If there is no reached state ie. initially the tree has only the start node but none of its states is reached
            if random.random() > self.rpho or not unreached_states:
                # Keeping the same contact pose
                nearest_state = random.choice(reached_states)
            else:
                # Switch to a different contact pose, leading to a new reached state to the tree
                nearest_state = random.choice(unreached_states)

                # Find the state in reached_states that has the minimum cost
                min_cost_parent = min(reached_states, key=lambda state: state.cost)
                nearest_state.parent = min_cost_parent
                nearest_state.cost = min_cost_parent.cost + self.w2 # The cost of switching contact pose is w2
                nearest_state.reached = True
                nearest_state.control = None
                min_cost_parent.children.append(nearest_state)
            
            print("nearest_state cost: ", nearest_state.cost)

            # sample control of cp_id
            u = self.random_control(nearest_state)
            new_node = self.propagate(nearest_state.node, u)  # Function 'f' must be implemented based on your specific motion model.
            
            if new_node is None:
                continue

            new_node_key = self.node_to_key(new_node)

            # expand the tree
            if new_node_key not in self.frontier:
                self.frontier[new_node_key] = new_node
            
            new_state = State(new_node, nearest_state.cp_id)
            new_state.parent = nearest_state
            new_state.control = u
            new_state.cost = nearest_state.cost + self.w1
            new_state.reached = True

            # get the state in the configuration space
            state = self.configuration_space.transition_dict[new_state.node['layer']][new_state.node['index_x']][new_state.node['index_y']][new_state.cp_id]
            
            if state not in nearest_state.children:
                nearest_state.children.append(state)
                # use the new state to update the cs_dict
                self.update_and_propagate_cost(new_state)

        # record time
        end_time = time.time()
        print("time: ", end_time - start_time)

        # At the end of find_path, plot cost over iterations
        plt.plot(range(0, self.max_iter, self.max_iter // 10), cost_records)
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Cost')
        plt.title('Minimum Cost over Iterations')
        plt.show()

        print("last record: ", cost_records[-1])

        # if the goal node is in the tree, return the path from start to goal, the goal state is the state with minimum cost
        # if self.node_to_key(self.goal_node) in self.frontier:
        #     goal_state = min(self.configuration_space.transition_dict[self.goal_node['layer']][self.goal_node['index_x']][self.goal_node['index_y']], key=lambda state: state.cost)
        #     path = []
        #     while goal_state is not None:
        #         path.append((goal_state, goal_state.control))
        #         goal_state = goal_state.parent

            
        #     return path[::-1]
        # else:
            # # find the nodes in the tree that are closest to the goal node
            # closest_node = self.nearest_node(self.goal_node)
            # closest_state = min(self.configuration_space.transition_dict[closest_node['layer']][closest_node['index_x']][closest_node['index_y']], key=lambda state: state.cost)
            # path = []
            # while closest_state is not None:
            #     path.append((closest_state, closest_state.control))
            #     closest_state = closest_state.parent
        # five_closest_nodes = heapq.nsmallest(5, self.frontier.values(), key=lambda node: self.distance(node, self.goal_node))

        #     # Find the node with the smallest cost among these 5 nodes
        #     # smallest_cost_node = min(five_closest_nodes, key=lambda node: node.cost)
        #     # Extract all states from these 5 nodes
        # states = [state for node in five_closest_nodes for state in self.configuration_space.transition_dict[node['layer']][node['index_x']][node['index_y']]]

        #     # Find the state with the smallest cost
        # smallest_cost_state = min(states, key=lambda state: state.cost)

        min_cost_state = self.get_closest_reached_state()
            
        path = []
        while min_cost_state is not None:
            path.append((min_cost_state, min_cost_state.control))
            min_cost_state = min_cost_state.parent

        print('Path found!')
        print('Path length: ', len(path))
        print('Path cost: ', path[0][0].cost)

        return path[::-1]


    @staticmethod
    def visualize_cs(ml_dict):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # # Iterate through layers and nodes to collect node positions
        # x = []
        # y = []
        # z = []
        # for layer, layer_data in ml_dict.items():
        #     for index_x, row_data in enumerate(layer_data):
        #         for index_y, node_data in enumerate(row_data):
        #                 # px, py = graph.index_to_coord(index_x, index_y)
        #                 # if px == -0 and py == 0 and layer == -180:
        #                 #     print("target reachable")
        #                 # x.append(px)
        #                 # y.append(py)
        #                 # z.append(layer)
        #                 x.append(index_x)
        #                 y.append(index_y)
        #                 z.append(layer)

        # Plot the nodes as points
        # ax.scatter(x, y, z, alpha=0.5, s=1)

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


def visualize_path(path, x_lim, y_lim):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the nodes as points
    x = []
    y = []
    z = []
    for state, _ in path:
        x.append(state.node["index_x"])
        y.append(state.node["index_y"])
        z.append(state.node["layer"])

    ax.scatter(x, y, z)

    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "gray", "olive", "cyan", "magenta"]
    c_id = 0

    last_push_pose = None
    # Plot edges
    for i in range(len(path) - 1):
        state, _ = path[i]
        next_state, control = path[i + 1]

        # current_push_pose = (control['push_x'], control['push_y'], control['push_ori'])

        # Change color if push_pose has changed
        # if last_push_pose is not None and current_push_pose != last_push_pose:
        #     c_id += 1
        if control is None:
            # switch contact pose
            c_id += 1

        ax.plot([state.node["index_x"], next_state.node["index_x"]], [state.node["index_y"], next_state.node["index_y"]], [state.node["layer"], next_state.node["layer"]], color=colors[c_id])
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

    path = rrt_star.find_path()

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
    

    # plot the configuration space
    # rrt_star.visualize_cs(rrt_star.configuration_space.transition_dict)

    # plot the path
    print(path)
    visualize_path(path, 20, 20)

