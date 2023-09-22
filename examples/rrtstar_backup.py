import random
import math

class State:
    def __init__(self, node, cp_id):
        self.node = node  # a tuple (x, y, theta)
        self.cp_id = cp_id  # contact pose id
        self.reached = False
        self.cost = float('inf')
        self.parent = None
        self.control = None
        self.children = []

class RRTStar:
    def __init__(self, start_node, goal_node, w1, w2, p, alpha, max_iter=1000):
        self.start_node = start_node
        self.goal_node = goal_node
        self.w1 = w1
        self.w2 = w2
        self.p = p
        self.alpha = alpha
        self.max_iter = max_iter

        # self.start_node['contact_pose'] = 0
        self.start_state = State(self.start_node, 0)
        self.start_state.reached = True
        self.start_state.cost = 0
        self.nodes = [self.start_state]

        # Create a set of goal nodes with different contact poses
        self.goal_nodes = [State(self.goal_node, i) for i in range(1, 13)]

    def node_to_key(self, state):
        return f"{state.node['index_x']},{state.node['index_y']},{state.node['layer']},{state.cp_id}"

    # [ ... Skipping some code for brevity ... ]

    def distance(self, node1, node2):
        dx = node1['index_x'] - node2['index_x']
        dy = node1['index_y'] - node2['index_y']
        dz = node1['layer'] - node2['layer']
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def nearest_state(self, random_state):
        nearest_state = None
        min_distance = float('inf')
        for state in self.nodes:
            if state.reached:
                d = self.distance(state.node, random_state.node)
                if d < min_distance:
                    nearest_state = state
                    min_distance = d
        return nearest_state

    def steer(self, from_state, to_state):
        d = self.distance(from_state.node, to_state.node)
        if d <= self.step_size:
            return to_state
        else:
            dx = to_state.node['index_x'] - from_state.node['index_x']
            dy = to_state.node['index_y'] - from_state.node['index_y']
            dz = to_state.node['layer'] - from_state.node['layer']
            scale = self.step_size / d
            return State({'index_x': from_state.node['index_x'] + dx * scale,
                          'index_y': from_state.node['index_y'] + dy * scale,
                          'layer': from_state.node['layer'] + dz * scale},
                         to_state.cp_id)

    def rewire(self, new_state):
        for state in self.nodes:
            if state == new_state:
                continue
            if not self.is_collision_free(state.node, new_state.node):
                continue
            cost = state.cost + self.w1 * self.distance(state.node, new_state.node) + self.w2 * (state.cp_id != new_state.cp_id)
            if cost < new_state.cost:
                new_state.parent = state
                new_state.cost = cost
                state.children.append(new_state)

    def propagate_cost(self, parent_state):
        for child_state in parent_state.children:
            new_cost = parent_state.cost + self.w1 * self.distance(parent_state.node, child_state.node) + self.w2 * (parent_state.cp_id != child_state.cp_id)
            if new_cost < child_state.cost:
                child_state.parent = parent_state
                child_state.cost = new_cost
                self.propagate_cost(child_state)

    def find_path(self):
        for i in range(self.max_iter):
            if random.random() < self.alpha:
                random_state = random.choice(self.goal_nodes)
            else:
                random_node = {'index_x': random.randint(0, self.width - 1),
                               'index_y': random.randint(0, self.height - 1),
                               'layer': random.choice(range(-180, 181, 15))}
                random_cp_id = random.choice(range(1, 13))
                random_state = State(random_node, random_cp_id)

            nearest_state = self.nearest_state(random_state)
            if nearest_state is None:
                continue

            # new_state = self.steer(nearest_state, random_state)

            if not self.is_collision_free(nearest_state.node, new_state.node):
                continue

            if random.random() < self.p or all([state.reached for state in self.nodes if state.node == new_state.node]):
                # Keeping the same contact pose or all contact poses of this node have been reached
                reached_states = [state for state in self.nodes if state.node == new_state.node and state.reached]
                min_cost_state = min(reached_states, key=lambda state: state.cost)
                new_state.cp_id = min_cost_state.cp_id
                new_state.cost = min_cost_state.cost + self.w2
                new_state.parent = min_cost_state
                new_state.control = min_cost_state.control
            else:
                # Switch to a different contact pose
                unreached_states = [state for state in self.nodes if state.node == new_state.node and not state.reached]
                if unreached_states:
                    new_state = random.choice(unreached_states)
                    new_state.reached = True
                    new_state.cost = nearest_state.cost + self.w2
                    new_state.parent = nearest_state
                    new_state.control = None  # update this with the selected control

            self.nodes.append(new_state)
            self.rewire(new_state)
            self.propagate_cost(new_state)

            if new_state in self.goal_nodes and new_state.reached:
                path = [new_state]
                current_state = new_state
                while current_state != self.start_node:
                    current_state = current_state.parent
                    path.append(current_state)
                path.reverse()
                return path

        return None
