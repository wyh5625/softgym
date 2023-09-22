from ompl import base
from ompl import geometric
from ompl import util

class MySteerFunction(oc.SteerFunction):
    def __init__(self, si):
        super().__init__(si)
        self.control_space = si.getControlSpace()
        self.simulator = oc.SimpleSimulator(si)

    def steer(self, from_state, to_state):
        control = self.control_space.allocVector()
        self.control_space.interpolate(from_state.getControl(), to_state.getControl(), 0.5, control)
        duration = self.simulator.propagate(from_state, control, 1.0, to_state)
        return to_state

class RRTStar:
    def __init__(self, start_node, goal_node, action_samples_df, max_iter=1000, goal_sample_rate=0.1, step_size=1.0):
        self.start_node = start_node
        self.goal_node = goal_node
        self.action_samples_df = action_samples_df
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.step_size = step_size
        self.nodes = [start_node]
        self.edges = {}
        self.costs = {self.node_to_key(start_node): 0}
        self.area_size = 2
        self.grid_step_size = 0.1
        self.layer_step_size = 15
        self.width = int(self.area_size / self.grid_step_size)
        self.height = int(self.area_size / self.grid_step_size)

    def node_to_key(self, node):
        return f"{node['index_x']},{node['index_y']},{node['layer']}"

    def coord_to_index(self, x, y):
        index_x = int((x + self.area_size / 2) / self.grid_step_size)
        index_y = int((y + self.area_size / 2) / self.grid_step_size)
        # constrains the index to be within the area
        index_x = min(max(index_x, 0), self.width - 1)
        index_y = min(max(index_y, 0), self.height - 1)
        return index_x, index_y

    def angle_to_layer(self, angle):
        adjusted_angle = angle + self.layer_step_size/2
        approx_angle = int(adjusted_angle // self.layer_step_size) * self.layer_step_size
        return approx_angle

    def create_action(self, push_x, push_y, push_ori, trans_x, trans_y, rot):
        return {'push_x': push_x, 'push_y': push_y, 'push_ori': push_ori, 'trans_x': trans_x, 'trans_y': trans_y, 'rot': rot}

    def next_node(self, current_node, delta_p, delta_r):
        next_node = current_node.copy()
        x = next_node['index_x'] + delta_p[0]
        y = next_node['index_y'] + delta_p[1]
        z = next_node['layer'] + delta_r

        # get the index of the next node
        next_x, next_y = self.coord_to_index(x, y)

        # get the layer of the next node
        next_layer = self.angle_to_layer(z)

        next_node['index_x'] = next_x
        next_node['index_y'] = next_y
        next_node['layer'] = next_layer

        if self.is_valid_node(next_node):
            return next_node
        else:
            return None

    def is_valid_node(self, node):
        # Check if the node is within the bounds of the environment
        if node['index_x'] < 0 or node['index_x'] > self.width - 1 or \
           node['index_y'] < 0 or node['index_y'] > self.height - 1 or \
           node['layer'] < -180 or node['layer'] > 180:
            return False

        # Check if the node collides with any obstacles
        # ...

        return True

    def distance(self, node1, node2):
        dx = node1['index_x'] - node2['index_x']
        dy = node1['index_y'] - node2['index_y']
        dz = node1['layer'] - node2['layer']
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def nearest_node(self, random_node):
        nearest_node = None
        min_distance = float('inf')
        for node in self.nodes:
            d = self.distance(node, random_node)
            if d < min_distance:
                nearest_node = node
                min_distance = d
        return nearest_node

    def steer(self, from_node, to_node):
        d = self.distance(from_node, to_node)
        if d <= self.step_size:
            return to_node
        else:
            dx = to_node['index_x'] - from_node['index_x']
            dy = to_node['index_y'] - from_node['index_y']
            dz = to_node['layer'] - from_node['layer']
            scale = self.step_size / d
            return {'index_x': from_node['index_x'] + dx * scale,
                    'index_y': from_node['index_y'] + dy * scale,
                    'layer': from_node['layer'] + dz * scale}

    def is_collision_free(self, from_node, to_node):
        # Check if the path between from_node and to_node collides with any obstacles
        # ...
        return True

    def rewire(self, new_node):
        for node in self.nodes:
            if node == new_node:
                continue
            if not self.is_collision_free(node, new_node):
                continue
            cost = self.costs[self.node_to_key(node)] + self.distance(node, new_node)
            if cost < self.costs[self.node_to_key(new_node)]:
                self.edges[self.node_to_key(new_node)] = node
                self.costs[self.node_to_key(new_node)] = cost

    def find_path(self):
        # Create an OMPL state space with three dimensions
        space = oc.RealVectorStateSpace(3)

        # Set the bounds of the state space to be centered at the origin with a size of self.area_size
        bounds = oc.RealVectorBounds(3)
        bounds.setLow(-self.area_size / 2)
        bounds.setHigh(self.area_size / 2)
        bounds.setLow(2, -180)  # Set the lower bound of the third dimension to -180
        bounds.setHigh(2, 180)  # Set the upper bound of the third dimension to 180
        space.setBounds(bounds)

        # Create an OMPL control space with two dimensions
        control_space = oc.RealVectorControlSpace(space, 2)

        # Set the bounds of the control space to be centered at the origin with a size of 1.0
        control_bounds = oc.RealVectorBounds(2)
        control_bounds.setLow(-1.0)
        control_bounds.setHigh(1.0)
        control_space.setBounds(control_bounds)

        # Set the steer function of the control space to an instance of MySteerFunction
        steer_fn = MySteerFunction(control_space)
        control_space.setSteerFunction(steer_fn)

        # Create an OMPL state validity checker
        state_validator = ValidityChecker(self.obstacles)

        # Create an OMPL state space information object
        si = oc.SpaceInformation(space, control_space)
        si.setStateValidityChecker(state_validator)

        # Create an OMPL problem definition with the start and goal states
        start_state = oc.RealVectorStateSpace.StateType(3)
        start_state.values = self.start
        goal_state = oc.RealVectorStateSpace.StateType(3)
        goal_state.values = self.goal
        pdef = oc.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)

        # Set the optimization objective to minimize the path length
        objective = oc.PathLengthOptimizationObjective(si)
        pdef.setOptimizationObjective(objective)

        # Create an RRT* planner and set its parameters
        planner = oc.RRTstar(si)
        planner.setGoalBias(self.goal_sample_rate)
        planner.setRange(self.step_size)
        planner.setMaxIterations(self.max_iter)

        # Set the problem definition and run the planner
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(1.0)

        # Extract the solution path from the problem definition
        if solved:
            path = pdef.getSolutionPath()
            path.interpolate(100)
            states = path.getStates()
            path = [[state[0], state[1], state[2]] for state in states]
            return path
        else:
            return None